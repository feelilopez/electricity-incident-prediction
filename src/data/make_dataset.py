from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.data.ree_client import REEClient, month_ranges
from src.features.labeling import rolling_seasonal_threshold_labels
from src.features.windowing import build_supervised_windows

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
RAW_OUT = ROOT / "data" / "raw" / "demand_2021_2025.csv"
PROCESSED_OUT = ROOT / "data" / "processed" / "supervised_2021_2025.csv"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_full_range(start_date: str, end_date: str) -> pd.DataFrame:
    client = REEClient()
    chunks = []

    # Download in month-sized chunks and concatenate into a single time series.
    # month_ranges yields tuples of (chunk_start, chunk_end) for each month in the range.
    for chunk_start, chunk_end in month_ranges(start_date, end_date):
        print(f"Fetching demand from {chunk_start} to {chunk_end}...")
        # make request for this chunk
        frame = client.fetch_demand(
            start_date=chunk_start.isoformat(),
            end_date=chunk_end.isoformat(),
            time_trunc="hour",
        )
        chunks.append(frame)

    # join all chunks
    joined = pd.concat(chunks, axis=0, ignore_index=True)
    joined = joined.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    return joined


def clean_hourly(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    data = frame.copy()
    # Convert to Spain timezone before reindexing to maintain local-hour seasonality.
    data["datetime"] = pd.to_datetime(data["datetime"], utc=True).dt.tz_convert(
        timezone
    )
    data = data.set_index("datetime").sort_index()

    full_index = pd.date_range(
        data.index.min(), data.index.max(), freq="h", tz=timezone
    )
    data = data.reindex(full_index)
    # Interpolation keeps continuity; ffill/bfill handles edge gaps.
    data["demand_mw"] = data["demand_mw"].interpolate(method="time").ffill().bfill()
    data = data.reset_index().rename(columns={"index": "datetime"})
    return data


def main() -> None:
    cfg = load_config()

    demand_cfg = cfg["data"]
    label_cfg = cfg["labeling"]
    win_cfg = cfg["windowing"]

    raw = fetch_full_range(demand_cfg["start_date"], demand_cfg["end_date"])
    cleaned = clean_hourly(raw, timezone=demand_cfg["timezone"])

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(RAW_OUT, index=False)

    s = cleaned.set_index("datetime")["demand_mw"]
    # Labels are built with a causal method (uses only past data).
    labels = rolling_seasonal_threshold_labels(
        s,
        history_weeks=label_cfg["seasonal_history_weeks"],
        zscore_k=label_cfg["zscore_k"],
        fallback_window=label_cfg["fallback_window_hours"],
    )

    # Report positive rate for raw rolling-threshold labels (before windowing)
    label_positive_count = int(labels.sum())
    label_total = len(labels)
    label_positive_rate = float(labels.mean()) if label_total else 0.0
    print(
        f"Rolling-threshold labels — Positives: {label_positive_count}/{label_total} "
        f"| Positive rate: {label_positive_rate:.4f}"
    )

    # Each sample uses last p lags and predicts incident occurrence in next n steps.
    supervised = build_supervised_windows(
        demand=s,
        incident_label=labels,
        past_steps=win_cfg["past_steps"],
        horizon_steps=win_cfg["horizon_steps"],
    )

    # Persist prediction start timestamps in UTC to avoid DST-related ambiguity
    # when reloading the dataset for splitting and training.
    supervised["timestamp_target_start"] = pd.to_datetime(
        supervised["timestamp_target_start"], utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    PROCESSED_OUT.parent.mkdir(parents=True, exist_ok=True)
    supervised.to_csv(PROCESSED_OUT, index=False)

    positive_rate = float(supervised["target"].mean()) if len(supervised) else 0.0
    print(f"Saved raw demand to {RAW_OUT}")
    print(f"Saved supervised dataset to {PROCESSED_OUT}")
    print(f"Rows: {len(supervised)} | Positive rate: {positive_rate:.4f}")


if __name__ == "__main__":
    main()
