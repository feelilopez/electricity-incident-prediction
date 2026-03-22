from __future__ import annotations

import pandas as pd


def build_supervised_windows(
    demand: pd.Series,
    incident_label: pd.Series,
    past_steps: int,
    horizon_steps: int,
) -> pd.DataFrame:
    """
    Build a supervised learning dataset with lag features and binary incident targets.
    For each timestamp t, use the past `past_steps` demand values as features and
    predict whether any incident occurs in the next `horizon_steps` hours.
    """

    rows = []
    total = len(demand)

    # t is the prediction start time: use [t-p, t) as features and [t, t+n) for target.
    for t in range(past_steps, total - horizon_steps):
        past_window = demand.iloc[t - past_steps : t].to_numpy(dtype=float)
        future_window = incident_label.iloc[t : t + horizon_steps].to_numpy(dtype=int)
        # Positive target if any incident appears within the prediction horizon.
        target = int(future_window.max() > 0)

        row = {f"lag_{k + 1}": past_window[-(k + 1)] for k in range(past_steps)}
        row["target"] = target
        row["timestamp_target_start"] = demand.index[t]
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("timestamp_target_start").reset_index(drop=True)
    return frame
