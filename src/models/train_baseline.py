from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
DATA_PATH = ROOT / "data" / "processed" / "supervised_2021_2025.csv"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_time(
    frame: pd.DataFrame, train_end: str, valid_end: str, timezone: str
):
    # Chronological split avoids training on future information.
    # Timestamp column is parsed as UTC, while split boundaries are configured as
    # local time (Europe/Madrid). Convert boundaries to UTC for robust comparison.
    ts = pd.to_datetime(frame["timestamp_target_start"], utc=True)
    train_end_utc = pd.Timestamp(train_end).tz_localize(timezone).tz_convert("UTC")
    valid_end_utc = pd.Timestamp(valid_end).tz_localize(timezone).tz_convert("UTC")

    train_mask = ts <= train_end_utc
    valid_mask = (ts > train_end_utc) & (ts <= valid_end_utc)
    test_mask = ts > valid_end_utc

    return frame[train_mask], frame[valid_mask], frame[test_mask]


def pick_threshold_max_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    # Convert probabilities to class labels using a validation-optimized threshold.
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5

    f1 = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx])


def evaluate(
    name: str, y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> None:
    y_pred = (y_proba >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, y_proba)

    print("=" * 70)
    print(f"{name}")
    print(f"PR-AUC: {pr_auc:.4f} | Threshold: {threshold:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4))


def main() -> None:
    cfg = load_config()
    split_cfg = cfg["split"]
    model_cfg = cfg["model"]

    data = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in data.columns if c.startswith("lag_")]

    train_df, valid_df, test_df = split_time(
        data,
        train_end=split_cfg["train_end"],
        valid_end=split_cfg["valid_end"],
        timezone=cfg["data"]["timezone"],
    )

    x_train, y_train = train_df[feature_cols].to_numpy(), train_df["target"].to_numpy()
    x_valid, y_valid = valid_df[feature_cols].to_numpy(), valid_df["target"].to_numpy()
    x_test, y_test = test_df[feature_cols].to_numpy(), test_df["target"].to_numpy()

    # Baseline 1: linear model over standardized lag features.
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=model_cfg["random_state"],
                ),
            ),
        ]
    )
    lr.fit(x_train, y_train)

    valid_proba_lr = lr.predict_proba(x_valid)[:, 1]
    test_proba_lr = lr.predict_proba(x_test)[:, 1]
    best_thr_lr = pick_threshold_max_f1(y_valid, valid_proba_lr)
    evaluate("Logistic Regression (test)", y_test, test_proba_lr, best_thr_lr)

    # Baseline 2: non-linear tree ensemble.
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=model_cfg["random_state"],
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)

    valid_proba_rf = rf.predict_proba(x_valid)[:, 1]
    test_proba_rf = rf.predict_proba(x_test)[:, 1]
    best_thr_rf = pick_threshold_max_f1(y_valid, valid_proba_rf)
    evaluate("Random Forest (test)", y_test, test_proba_rf, best_thr_rf)


if __name__ == "__main__":
    main()
