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
    fbeta_score,
    make_scorer,
    precision_score,
    precision_recall_curve,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
DATA_PATH = ROOT / "data" / "processed" / "supervised_2021_2025.csv"


def load_config() -> dict:
    # Centralized experiment settings (time split and model parameters).
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_time(frame: pd.DataFrame, train_end: str, valid_end: str, timezone: str):
    # Chronological split avoids training on future information: (2021-2023 for training, 2024 for validation, 2025 for testing).
    # Convert boundaries to UTC for robust comparison.
    ts = pd.to_datetime(frame["timestamp_target_start"], utc=True)
    train_end_utc = pd.Timestamp(train_end).tz_localize(timezone).tz_convert("UTC")
    valid_end_utc = pd.Timestamp(valid_end).tz_localize(timezone).tz_convert("UTC")

    train_mask = ts <= train_end_utc
    valid_mask = (ts > train_end_utc) & (ts <= valid_end_utc)
    test_mask = ts > valid_end_utc

    return frame[train_mask], frame[valid_mask], frame[test_mask]


def pick_threshold_max_fbeta(
    y_true: np.ndarray, y_proba: np.ndarray, beta: float = 2.0
) -> float:
    # Convert probabilities to class labels using a validation-optimized threshold.
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5

    beta_sq = beta * beta
    fbeta = ((1 + beta_sq) * precision[:-1] * recall[:-1]) / (
        beta_sq * precision[:-1] + recall[:-1] + 1e-12
    )
    best_idx = int(np.argmax(fbeta))
    return float(thresholds[best_idx])


def build_threshold_tradeoff(
    y_true: np.ndarray, y_proba: np.ndarray, beta: float = 2.0
) -> pd.DataFrame:
    # Each row represents one candidate threshold and its precision/recall/F-beta.
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return pd.DataFrame(
            [{"threshold": 0.5, "precision": 0.0, "recall": 0.0, f"f{beta:g}": 0.0}]
        )

    beta_sq = beta * beta
    fbeta = ((1 + beta_sq) * precision[:-1] * recall[:-1]) / (
        beta_sq * precision[:-1] + recall[:-1] + 1e-12
    )
    return (
        pd.DataFrame(
            {
                "threshold": thresholds,
                "precision": precision[:-1],
                "recall": recall[:-1],
                f"f{beta:g}": fbeta,
            }
        )
        .sort_values("threshold")
        .reset_index(drop=True)
    )


def pick_threshold_with_min_precision(
    y_true: np.ndarray, y_proba: np.ndarray, min_precision: float, beta: float = 2.0
) -> float:
    # Operational rule: satisfy minimum precision, then retain maximum recall.
    tradeoff = build_threshold_tradeoff(y_true, y_proba, beta=beta)
    feasible = tradeoff[tradeoff["precision"] >= min_precision]

    if feasible.empty:
        # If target precision is impossible, fall back to best F-beta.
        return pick_threshold_max_fbeta(y_true, y_proba, beta=beta)

    # Among points that satisfy precision floor, keep as much recall as possible.
    best_idx = feasible["recall"].idxmax()
    return float(feasible.loc[best_idx, "threshold"])


def print_threshold_scenarios(
    y_true: np.ndarray, y_proba: np.ndarray, min_precision_targets: list[float]
) -> None:
    # Quick table to visualize how recall drops as precision constraints get stricter.
    print(
        "Validation threshold scenarios (higher threshold usually increases precision):"
    )
    print("min_precision | threshold | precision | recall")
    for min_p in min_precision_targets:
        thr = pick_threshold_with_min_precision(
            y_true, y_proba, min_precision=min_p, beta=2.0
        )
        y_pred = (y_proba >= thr).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        print(f"{min_p:12.2f} | {thr:9.4f} | {p:9.4f} | {r:6.4f}")


def tune_model(
    estimator,
    param_distributions: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    n_iter: int = 12,
):
    # Use only historical folds while tuning to avoid leaking future information.
    n_splits = 4
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scorer = make_scorer(fbeta_score, beta=2)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=tscv,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate(
    name: str, y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> None:
    # Convert probabilities into class labels using the chosen operating threshold.
    y_pred = (y_proba >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    print("=" * 70)
    print(f"{name}")
    print(
        f"PR-AUC: {pr_auc:.4f} | Threshold: {threshold:.4f} | "
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | F2: {f2:.4f}"
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4))


def main() -> None:
    # 1) Read configuration and dataset.
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
    # Train on history, pick threshold on validation, report once on test.

    # Baseline 1: linear model over standardized lag features.
    lr_base = Pipeline(
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
    lr_param_dist = {
        "clf__C": [0.01, 0.1, 1.0, 5.0, 10.0, 20.0],
        "clf__solver": ["lbfgs", "liblinear"],
    }
    lr, lr_best_params, lr_best_score = tune_model(
        estimator=lr_base,
        param_distributions=lr_param_dist,
        x_train=x_train,
        y_train=y_train,
        random_state=model_cfg["random_state"],
        n_iter=8,
    )
    print("-" * 70)
    print("Best Logistic Regression CV-F2:", f"{lr_best_score:.4f}")
    print("Best Logistic Regression params:", lr_best_params)

    # Validation probabilities are used only to choose an operating threshold.
    valid_proba_lr = lr.predict_proba(x_valid)[:, 1]
    test_proba_lr = lr.predict_proba(x_test)[:, 1]
    # Simple operating policy: enforce a minimum precision, then maximize recall.
    min_precision_lr = 0.35
    print_threshold_scenarios(
        y_valid, valid_proba_lr, min_precision_targets=[0.30, 0.35, 0.40]
    )
    best_thr_lr = pick_threshold_with_min_precision(
        y_valid, valid_proba_lr, min_precision=min_precision_lr, beta=2.0
    )
    print(
        f"Selected LR threshold with min precision {min_precision_lr:.2f}: {best_thr_lr:.4f}"
    )
    evaluate("Logistic Regression (test)", y_test, test_proba_lr, best_thr_lr)

    # Baseline 2: non-linear tree ensemble. No scale needed for tree-based models.
    rf_base = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=model_cfg["random_state"],
        n_jobs=-1,
    )
    rf_param_dist = {
        "n_estimators": [200, 300, 400],
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features": ["sqrt", "log2", None],
    }
    rf, rf_best_params, rf_best_score = tune_model(
        estimator=rf_base,
        param_distributions=rf_param_dist,
        x_train=x_train,
        y_train=y_train,
        random_state=model_cfg["random_state"],
        n_iter=12,
    )
    print("-" * 70)
    print("Best Random Forest CV-F2:", f"{rf_best_score:.4f}")
    print("Best Random Forest params:", rf_best_params)

    # Same threshold-selection policy for RF, so model comparisons stay fair.
    valid_proba_rf = rf.predict_proba(x_valid)[:, 1]
    test_proba_rf = rf.predict_proba(x_test)[:, 1]
    min_precision_rf = 0.35
    print_threshold_scenarios(
        y_valid,
        valid_proba_rf,
        min_precision_targets=[0.30, 0.35, 0.40],
    )
    best_thr_rf = pick_threshold_with_min_precision(
        y_valid,
        valid_proba_rf,
        min_precision=min_precision_rf,
        beta=2.0,
    )
    print(
        f"Selected RF threshold with min precision {min_precision_rf:.2f}: {best_thr_rf:.4f}"
    )
    evaluate("Random Forest (test)", y_test, test_proba_rf, best_thr_rf)


if __name__ == "__main__":
    main()
