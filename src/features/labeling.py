from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_seasonal_threshold_labels(
    series: pd.Series,
    history_weeks: int = 8,
    zscore_k: float = 3.0,
    fallback_window: int = 168,
) -> pd.Series:
    """
    Label incidents using a causal seasonal threshold.

    For each timestamp t, compare demand[t] against historical values observed at
    the same hour-of-week during previous weeks. If there is insufficient history,
    fallback to a causal rolling mean + k * std threshold.
    """
    values = series.astype(float).copy()
    labels = pd.Series(0, index=values.index, dtype=int)

    # Seasonal context: compare each point with same day-of-week and hour.
    dow = values.index.dayofweek
    hour = values.index.hour

    for i in range(len(values)):
        current = values.iloc[i]
        hist_start = max(0, i - history_weeks * 7 * 24)
        history = values.iloc[hist_start:i]

        # Find historical values at the same hour-of-week (e.g. all past Mondays at 3pm).
        same_slot = history[
            (dow[hist_start:i] == dow[i]) & (hour[hist_start:i] == hour[i])
        ]

        # Require at least 4 past same-slot points (2021-2025) for a robust seasonal threshold.
        if len(same_slot) >= 4:
            # Robust baseline via median + MAD to reduce influence of spikes.
            baseline = float(np.median(same_slot))
            mad = float(np.median(np.abs(same_slot - baseline)))
            robust_sigma = 1.4826 * mad
            threshold = baseline + zscore_k * max(robust_sigma, 1e-6)

        # Otherwise, fallback to a causal rolling mean + k * std threshold.
        else:
            fallback = values.iloc[max(0, i - fallback_window) : i]
            if len(fallback) < 8:
                continue
            threshold = float(fallback.mean() + zscore_k * fallback.std(ddof=0))

        # Incident if current demand exceeds dynamic threshold.
        labels.iloc[i] = int(current > threshold)

    return labels
