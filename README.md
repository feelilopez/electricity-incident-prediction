# Electricity Incident Prediction

Univariate time-series project to predict whether an electricity demand incident will occur in the next `n` steps, using the last `p` steps.

## 1) Problem Definition

- Objective: Predict if an incident happens in the next `n` time steps.
- Input: Last `p` demand values from REE demand series.
- Output: Binary label (`1` incident, `0` non-incident).
- Incident definition: Rolling Seasonal Threshold (adaptive threshold based on historical behavior).

This project uses REE API data from 2021-2025 to reduce COVID-period distortions.

## 2) Scope (Fast Delivery)

- Keep data univariate (demand only) for the first version.
- Use hourly frequency.
- Start with:
	- `p = 168` (last 7 days)
	- `n = 24` (next 24 hours)
- Train two models:
	- Baseline: Logistic Regression
	- Stronger baseline: Random Forest

## 3) Repository Structure

```
data/
	raw/                 # Data pulled from REE API
	processed/           # Cleaned and model-ready datasets
notebooks/             # EDA and results storytelling
src/
	data/                # Download and dataset assembly
	features/            # Labeling and sliding-window utilities
	models/              # Training scripts
	evaluation/          # Metrics and analysis helpers
configs/               # Experiment settings
reports/               # Figures and short result reports
```

## 4) Pipeline

1. Download REE demand data (`/demanda/evolucion`) across time slices.
2. Merge and clean into a regular hourly series.
3. Build incident labels with rolling seasonal threshold.
4. Create supervised samples with sliding window (`X`: last `p`, `y`: any incident in next `n`).
5. Split chronologically (train/validation/test).
6. Train baseline models and evaluate with incident-focused metrics.
7. Analyze errors and threshold trade-offs.

## 5) Suggested Time Split

- Train: 2021-2023
- Validation: 2024
- Test: 2025

Use strict chronological order to avoid leakage.

## 6) Metrics

Focus on metrics robust to class imbalance:

- PR-AUC
- Recall (incident class)
- Precision (incident class)
- F1 (incident class)
- Confusion matrix

Accuracy is secondary.

## 7) Quick Start

1. Create and activate your Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run dataset creation:

```bash
python -m src.data.make_dataset
```

To exclude known anomaly windows (for example, the Iberia outage), set
`data.excluded_periods` in `configs/experiment.yaml` with `start` and `end`
timestamps. These ranges are removed before hourly interpolation and
window/label generation.

4. Train baseline models:

```bash
python -m src.models.train_baseline
```

5. Open notebooks for EDA and result analysis.

## 8) Important Risks and Mitigations

- Label noise: Tune rolling threshold parameters with validation split.
- Class imbalance: Use class weights and threshold tuning.
- Leakage: Fit preprocessing on train data only.
- Missing timestamps: Enforce regular frequency and log imputations.

## 9) Next Steps

- Implement robust API fetching with retries and time slicing.
- Generate first labeled dataset.
- Train Logistic Regression baseline and calibrate decision threshold.
- Compare against Random Forest.
- Document findings and limitations in a short report.

