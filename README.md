# Electricity Incident Prediction

This project detects high-demand electricity incidents using REE demand data (2021–2025). Using the previous week’s demand, the model predicts whether a high-demand incident will occur in the next few hours. Positive predictions act as alerts that enable proactive resource planning and help prevent possible outages on the grid.


Data: raw pulls and processed datasets live under `data/` (see `data/processed/supervised_2021_2025.csv`). Source: [REE API](https://www.ree.es/en/datos/apidata). 

## Repository structure

- `data/` — raw and processed datasets
- `notebooks/` — EDA and result notebooks
- `src/` — data ingestion, labeling, feature and model code
- `configs/` — experiment settings

## Requirements

- See `requirements.txt` for Python dependencies.

## Quick start

1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Build the dataset (produces files in `data/processed/`):

```bash
python -m src.data.make_dataset
```

3. Explore the '01_EDA' and '02_Results' notebooks in the `notebooks/` folder.

## Future work

- Try more advanced models.
- Turn univariate time-series into multivariate by adding features (weather, calendar, holidays).
- Package for periodic runs and simple deployment.

For details, see the `notebooks/` folder.