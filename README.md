# Electricity Incident Prediction

Brief project to detect short-term high-demand incidents using REE demand data (2021–2025). 

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

3. Explore the 'EDA' and 'results' notebooks in the `notebooks/` folder.

## Future work

- Try more advanced models.
- Turn univariate time-series into multivariate by adding features (weather, calendar, holidays).
- Packaging for periodic runs and simple deployment.

For details and experiments, see the `notebooks/` folder.