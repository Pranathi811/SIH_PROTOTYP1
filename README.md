# Satellite Health Monitoring

ESA-ADB Prototype Mission 1 dashboard and API built from the supplied synthetic dataset. The dataset is synthetic and must not be presented as official ESA-ADB benchmark data.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Start the API separately with:

```powershell
uvicorn api:app --reload
```

API documentation is available at `http://localhost:8000/docs`.

## Substations

Substation metadata is configured in `config/substations.json`. Each substation can define its own display name for `channel_1` through `channel_5`, and the dashboard/API accept `substation_id`. The supplied synthetic archive has no substation column or physical data dictionary, so the included parameter names are examples only; replace them with approved mappings and separate site data before treating them as real measurements.

## Model artifacts

Run `python train.py` to persist the fitted scaler, Isolation Forest, and reconstruction autoencoder under `models/`. The dashboard and API train in memory with caching so they work immediately after installation.

## Dataset

The archive contents are stored under `data/raw/`: 4,000 training records, 3,000 test records, 7,000 combined telemetry/label records, and 270 anomaly-type annotations. Channels remain named `channel_1` through `channel_5` because the supplied data has no physical data dictionary.
