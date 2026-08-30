"""FastAPI REST interface for telemetry anomaly detection."""

from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_loader import CHANNELS, load_dataset
from src.models import build_models, channel_deviations
from src.substations import load_substations

app = FastAPI(title="Satellite Health Monitoring API", version="1.0.0")
frames = load_dataset()
models = build_models(frames)
substations = load_substations()


class TelemetryReading(BaseModel):
    substation_id: str = "substation_alpha"
    timestamp: datetime
    channel_1: float
    channel_2: float
    channel_3: float
    channel_4: float
    channel_5: float


class AnalysisRequest(BaseModel):
    start_time: datetime
    end_time: datetime
    parameters: list[str] = Field(default_factory=lambda: CHANNELS)


@app.get("/health")
def health():
    return {"status": "healthy", "dataset": "ESA-ADB Mission 1 synthetic", "substations": len(substations)}


@app.get("/substations")
def list_substations():
    return [{"id": substation_id, **config} for substation_id, config in substations.items()]


@app.post("/detect-anomaly")
def detect_anomaly(reading: TelemetryReading):
    if reading.substation_id not in substations:
        raise HTTPException(status_code=400, detail=f"Unknown substation: {reading.substation_id}")
    values = [[getattr(reading, channel) for channel in CHANNELS]]
    scaled = models["scaler"].transform(values)
    score = float(-models["isolation_forest"].score_samples(scaled)[0])
    anomaly = score >= models["thresholds"]["isolation_forest"]
    deviations = channel_deviations(models, values[0])
    strongest = max(deviations, key=lambda item: item["absolute_z_score"])
    return {"substation_id": reading.substation_id, "timestamp": reading.timestamp, "anomaly": bool(anomaly), "anomaly_score": score,
            "status": "Anomaly" if anomaly else "Normal", "strongest_channel": strongest["channel"],
            "channel_deviations": deviations}


@app.post("/analyze-telemetry")
def analyze_telemetry(request: AnalysisRequest):
    unsupported = set(request.parameters) - set(CHANNELS)
    if unsupported:
        raise HTTPException(status_code=400, detail=f"Unsupported parameters: {sorted(unsupported)}")
    selected = frames["telemetry"].query("@request.start_time <= timestamp <= @request.end_time")
    values = selected[CHANNELS].to_numpy()
    scores = -models["isolation_forest"].score_samples(models["scaler"].transform(values))
    flagged = scores >= models["thresholds"]["isolation_forest"]
    return {"records_analyzed": len(selected), "anomalies_detected": int(flagged.sum()), "anomaly_percentage": round(float(flagged.mean() * 100), 2) if len(flagged) else 0, "highest_anomaly_score": round(float(scores.max()), 4) if len(scores) else 0}


@app.get("/anomaly-types")
def anomaly_types():
    return [{"type": name, "count": int(count)} for name, count in frames["anomaly_types"]["anomaly_type"].value_counts().items()]


@app.get("/model-performance")
def model_performance():
    """Get ML model performance metrics (precision, recall, F1-score)."""
    performance = {}
    for model_name, metrics in models["metrics"].items():
        performance[model_name] = {
            "f1_score": round(float(metrics["f1_score"]), 4),
            "precision": round(float(metrics["precision"]), 4),
            "recall": round(float(metrics["recall"]), 4),
            "description": {
                "f1_score": "Overall model accuracy (0-1, higher is better)",
                "precision": "Accuracy of positive predictions (fewer false alarms)",
                "recall": "Percentage of actual anomalies detected (doesn't miss)"
            }
        }
    return {
        "models": performance,
        "best_model": max(models["metrics"].items(), key=lambda x: x[1]["f1_score"])[0],
        "summary": f"Isolation Forest F1: {models['metrics']['isolation_forest']['f1_score']:.2%} | Autoencoder F1: {models['metrics']['autoencoder']['f1_score']:.2%}"
    }
