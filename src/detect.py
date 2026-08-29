"""Inference and anomaly detection engine for satellite telemetry.

Implements FR-11, FR-12, and FR-13.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_loader import CHANNELS


def compute_anomaly_scores(
    model: Any,
    model_type: str,
    X_scaled: np.ndarray
) -> np.ndarray:
    """
    Compute raw anomaly scores for scaled telemetry features.
    Higher score indicates higher likelihood of anomaly.
    """
    if model_type == "isolation_forest":
        # -score_samples returns higher value for anomalies
        raw_scores = -model.score_samples(X_scaled)
        return raw_scores
    elif model_type == "autoencoder":
        if hasattr(model, "get_reconstruction_errors"):
            errors = model.get_reconstruction_errors(X_scaled)
        elif hasattr(model, "predict"):
            recon = model.predict(X_scaled)
            errors = np.mean((X_scaled - recon) ** 2, axis=1)
        else:
            raise ValueError("Unsupported autoencoder model type.")
        return errors
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def calibrate_threshold(
    model: Any,
    model_type: str,
    X_train_scaled: np.ndarray,
    percentile: float = 98.0
) -> float:
    """Calibrate anomaly score decision threshold based on training set quantile."""
    train_scores = compute_anomaly_scores(model, model_type, X_train_scaled)
    threshold = float(np.percentile(train_scores, percentile))
    return threshold


def predict_anomalies(
    scores: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Convert continuous anomaly scores into binary labels (0 = Normal, 1 = Anomaly)."""
    return (scores >= threshold).astype(int)


def calculate_channel_deviations(
    scaler: StandardScaler,
    raw_values: Union[List[float], np.ndarray]
) -> List[Dict[str, Any]]:
    """
    Calculate standardized z-scores and absolute deviations for each telemetry channel.
    Useful for root cause diagnostics on anomalous readings.
    """
    values_arr = np.array(raw_values, dtype=np.float64).reshape(1, -1)
    scaled = scaler.transform(values_arr)[0]
    
    deviations = []
    for ch, val, z in zip(CHANNELS, values_arr[0], scaled):
        deviations.append({
            "channel": ch,
            "raw_value": float(val),
            "z_score": float(z),
            "absolute_z_score": float(abs(z)),
            "is_deviating": bool(abs(z) >= 2.5)
        })
    return deviations


def detect_single_reading(
    reading: Dict[str, Any],
    models: Dict[str, Any],
    thresholds: Dict[str, float],
    scaler: StandardScaler,
    primary_model: str = "isolation_forest"
) -> Dict[str, Any]:
    """Perform real-time detection on a single telemetry dictionary."""
    raw_vals = [float(reading[ch]) for ch in CHANNELS]
    scaled_vals = scaler.transform([raw_vals])
    
    model = models[primary_model]
    threshold = thresholds[primary_model]
    
    score = float(compute_anomaly_scores(model, primary_model, scaled_vals)[0])
    is_anomaly = bool(score >= threshold)
    
    deviations = calculate_channel_deviations(scaler, raw_vals)
    strongest_channel = max(deviations, key=lambda x: x["absolute_z_score"])
    
    return {
        "timestamp": reading.get("timestamp"),
        "anomaly": is_anomaly,
        "anomaly_score": round(score, 4),
        "status": "Anomaly" if is_anomaly else "Normal",
        "primary_model": primary_model,
        "threshold": round(threshold, 4),
        "strongest_channel": strongest_channel["channel"],
        "channel_deviations": deviations
    }
