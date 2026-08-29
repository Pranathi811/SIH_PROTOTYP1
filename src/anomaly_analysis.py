"""Anomaly type breakdown and diagnostic analysis module.

Implements FR-21 and supporting analytics for dashboard and API.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from .data_loader import CHANNELS


def get_anomaly_type_counts(anomaly_types_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return count of each synthetic anomaly category from anomaly_types.csv."""
    counts = anomaly_types_df["anomaly_type"].value_counts().to_dict()
    categories = ["point", "collective", "level_shift", "trend"]
    return [
        {"type": cat, "count": int(counts.get(cat, 0))}
        for cat in categories
    ]


def analyze_channel_contributions(
    test_df: pd.DataFrame,
    scaler_means: np.ndarray,
    scaler_scales: np.ndarray
) -> pd.DataFrame:
    """Analyze per-channel mean z-score magnitudes during anomaly intervals vs normal."""
    res = []
    normal_df = test_df[test_df["label"] == 0]
    anomaly_df = test_df[test_df["label"] == 1]
    
    for i, ch in enumerate(CHANNELS):
        mean = scaler_means[i]
        scale = scaler_scales[i]
        
        norm_z = np.abs((normal_df[ch] - mean) / scale).mean() if len(normal_df) > 0 else 0
        anom_z = np.abs((anomaly_df[ch] - mean) / scale).mean() if len(anomaly_df) > 0 else 0
        
        res.append({
            "channel": ch,
            "normal_mean_abs_z": round(float(norm_z), 3),
            "anomaly_mean_abs_z": round(float(anom_z), 3),
            "z_score_divergence": round(float(anom_z - norm_z), 3)
        })
    return pd.DataFrame(res)


def get_anomaly_intervals(test_df: pd.DataFrame, label_col: str = "label") -> List[Dict[str, Any]]:
    """Group contiguous anomaly timestamps into discrete anomaly events/intervals."""
    df = test_df.sort_values("timestamp").reset_index(drop=True)
    intervals = []
    
    in_interval = False
    start_time = None
    count = 0
    
    for idx, row in df.iterrows():
        is_anom = (row[label_col] == 1)
        if is_anom and not in_interval:
            in_interval = True
            start_time = row["timestamp"]
            count = 1
        elif is_anom and in_interval:
            count += 1
        elif not is_anom and in_interval:
            in_interval = False
            end_time = df.loc[idx - 1, "timestamp"]
            intervals.append({
                "start_time": start_time,
                "end_time": end_time,
                "duration_minutes": count,
                "point_count": count
            })
            
    if in_interval:
        intervals.append({
            "start_time": start_time,
            "end_time": df.iloc[-1]["timestamp"],
            "duration_minutes": count,
            "point_count": count
        })
        
    return intervals
