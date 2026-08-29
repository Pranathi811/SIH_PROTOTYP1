"""Data loader module for ESA-ADB Mission 1 synthetic dataset."""

from pathlib import Path
import pandas as pd

CHANNELS = [f"channel_{i}" for i in range(1, 6)]
DATA_FILES = ["train.csv", "test.csv", "telemetry.csv", "labels.csv", "anomaly_types.csv"]


def load_dataset(data_dir: str | Path = "data/raw") -> dict[str, pd.DataFrame]:
    """
    Load all five dataset CSV files with parsed timestamps and chronological order.
    
    Args:
        data_dir: Directory containing raw CSV files.
        
    Returns:
        dict: Mapping of dataset name (e.g., 'train', 'test', 'labels', 'anomaly_types', 'telemetry')
              to clean pandas DataFrames.
    """
    root = Path(data_dir)
    missing = [name for name in DATA_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files in {root}: {', '.join(missing)}")
    
    frames = {name.removesuffix(".csv"): pd.read_csv(root / name) for name in DATA_FILES}
    
    for name, frame in frames.items():
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
            frame.sort_values("timestamp", inplace=True)
            frame.reset_index(drop=True, inplace=True)
            
    return frames
