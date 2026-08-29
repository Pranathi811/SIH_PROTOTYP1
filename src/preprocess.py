"""Time-series preprocessing pipeline for ESA-ADB Mission 1 dataset.

Implements FR-03 (FR-03A through FR-03D) requirements.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from .data_loader import CHANNELS, load_dataset
from .validate import validate_dataset


def preprocess_data(
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    save_processed: bool = True,
    output_dir: str | Path = "data/processed",
    models_dir: str | Path = "models"
) -> Dict[str, Any]:
    """
    Execute full preprocessing pipeline:
    1. Validation checks on raw data (FR-03A, FR-03B)
    2. Respect strict train/test partition (FR-03C)
    3. Fit StandardScaler on train.csv and transform test.csv (FR-03D)
    4. Save processed datasets and scaler artifact
    
    Returns:
        dict containing:
            - scaler: Fitted StandardScaler
            - X_train: numpy array of scaled training features (4000, 5)
            - X_test: numpy array of scaled test features (3000, 5)
            - y_test: numpy array of test ground truth labels (3000,)
            - train_df: DataFrame of scaled train data with timestamps
            - test_df: DataFrame of scaled test data with timestamps and labels
    """
    if frames is None:
        frames = load_dataset()
        
    validation = validate_dataset(frames)
    if not validation["is_valid"]:
        raise ValueError(f"Dataset validation failed: {'; '.join(validation['errors'])}")
        
    train_df = frames["train"].copy()
    test_df = frames["test"].copy()
    
    # Extract raw channel features
    X_train_raw = train_df[CHANNELS].to_numpy(dtype=np.float64)
    X_test_raw = test_df[CHANNELS].to_numpy(dtype=np.float64)
    y_test = test_df["label"].to_numpy(dtype=np.int64)
    
    # Fit scaler strictly on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Create processed DataFrames
    train_processed = pd.DataFrame(X_train_scaled, columns=CHANNELS)
    train_processed.insert(0, "timestamp", train_df["timestamp"])
    
    test_processed = pd.DataFrame(X_test_scaled, columns=CHANNELS)
    test_processed.insert(0, "timestamp", test_df["timestamp"])
    test_processed["label"] = y_test
    
    if save_processed:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        train_processed.to_csv(out_path / "train_processed.csv", index=False)
        test_processed.to_csv(out_path / "test_processed.csv", index=False)
        
        m_path = Path(models_dir)
        m_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, m_path / "scaler.pkl")
        print(f"[OK] Processed data saved to {out_path} and scaler to {m_path / 'scaler.pkl'}")
        
    return {
        "scaler": scaler,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "train_df": train_processed,
        "test_df": test_processed
    }


if __name__ == "__main__":
    preprocess_data()
