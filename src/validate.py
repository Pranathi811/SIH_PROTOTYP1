"""Dataset validation module for ESA-ADB Mission 1 synthetic dataset.

Implements FR-02 (FR-02A through FR-02H) validation checks.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from .data_loader import CHANNELS

ALLOWED_ANOMALY_TYPES = {"point", "collective", "level_shift", "trend"}


def validate_dataset(frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Run comprehensive validation checks on all loaded dataset frames.
    
    Args:
        frames: Dictionary of DataFrames loaded by data_loader.load_dataset()
        
    Returns:
        dict containing:
            - is_valid (bool): True if all checks pass
            - errors (list[str]): List of fatal error messages
            - warnings (list[str]): List of non-fatal warnings
            - summary (dict): Validation metrics and counts
    """
    errors: List[str] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {}

    # Check required dataset keys
    required_keys = {"train", "test", "telemetry", "labels", "anomaly_types"}
    missing_keys = required_keys - set(frames.keys())
    if missing_keys:
        errors.append(f"Missing required dataset partitions: {', '.join(sorted(missing_keys))}")
        return {"is_valid": False, "errors": errors, "warnings": warnings, "summary": summary}

    # FR-02A, FR-02B, FR-02C, FR-02D, FR-02E, FR-02F checks for train, test, telemetry
    for name in ("train", "test", "telemetry"):
        df = frames[name]
        summary[f"{name}_record_count"] = len(df)
        
        # FR-02A: Check required columns
        required_cols = {"timestamp", *CHANNELS}
        if name == "test":
            required_cols.add("label")
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            errors.append(f"{name}.csv missing required columns: {', '.join(sorted(missing_cols))}")
        
        # FR-02C: Duplicate timestamps
        if "timestamp" in df.columns:
            dup_count = df["timestamp"].duplicated().sum()
            if dup_count > 0:
                errors.append(f"{name}.csv contains {dup_count} duplicate timestamp entries")
            
            # FR-02D: Chronological ordering
            if not df["timestamp"].is_monotonic_increasing:
                errors.append(f"{name}.csv timestamps are not in strictly chronological order")
                
        # FR-02E: Numeric float values
        for ch in CHANNELS:
            if ch in df.columns:
                if not pd.api.types.is_numeric_dtype(df[ch]):
                    errors.append(f"{name}.csv column '{ch}' contains non-numeric values")
                    
        # FR-02F: Missing values
        nan_counts = df[[c for c in CHANNELS if c in df.columns]].isna().sum().to_dict()
        total_nans = sum(nan_counts.values())
        if total_nans > 0:
            errors.append(f"{name}.csv contains missing values across channels: {nan_counts}")
            
    # Record counts verification
    if len(frames["train"]) != 4000:
        warnings.append(f"train.csv expected 4,000 records, found {len(frames['train'])}")
    if len(frames["test"]) != 3000:
        warnings.append(f"test.csv expected 3,000 records, found {len(frames['test'])}")
    if len(frames["telemetry"]) != 7000:
        warnings.append(f"telemetry.csv expected 7,000 records, found {len(frames['telemetry'])}")

    # FR-02G: Validate label values are binary (0 or 1)
    if "label" in frames["test"].columns:
        test_labels = set(frames["test"]["label"].unique())
        if not test_labels.issubset({0, 1}):
            errors.append(f"test.csv labels must be binary (0 or 1), found: {test_labels}")
            
    if "label" in frames["labels"].columns:
        labels_vals = set(frames["labels"]["label"].unique())
        if not labels_vals.issubset({0, 1}):
            errors.append(f"labels.csv labels must be binary (0 or 1), found: {labels_vals}")

    # FR-02H: Validate anomaly_type values restricted to {point, collective, level_shift, trend}
    for source in ("labels", "anomaly_types"):
        if "anomaly_type" in frames[source].columns:
            types_found = set(frames[source]["anomaly_type"].dropna().unique())
            invalid_types = types_found - ALLOWED_ANOMALY_TYPES
            if invalid_types:
                errors.append(f"{source}.csv contains unknown anomaly types: {invalid_types}")
            summary[f"{source}_types"] = list(types_found)

    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "summary": summary
    }


def validate_and_report(frames: Dict[str, pd.DataFrame]) -> bool:
    """Convenience function to validate and print report to stdout."""
    res = validate_dataset(frames)
    if res["is_valid"]:
        print("[OK] Data validation PASSED: All schema, timestamp, channel, and label checks succeeded.")
        if res["warnings"]:
            for w in res["warnings"]:
                print(f"[WARN] Warning: {w}")
    else:
        print("[ERROR] Data validation FAILED with errors:")
        for err in res["errors"]:
            print(f"   - {err}")
    return res["is_valid"]
