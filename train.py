"""End-to-end training and evaluation script for Satellite Health Monitoring System.

Usage:
    python train.py
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

from src.data_loader import load_dataset, CHANNELS
from src.validate import validate_and_report
from src.preprocess import preprocess_data
from src.models import train_all_models, save_trained_models
from src.detect import compute_anomaly_scores, calibrate_threshold, predict_anomalies
from src.evaluate import compute_metrics, evaluate_per_anomaly_type


def run_training_pipeline():
    print("=" * 70)
    print("SATELLITE HEALTH MONITORING SYSTEM - TRAINING & EVALUATION PIPELINE")
    print("Dataset: ESA-ADB Prototype Mission 1 (Synthetic)")
    print("=" * 70)
    
    # 1. Ingestion
    print("\n[Step 1/5] Loading raw datasets...")
    frames = load_dataset()
    print(f"Loaded {len(frames['train']):,} train rows, {len(frames['test']):,} test rows, {len(frames['telemetry']):,} full telemetry rows.")
    
    # 2. Validation
    print("\n[Step 2/5] Running dataset validation...")
    if not validate_and_report(frames):
        raise RuntimeError("Validation failed. Aborting training.")
        
    # 3. Preprocessing
    print("\n[Step 3/5] Preprocessing and feature scaling...")
    prep = preprocess_data(frames, save_processed=True)
    X_train = prep["X_train"]
    X_test = prep["X_test"]
    y_test = prep["y_test"]
    scaler = prep["scaler"]
    
    # 4. Training
    print("\n[Step 4/5] Training machine learning models...")
    models = train_all_models(X_train, random_state=42)
    save_trained_models(models, scaler, output_dir="models")
    
    # 5. Threshold Calibration & Evaluation
    print("\n[Step 5/5] Scoring, threshold calibration & evaluation on test.csv (3,000 records)...")
    results = {}
    thresholds = {}
    
    for name in ["isolation_forest", "autoencoder"]:
        model = models[name]
        # Calibrate threshold on training data (97.5th percentile)
        thresh = calibrate_threshold(model, name, X_train, percentile=97.5)
        thresholds[name] = thresh
        
        # Test scores and predictions
        test_scores = compute_anomaly_scores(model, name, X_test)
        test_pred = predict_anomalies(test_scores, thresh)
        
        metrics = compute_metrics(y_true=y_test, y_pred=test_pred, y_scores=test_scores)
        per_type = evaluate_per_anomaly_type(frames["test"], frames["labels"], test_pred)
        
        results[name] = {
            "threshold": thresh,
            "metrics": metrics,
            "per_type_metrics": per_type
        }
        
        print(f"\n--- Model: {name.upper().replace('_', ' ')} ---")
        print(f"Decision Threshold: {thresh:.4f}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1-Score  : {metrics['f1_score']:.4f}")
        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"PR-AUC    : {metrics['pr_auc']:.4f}")
        print("Confusion Matrix:")
        cm = metrics["confusion_matrix"]
        print(f"   TN: {cm['true_negatives']} | FP: {cm['false_positives']}")
        print(f"   FN: {cm['false_negatives']} | TP: {cm['true_positives']}")
        
        print("\nPer-Anomaly-Type Breakdown:")
        for t in per_type:
            print(f"   - {t['type'].ljust(12)} (Count: {str(t['count']).rjust(3)}) -> Precision: {t['precision']:.3f} | Recall: {t['recall']:.3f} | F1: {t['f1_score']:.3f}")
            
    # Save benchmark metrics summary
    models_dir = Path("models")
    with open(models_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] All evaluation metrics saved to {models_dir / 'evaluation_results.json'}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_training_pipeline()
