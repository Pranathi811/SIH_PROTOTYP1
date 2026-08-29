ESA-ADB Prototype Synthetic Mission 1 Dataset
================================================

This is SYNTHETIC data created only for prototyping and debugging an
anomaly-detection pipeline. It is NOT ESA telemetry and must not be used
as ESA-ADB benchmark results.

Files
-----
telemetry.csv       Full synthetic telemetry with train/test split and labels
train.csv           Training telemetry
test.csv            Test telemetry with labels
labels.csv          Timestamp-level ground truth
anomaly_types.csv   Synthetic anomaly intervals/types

Channels
--------
channel_1 ... channel_5

Synthetic anomaly types
-----------------------
point
collective
level_shift
trend

Recommended prototype workflow
------------------------------
1. Train your detector on train.csv.
2. Generate one anomaly score per timestamp in test.csv.
3. Compare scores against test.csv labels.
4. Once the pipeline works, replace this synthetic dataset with the
   real ESA-AD/ESA-ADB preprocessed data.

Note: This directory is intentionally a simple, model-development format.
The exact TimeEval dataset format should be matched to the ESA-ADB fork
before running the official benchmark.
