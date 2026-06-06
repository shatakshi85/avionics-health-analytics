# Avionics Health Analytics

## Overview

Avionics Health Analytics is an AI-based health monitoring system designed to assess the condition of avionics subsystems using flight telemetry data. The project combines statistical monitoring, unsupervised anomaly detection, and temporal degradation analysis to generate a unified health score that supports avionics reuse assessment.

The system is inspired by challenges in reusable launch vehicles (RLVs), where avionics components must be evaluated after each mission to ensure safe and efficient reuse.

---

## Problem Statement

Traditional avionics requalification methods rely on manual inspection and fixed threshold checks. These approaches are effective for detecting major faults but may fail to identify subtle anomalies and gradual performance degradation.

This project proposes a layered analytics framework that analyzes multivariate telemetry data and provides interpretable health scores for reuse decision support.

---

## Dataset

The system uses multivariate avionics telemetry data stored in NumPy (`.npy`) format.

### Dataset Characteristics

* 164 telemetry log files
* High-dimensional multivariate time-series data
* 25-sensor configuration (~65.9%)
* 55-sensor configuration (~34.1%)
* 50% Training Data
* 50% Testing Data

Each file contains:

* Rows → Time steps
* Columns → Sensor parameters

Telemetry captures thermal, electrical, and stability-related indicators.

---

## Methodology

### Layer 1 – Statistical Boundary Monitoring

Applies the Three-Sigma Rule to detect sensor readings that deviate significantly from normal operating conditions.

### Layer 2 – Multivariate Anomaly Detection

Uses Isolation Forest to identify contextual anomalies within high-dimensional telemetry space.

### Layer 3 – Temporal Degradation Analysis

Uses Rolling Variance analysis to detect gradual degradation and instability patterns over time.

### Health Score Aggregation

Outputs from all layers are combined into a normalized health score ranging from 0 to 100.

---

## Key Metrics

* Anomaly Rate
* Health Score (0–100)
* Three-Sigma Violations
* Isolation Forest Anomaly Score
* Rolling Variance
* Fleet Stability
* Dimensional Scalability
* Hyperparameter Robustness

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

## Results

* Stable anomaly detection performance (0.5%–2%)
* Robust behavior across 25-sensor and 55-sensor configurations
* Health scores primarily within the 95–100 range
* No fleet-wide instability observed

---



