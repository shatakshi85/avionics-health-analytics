import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Project imports ----
import config
from utils import load_all_npy, setup_logging, ensure_dirs
from preprocess import preprocess_train, preprocess_test
from anomaly_model import train_isolation_forest, predict_anomalies
from rule_based import rule_based_analysis
from degradation import temporal_degradation_score
from health_score import compute_health_score

# Setup Logging and Directories
logger = setup_logging(config.LOGS_DIR)
ensure_dirs([config.PLOTS_DIR, config.METRICS_DIR])

# TRAINING PHASE
logger.info("================ TRAINING PHASE ================")

# Load all training telemetry for supported feature counts
train_df = load_all_npy(
    folder_path=config.TRAIN_DIR,
    expected_features=config.SUPPORTED_FEATURES
)

if train_df.empty:
    logger.error("No training data loaded. Exiting.")
    exit(1)

# Dictonaries to store models and scalers per feature count
models = {}
scalers = {}

for f_count in config.SUPPORTED_FEATURES:
    subset = train_df[train_df["feature_count"] == f_count]
    if subset.empty:
        logger.warning(f"No training data found for {f_count} features. Skipping model training.")
        continue
    
    logger.info(f"Training model for {f_count} features (samples={len(subset)})")
    
    # Remove metadata columns and any phantom columns from other configurations
    X_train_raw = subset.drop(columns=["source_file", "feature_count"]).dropna(axis=1, how='all')
    
    logger.info(f"Refined training feature count for {f_count}: {X_train_raw.shape[1]}")
    
    # Preprocess
    X_train, scaler = preprocess_train(X_train_raw)
    
    # Train
    iso_model = train_isolation_forest(X_train)
    
    # Store
    models[f_count] = iso_model
    scalers[f_count] = scaler
    logger.info(f"Isolation Forest trained for {f_count} features.")

# TESTING & EVALUATION PHASE
logger.info("================ TESTING PHASE ================")

all_metrics = []
all_health_scores = []
all_anomaly_rates_list = []

for file in os.listdir(config.TEST_DIR):
    if not file.endswith(".npy"):
        continue

    file_path = os.path.join(config.TEST_DIR, file)
    data = np.load(file_path)
    f_count = data.shape[1]

    # Check if we have a model for this feature count
    if f_count not in models:
        logger.warning(f"Skipping {file}: No model trained for {f_count} features.")
        continue

    logger.info(f"Processing test file: {file} (features={f_count})")

    # Get the specific model and scaler
    iso_model = models[f_count]
    scaler = scalers[f_count]

    test_df = pd.DataFrame(data)

    # ---------- Preprocessing ----------
    X_test = preprocess_test(test_df, scaler)

    # ---------- Rule-based analytics ----------
    # Rule based uses standard k which is 3 by default
    rule_violations = rule_based_analysis(X_test)

    # ---------- Anomaly detection ----------
    anomaly_scores, anomaly_labels = predict_anomalies(iso_model, X_test)

    # ---------- Degradation analysis ----------
    degradation = temporal_degradation_score(X_test)

    # ---------- Health score ----------
    health_score = compute_health_score(rule_violations, anomaly_labels, degradation)
    
    all_health_scores.extend(health_score.tolist())
    current_anomaly_rate = (anomaly_labels == -1).mean()
    
    all_metrics.append({
        "file": file,
        "feature_count": f_count,
        "anomaly_rate": current_anomaly_rate,
        "health_mean": health_score.mean(),
        "health_min": health_score.min()
    })
    
    all_anomaly_rates_list.append({
        "file": file,
        "anomaly_rate": current_anomaly_rate
    })

# Save results to CSV
metrics_df = pd.DataFrame(all_metrics)
csv_path = os.path.join(config.METRICS_DIR, "evaluation_metrics_v2.csv")
metrics_df.to_csv(csv_path, index=False)
logger.info(f"Final Mini Project 2 metrics saved to {csv_path}")

# PLOTS
logger.info("Generating professional charts for Mini Project 2...")

# 1. Global health distribution
if all_health_scores:
    plt.figure(figsize=(7,5))
    plt.hist(all_health_scores, bins=50, color="indigo", alpha=0.7)
    plt.xlabel("Health Score")
    plt.ylabel("Frequency")
    plt.title("Overall Health Score Distribution (All Configurations)")
    plt.savefig(os.path.join(config.PLOTS_DIR, "v2_health_distribution.png"), dpi=300)
    plt.close()

# 2. Aggregated Health Score Trend (Smoothed)
if len(all_health_scores) > 200:
    health_array = np.array(all_health_scores)
    window = 200
    smoothed = np.convolve(health_array, np.ones(window)/window, mode="valid")
    plt.figure(figsize=(10,4))
    plt.plot(smoothed, color="darkgreen", linewidth=1.5)
    plt.xlabel("Aggregated Time (All Files)")
    plt.ylabel("Health Score")
    plt.title("Global Health Trend (Smoothed Across Fleet)")
    plt.savefig(os.path.join(config.PLOTS_DIR, "v2_health_trend.png"), dpi=300)
    plt.close()

# 3. Comparative Anomaly Rate
if all_anomaly_rates_list:
    anomaly_df = pd.DataFrame(all_anomaly_rates_list)
    plt.figure(figsize=(15, 6))
    plt.bar(anomaly_df["file"], anomaly_df["anomaly_rate"], color="teal")
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel("Test File")
    plt.ylabel("Anomaly Rate")
    plt.title("Anomaly Rate Across All Fleet (25 & 55 Channels)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "v2_anomaly_comparison.png"), dpi=300)
    plt.close()

# 4. Feature Count Distribution
if all_metrics:
    f_counts = pd.DataFrame(all_metrics)["feature_count"].value_counts()
    plt.figure(figsize=(6,6))
    plt.pie(f_counts, labels=[f"{c} Sensors" for c in f_counts.index], autopct='%1.1f%%', colors=["skyblue", "salmon"])
    plt.title("Dataset Composition (Sensor Configurations)")
    plt.savefig(os.path.join(config.PLOTS_DIR, "v2_feature_distribution.png"), dpi=300)
    plt.close()
    
    # ================= ADDITIONAL UNSUPERVISED EVALUATION =================

import seaborn as sns
from sklearn.ensemble import IsolationForest

logger.info("Running Advanced Unsupervised Evaluation...")

# ---------------- 1️⃣ Layer Contribution Analysis ----------------
layer_stats = []

for file_metric in all_metrics:
    file_name = file_metric["file"]
    f_count = file_metric["feature_count"]

    # Load file again
    file_path = os.path.join(config.TEST_DIR, file_name)
    data = np.load(file_path)
    test_df = pd.DataFrame(data)

    iso_model = models[f_count]
    scaler = scalers[f_count]

    X_test = preprocess_test(test_df, scaler)

    rule_violations = rule_based_analysis(X_test)
    anomaly_scores, anomaly_labels = predict_anomalies(iso_model, X_test)
    degradation = temporal_degradation_score(X_test)

    layer_stats.append({
        "file": file_name,
        "feature_count": f_count,
        "rules_risk_mean": np.mean(rule_violations),
        "iforest_anomaly_rate": (anomaly_labels == -1).mean(),
        "degradation_mean": np.mean(degradation)
    })

layer_df = pd.DataFrame(layer_stats)
layer_df.to_csv(os.path.join(config.METRICS_DIR, "layer_contributions.csv"), index=False)

# Layer Contribution Plot
plt.figure(figsize=(10,5))
sns.barplot(data=layer_df, x="file", y="iforest_anomaly_rate")
plt.xticks(rotation=90, fontsize=8)
plt.title("Isolation Forest Contribution by File")
plt.tight_layout()
plt.savefig(os.path.join(config.PLOTS_DIR, "v2_layer_contribution.png"), dpi=300)
plt.close()

# ---------------- 2️⃣ Contamination Sensitivity Analysis ----------------
contamination_values = [0.005, 0.01, 0.02]
sensitivity_results = []

for f_count in models:
    subset = train_df[train_df["feature_count"] == f_count]
    X_subset_raw = subset.drop(columns=["source_file", "feature_count"]).dropna(axis=1, how='all')
    X_subset, _ = preprocess_train(X_subset_raw)

    for c in contamination_values:
        model = IsolationForest(contamination=c, random_state=42)
        model.fit(X_subset)
        labels = model.predict(X_subset)
        anomaly_rate = (labels == -1).mean()

        sensitivity_results.append({
            "feature_count": f_count,
            "contamination": c,
            "anomaly_rate": anomaly_rate
        })

sensitivity_df = pd.DataFrame(sensitivity_results)
sensitivity_df.to_csv(os.path.join(config.METRICS_DIR, "contamination_sensitivity.csv"), index=False)

# Sensitivity Plot
plt.figure(figsize=(6,4))
sns.lineplot(data=sensitivity_df, x="contamination", y="anomaly_rate", hue="feature_count", marker="o")
plt.title("Contamination Sensitivity Analysis")
plt.savefig(os.path.join(config.PLOTS_DIR, "v2_contamination_sensitivity.png"), dpi=300)
plt.close()

# ---------------- 3️⃣ Global Health Stability Statistics ----------------
health_array = np.array(all_health_scores)

health_stats = {
    "mean": np.mean(health_array),
    "std_dev": np.std(health_array),
    "min": np.min(health_array),
    "max": np.max(health_array)
}

health_stats_df = pd.DataFrame([health_stats])
health_stats_df.to_csv(os.path.join(config.METRICS_DIR, "global_health_statistics.csv"), index=False)

logger.info(f"Health Mean: {health_stats['mean']}")
logger.info(f"Health Std Dev: {health_stats['std_dev']}")
logger.info(f"Health Min: {health_stats['min']}")
logger.info(f"Health Max: {health_stats['max']}")

# ---------------- 4️⃣ Cross-Configuration Robustness ----------------
config_stats = metrics_df.groupby("feature_count").agg({
    "anomaly_rate": ["mean", "std"],
    "health_mean": ["mean", "std"]
})

config_stats.columns = ["_".join(col) for col in config_stats.columns]
config_stats.reset_index(inplace=True)

config_stats.to_csv(os.path.join(config.METRICS_DIR, "configuration_comparison.csv"), index=False)

# Configuration Comparison Plot
plt.figure(figsize=(6,4))
sns.barplot(data=config_stats, x="feature_count", y="anomaly_rate_mean")
plt.title("Average Anomaly Rate by Sensor Configuration")
plt.savefig(os.path.join(config.PLOTS_DIR, "v2_configuration_comparison.png"), dpi=300)
plt.close()


logger.info("Mini Project 2 Analysis complete.")
