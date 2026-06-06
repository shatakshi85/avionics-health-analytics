import os

# Get project root (go up from src folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Paths ---

DATA_DIR = os.path.join(BASE_DIR, "data", "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

# --- Model Parameters ---
SUPPORTED_FEATURES = [25, 55]
CONTAMINATION = 0.01
RANDOM_STATE = 42

# --- Analysis Parameters ---
RULE_K = 3
DEGRADATION_WINDOW = 100

# --- Health Score Weights ---
WEIGHT_RULES = 0.4
WEIGHT_ANOMALY = 0.4
WEIGHT_DEGRADATION = 0.2


