import numpy as np

def compute_health_score(rule_violations, anomaly_labels, degradation):
    """
    Compute avionics health score by fusing rule violations,
    anomaly detections, and degradation trends.
    """

    # Convert anomaly labels (-1 anomaly, 1 normal) to binary
    anomaly_count = (anomaly_labels == -1).astype(int)

    # Normalize components
    rule_norm = rule_violations / (rule_violations.max() + 1e-6)
    anomaly_norm = anomaly_count / (anomaly_count.max() + 1e-6)
    degr_norm = degradation / (degradation.max() + 1e-6)

    # Weighted fusion
    combined_risk = (
        0.4 * rule_norm +
        0.4 * anomaly_norm +
        0.2 * degr_norm
    )

    # Convert to health score
    health_score = 100 * (1 - combined_risk)

    # Ensure valid bounds
    health_score = np.clip(health_score, 0, 100)

    # ✅ THIS LINE IS CRITICAL
    return health_score
