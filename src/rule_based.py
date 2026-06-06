import numpy as np
import config

def rule_based_analysis(X, k=config.RULE_K):
    """
    Apply statistical threshold checks on telemetry data.

    Parameters:
    X : ndarray (timesteps × channels)
    k : number of standard deviations for bounds

    Returns:
    rule_violation_count : array of violations per timestep
    """

    # Compute mean and std for each channel
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Define upper and lower bounds
    upper_bound = mean + k * std
    lower_bound = mean - k * std

    # Check violations
    violations = (X > upper_bound) | (X < lower_bound)

    # Count number of violated channels at each timestep
    rule_violation_count = violations.sum(axis=1)

    return rule_violation_count
