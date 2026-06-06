import pandas as pd
import numpy as np
import config

def temporal_degradation_score(X, window_size=config.DEGRADATION_WINDOW):
    """
    Compute degradation trend using rolling variance.

    Parameters:
    X : ndarray (timesteps × channels)
    window_size : rolling window length

    Returns:
    degradation_score : 1D array representing instability trend
    """

    # Convert to DataFrame for rolling operations
    df = pd.DataFrame(X)

    # Rolling variance across time
    rolling_var = df.rolling(window=window_size).var()

    # Aggregate across channels
    degradation_score = rolling_var.mean(axis=1)

    # Handle NaNs at start
    degradation_score = degradation_score.fillna(0)

    return degradation_score.values
