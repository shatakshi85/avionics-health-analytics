import numpy as np
import sys
import os

# Add src to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from health_score import compute_health_score
from rule_based import rule_based_analysis
from degradation import temporal_degradation_score

def test_health_score_bounds():
    # Arrange
    rule_violations = np.array([0, 10, 20])
    anomaly_labels = np.array([1, -1, 1])
    degradation = np.array([0, 100, 0])
    
    # Act
    scores = compute_health_score(rule_violations, anomaly_labels, degradation)
    
    # Assert
    assert scores.shape == (3,)
    assert np.all(scores >= 0) and np.all(scores <= 100)
    assert scores[0] > scores[1]  # Health should drop when violations/anomalies exist

def test_rule_based_static():
    # Arrange
    # 5 timesteps, 2 channels
    # Create 100 steady points
    X = np.ones((100, 2)) * 10
    # Add one extreme outlier
    X[-1, 0] = 500  
    
    # Act
    violations = rule_based_analysis(X)
    
    # Assert
    assert violations.shape == (100,)
    assert violations[-1] > 0, f"Expected violation at last index, but got {violations[-1]}"
    assert violations[0] == 0, f"Expected no violation at index 0, but got {violations[0]}"

def test_degradation_trend():
    # Arrange
    # Steady signal vs noisy signal
    steady = np.ones((200, 1)) * 10
    noisy = steady.copy()
    noisy[150:] = np.random.normal(10, 5, (50, 1))
    
    # Act
    score_steady = temporal_degradation_score(steady, window_size=50)
    score_noisy = temporal_degradation_score(noisy, window_size=50)
    
    # Assert
    assert score_steady[-1] < score_noisy[-1]

if __name__ == "__main__":
    print("Running tests...")
    try:
        test_health_score_bounds()
        print("✅ health_score tests passed")
        test_rule_based_static()
        print("✅ rule_based tests passed")
        test_degradation_trend()
        print("✅ degradation tests passed")
        print("\nAll tests passed successfully!")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
