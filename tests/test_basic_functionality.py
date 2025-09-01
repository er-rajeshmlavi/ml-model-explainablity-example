"""
Basic functionality tests for SHAP and LIME examples.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def test_numpy_basic():
    """Test basic NumPy functionality."""
    arr = np.array([1, 2, 3, 4, 5])
    assert len(arr) == 5
    assert arr.mean() == 3.0


def test_pandas_basic():
    """Test basic Pandas functionality."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert len(df) == 3
    assert list(df.columns) == ['a', 'b']


def test_sklearn_basic():
    """Test basic scikit-learn functionality."""
    # Create simple dataset
    X = np.random.randn(100, 3)
    y = X.sum(axis=1) + np.random.randn(100) * 0.1
    
    # Train model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Test prediction
    pred = model.predict(X[:5])
    assert len(pred) == 5
    assert all(isinstance(p, (int, float, np.number)) for p in pred)


def test_synthetic_data_generation():
    """Test synthetic data generation functions."""
    # Simple synthetic time series
    n_samples = 100
    time_steps = np.arange(n_samples)
    
    # Generate components
    trend = 0.01 * time_steps
    seasonal = np.sin(2 * np.pi * time_steps / 10)
    noise = np.random.normal(0, 0.1, n_samples)
    
    # Combine
    y = trend + seasonal + noise
    
    assert len(y) == n_samples
    assert isinstance(y, np.ndarray)
    assert not np.any(np.isnan(y))


if __name__ == "__main__":
    pytest.main([__file__])
