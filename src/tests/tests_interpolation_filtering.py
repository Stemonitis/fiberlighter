"""Tests for fiberlight.preprocessing module."""

import numpy as np
from fiberlight.interpolation_filtering import interpolate


def test_interpolate_basic():
    """Test that interpolate returns correct length."""
    time_405 = np.array([0, 1, 2])
    signal_405 = np.array([10, 12, 14])
    time_target = np.array([0, 0.5, 1, 1.5, 2])
    
    result = interpolate(signal_405, time_405, time_target)
    
    assert len(result) == len(time_target)


def test_interpolate_endpoints():
    """Test that interpolation respects original values at endpoints."""
    time_405 = np.array([0, 1])
    signal_405 = np.array([10, 20])
    time_target = np.array([0, 0.5, 1])
    
    result = interpolate(signal_405, time_405, time_target)
    
    assert np.isclose(result[0], 10)
    assert np.isclose(result[-1], 20)