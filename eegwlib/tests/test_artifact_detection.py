import pytest
from typing import List

import numpy as np

from ..artifact_detection import detect_bad_channels


@pytest.mark.parametrize('criteria,bad_channels', [
    (15.0, [2]),
    (8.0, [0, 2]),
    (7.9, [0, 1, 2]),
    (20.0, []),
    (-1.0, [0, 1, 2])
])
def test_detect_bad_channels(criteria: float, bad_channels: List[int]):
    """Test bad channels are returned"""
    data = np.array([[-2, 6, -1, -4, 0],
                     [9, 3, 5, 2, 1],
                     [0, -4, -8, -3, 8]])
    assert detect_bad_channels(data, criteria) == bad_channels


def test_detect_bad_channels_wrong_shape():
    """Test error is thrown for non-2D array"""
    shape = (3, 3, 3)
    data = np.random.standard_normal(shape)
    with pytest.raises(ValueError) as exc_info:
        detect_bad_channels(data, 10.0)
    message = f'Input array must be 2-dimensional. Got shape: {shape}'
    assert str(exc_info.value) == message
