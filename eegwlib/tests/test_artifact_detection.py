"""
Copyright 2021 Brain Electrophysiology Laboratory Company LLC

Licensed under the ApacheLicense, Version 2.0(the "License");
you may not use this module except in compliance with the License.
You may obtain a copy of the License at:

http: // www.apache.org / licenses / LICENSE - 2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied.
"""
import pytest
from typing import List

import numpy as np

from ..artifact_detection import detect_bad_channels


@pytest.mark.parametrize('criterion,bad_channels', [
    (15.0, [2]),
    (8.0, [0, 2]),
    (7.9, [0, 1, 2]),
    (20.0, []),
    (-1.0, [0, 1, 2]),
])
def test_detect_bad_channels(criterion: float,
                             bad_channels: List[int]) -> None:
    """Test bad channels are returned"""
    data = np.array([[-2, 6, -1, -4, 0],
                     [9, 3, 5, 2, 1],
                     [0, -4, -8, -3, 8]])
    assert detect_bad_channels(data, criterion) == bad_channels


def test_detect_bad_channels_wrong_shape() -> None:
    """Test error is thrown for non-2D array"""
    shape = (3, 3, 3)
    data = np.random.standard_normal(shape)
    with pytest.raises(ValueError) as exc_info:
        detect_bad_channels(data, 10.0)
    message = f'Input array must be 2-dimensional. Got shape: {shape}'
    assert str(exc_info.value) == message
