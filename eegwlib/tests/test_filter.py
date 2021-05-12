from typing import Optional

import pytest
import numpy as np

from ..filter import butter, filtfilt


@pytest.mark.parametrize('fmin,fmax,sos_expected', [
    (1.0, 40.0, np.array(
        [[0.58675824, 0.0, -0.58675824, 1.0, -0.76790625, -0.17351649]]
    )),
    (40.0, 1.0, np.array(
        [[-2.38157056, 4.42555211, -2.38157056, 1.0, 4.42555211, -5.76314113]]
    )),
    (1.0, None, np.array(
        [[0.97603957, -0.97603957, 0.0, 1.0, -0.95207915, 0.0]]
    )),
    (None, 40.0, np.array(
        [[0.59945618, 0.59945618, 0.0, 1.0, 0.19891237, 0.0]]
    )),
])
def test_butter(fmin: Optional[float], fmax: Optional[float],
                sos_expected: np.ndarray) -> None:
    """Test Butterworth filter design with the 4 filter varieties"""
    sos = butter(1, 128.0, fmin=fmin, fmax=fmax)
    assert sos == pytest.approx(sos_expected)


@pytest.mark.parametrize('fmin,fmax,message', [
    (1.0, 1.0, 'fmin and fmax cannot be equal.'),
    (None, None, 'Neither fmin nor fmax provided.'),
])
def test_butter_bad_input(fmin: Optional[float], fmax: Optional[float],
                          message: str) -> None:
    """Test Butterworth filter design with bad critical frequencies"""
    with pytest.raises(ValueError) as exc_info:
        butter(4, 128.0, fmin, fmax)
    assert str(exc_info.value) == message


def test_filtfilt() -> None:
    """Test filtering of sample array"""
    input_signals = np.array([[
        4.3, 2.5, 1.0, 5.9, 3.0, 1.3, 4.5, 9.4,
        2.1, 4.3, 3.6, 2.3, 1.8, 4.5, 2.7, 0.3
    ]], dtype=np.float32)
    expected_signals = np.array(
        [[3.6762197, 3.2882233, 3.236965, 3.486441, 3.5170538, 3.6501458,
          4.3144593, 4.690515, 4.2830677, 3.7850027, 3.3860874, 2.956997,
          2.7436712, 2.5981774, 2.0859182, 1.3545489]], dtype=np.float32
    )
    filtered_signals = filtfilt(input_signals, 10.0, 1, fmax=1.0)
    assert filtered_signals == pytest.approx(expected_signals)
    # Test no critical frequency values returns unfiltered array
    unfiltered_signals = filtfilt(input_signals, 10.0, 1)
    assert unfiltered_signals == pytest.approx(input_signals)


def test_filtfilt_bad_shape() -> None:
    """Test filtering of array with wrong shape"""
    array_1d = np.array([
        4.3, 2.5, 1.0, 5.9, 3.0, 1.3, 4.5, 9.4,
        2.1, 4.3, 3.6, 2.3, 1.8, 4.5, 2.7, 0.3
    ], dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        filtfilt(array_1d, 10.0, 1, fmax=1.0)
    message = f'Input array must be 2-dimensional. Got shape: {array_1d.shape}'
    assert str(exc_info.value) == message
