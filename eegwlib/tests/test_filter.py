from typing import Optional

import pytest
import numpy as np

from ..filter import butter, filtfilt


@pytest.mark.parametrize('fmin,fmax,sos_expected', [
    (1.0, 40.0, np.array(
        [[0.014737, 0.0, -0.014737, 1.0, -1.97050281, 0.970526]]
    )),
    (40.0, 1.0, np.array(
        [[1.01518455, -2.03034521, 1.01518455, 1.0, -2.03034521, 1.0303691]]
    )),
    (1.0, None, np.array(
        [[0.99961665, -0.99961665, 0.0, 1.0, -0.9992333, 0.0]]
    )),
    (None, 40.0, np.array(
        [[0.01510922, 0.01510922, 0.0, 1.0, -0.96978156, 0.0]]
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
        [[3.6802871, 3.604227, 3.5445273, 3.4998417, 3.4480004, 3.398512,
          3.3629084, 3.3024743, 3.1935966, 3.06435, 2.926986, 2.7813084,
          2.6401021, 2.4995337, 2.3450487, 2.1873057]], dtype=np.float32)
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
