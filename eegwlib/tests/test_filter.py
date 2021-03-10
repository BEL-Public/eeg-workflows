from typing import Optional, Union

import pytest
import numpy as np
from scipy import signal

from ..filter import butter, filtfilt


@pytest.mark.parametrize('fmin,fmax,band,btype', [
    (1.0, 40.0, np.array([0.015625, 0.625], dtype=np.float32), 'bandpass'),
    (40.0, 1.0, np.array([0.625, 0.015625], dtype=np.float32), 'bandstop'),
    (1.0, None, 0.015625, 'highpass'),
    (None, 40.0, 0.625, 'lowpass'),
])
def test_butter(fmin: Optional[float], fmax: Optional[float],
                band: Union[np.array, float], btype: str) -> None:
    """Test Butterworth filter design with the 4 filter varieties"""
    expected_coefs = signal.butter(4, band, btype=btype, output='sos')
    coefs = butter(4, 128.0, fmin, fmax)
    assert coefs == pytest.approx(expected_coefs)


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
    expected_signals = np.array([[
        3.4575894, 3.241897, 3.1787324, 3.299958, 3.580789, 3.942394,
        4.274485, 4.4716654, 4.469146, 4.2595544, 3.8825028, 3.396784,
        2.8549209, 2.2941477, 1.7434243, 1.234144
    ]], dtype=np.float32)
    filtered_signals = filtfilt(input_signals, 4, 10.0, fmax=1.0)
    assert filtered_signals == pytest.approx(expected_signals)
    # Test no critical frequency values returns unfiltered array
    unfiltered_signals = filtfilt(input_signals, 4, 10.0)
    assert unfiltered_signals == pytest.approx(input_signals)


def test_filtfilt_bad_shape() -> None:
    """Test filtering of array with wrong shape"""
    array_1d = np.array([
        4.3, 2.5, 1.0, 5.9, 3.0, 1.3, 4.5, 9.4,
        2.1, 4.3, 3.6, 2.3, 1.8, 4.5, 2.7, 0.3
    ], dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        filtfilt(array_1d, 4, 10.0, fmax=1.0)
    message = f'Input array must be 2-dimensional. Got shape: {array_1d.shape}'
    assert str(exc_info.value) == message
