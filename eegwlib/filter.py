from typing import Optional

import numpy as np
from scipy import signal
from functools import lru_cache


@lru_cache(maxsize=128)
def butter(order: int, sr: float, fmin: Optional[float] = None,
           fmax: Optional[float] = None) -> signal.butter:
    """Design a Butterworth filter and return the coefficients

    Parameters
    ----------
    order
        Filter order
    sr
        Sampling rate (samples/sec)
    fmin
        Lower critical frequency (Hz)
    fmax
        Upper critical frequency (Hz)

    Raises
    ------
    ValueError
        If `fmin` and `fmax` are equal
    ValueError
        If `fmin` and `fmax` are both None

    Returns
    -------
    The Butterworth filter coefficients

    Notes
    -----
    The variety of filter depends on the values of `fmin` and `fmax`
    fmin < fmax: Bandpass
    fmin > fmax: Bandstop
    fmax == None: Highpass
    fmin == None: Lowpass
    """
    # Convert critical frequencies to half-cycles/sample
    fmin = 2 * fmin / sr if fmin else None
    fmax = 2 * fmax / sr if fmax else None
    if fmin and fmax:
        if fmin < fmax:
            btype = 'bandpass'
        elif fmin > fmax:
            btype = 'bandstop'
        else:
            raise ValueError('fmin and fmax cannot be equal.')
        band = np.array([fmin, fmax], dtype=np.float32)
    elif fmin:
        btype = 'highpass'
        band = fmin
    elif fmax:
        btype = 'lowpass'
        band = fmax
    else:
        raise ValueError('Neither fmin nor fmax provided.')
    return signal.butter(order, band, btype=btype, output='sos')


def filtfilt(arr: np.array, order: int, sr: float,
             fmin: Optional[float] = None,
             fmax: Optional[float] = None) -> np.array:
    """Apply a forward-backward Butterworth filter to `arr`

    Parameters
    ----------
    arr
        The array of signals to be filtered
    order
        Filter order
    sr
        Sampling rate of the input signals (cycles/sec)
    fmin
        Lower critical frequency (Hz)
    fmax
        Upper critical frequency (Hz)

    Returns
    -------
    The filtered signals. If `fmin` and `fmax` are both None,
    `arr` is returned.

    Raises
    ------
    ValueError
        If `arr` is not a 2-dimensional array
    """
    if len(arr.shape) != 2:
        raise ValueError('Input array must be 2-dimensional. '
                         f'Got shape: {arr.shape}')
    if fmin is None and fmax is None:
        return arr
    else:
        coefs = butter(order, sr, fmin, fmax)
        return signal.sosfiltfilt(coefs, arr, axis=1,
                                  padtype='constant').astype(np.float32)
