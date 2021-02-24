from typing import List

import numpy as np


def detect_bad_channels(data: np.ndarray, criterion: float) -> List[int]:
    """Return indices of channels that exceed peak-to-peak amplitude criterion

    Parameters
    ----------
    data
        2-dimensional array of shape (channels, samples)
    criterion
        Peak-to-peak amplitude criterion. The absolute value of `criterion` is
        used, so a negative value is accepted.

    Returns
    -------
    A list of the channel indices that exceed the peak-to-peak amplitude
    criterion

    Raises
    ------
    ValueError
        If `data` is not a 2-dimensional array
    """
    if len(data.shape) != 2:
        raise ValueError('Input array must be 2-dimensional. '
                         f'Got shape: {data.shape}')
    criterion = abs(criterion)
    xmin = np.min(data, axis=1)
    xmax = np.max(data, axis=1)
    bads = xmax - xmin > criterion
    return list(np.arange(data.shape[0])[bads])
