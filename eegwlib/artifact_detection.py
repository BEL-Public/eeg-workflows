from typing import List

import numpy as np


def detect_bad_channels(data: np.array, criteria: float) -> List[int]:
    """Return indices of channels that exceed peak-to-peak amplitude criteria

    Parameters
    ----------
    data
        2-dimensional array of shape (channels, samples)
    criteria
        Peak-to-peak amplitude criteria. The absolute value of `criteria` is
        used, so a negative value is accepted.

    Returns
    -------
    A list of the channel indices that exceed the peak-to-peak amplitude
    criteria

    Raises
    ------
    ValueError
        If `data` is not a 2-dimensional array
    """
    if len(data.shape) != 2:
        raise ValueError('Input array must be 2-dimensional. '
                         f'Got shape: {data.shape}')
    criteria = abs(criteria)
    bad_channels = []
    for i in range(len(data)):
        channel = data[i]
        if np.max(channel) - np.min(channel) > criteria:
            bad_channels.append(i)
    return bad_channels
