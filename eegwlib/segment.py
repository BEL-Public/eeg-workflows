import numpy as np


def slice_block(data_block: np.array, center: float,
                padl: float, padr: float, sr: float) -> np.array:
    """Return a slice of `data_block`

    A slice of data that is centered on `center` and extends `padl` to the left
    and `padr` to the right is extracted from `data_block`. `data_block` must
    be a 2D array and is sliced along axis 1.

    Parameters
    ----------
    data_block
        The 2D array to be sliced
    center
        The center of the slice in seconds relative
        to the beginning of `data_block`
    padl
        Time padding to the left of `center` in (sec)
    padr
        Time padding to the right of `center` (sec)
    sr
        Sampling rate of `data_block` (cycles/sec)

    Returns
    -------
    The slice of `data_block`

    Raises
    ------
    ValueError
        If `data_block` is not a 2-dimensional array

    Notes
    -----
    Because `center` is given in seconds, it is assumed this time point will
    fall between two consecutive samples in `data_block`. From this time point,
    the extracted slice is extended out in either direction by the number of
    samples that fit into `padl` and `padr`.
    """
    if len(data_block.shape) != 2:
        raise ValueError('Input array must be 2-dimensional. '
                         f'Got shape: {data_block.shape}')
    center_idx = int(center * sr)
    left_idx = center_idx - int(padl * sr)
    right_idx = center_idx + int(padr * sr)
    slice_indices = np.array(range(left_idx, right_idx))
    return data_block.take(slice_indices, axis=1)
