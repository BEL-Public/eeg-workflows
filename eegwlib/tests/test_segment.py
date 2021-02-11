import pytest

import numpy as np

from ..segment import slice_block


@pytest.mark.parametrize('center,padl,padr', [
    (3.2, 1.6, 1.3),
    (3.0, 1.5, 1.0),
    (3.49, 1.99, 1.49),
])
def test_slice_block(center: float, padl: float, padr: float) -> None:
    """Test slicing of data block

    Three different sets of values for `center`, `padl`, and `padr` are tested.
    Because of the rounding that occurs when converting from seconds to
    samples, all three sets of values should yield the same result.
    """
    array = np.array([[3, 6, 7, 4, 2, 1, 7, 5, 3, 0],
                      [4, 1, 7, 3, 5, 4, 7, 8, 0, 8]])
    segment = slice_block(array, center=center, padl=padl, padr=padr, sr=2.0)
    expected_segment = np.array([[4, 2, 1, 7, 5],
                                 [3, 5, 4, 7, 8]])
    assert segment == pytest.approx(expected_segment)


def test_slice_block_bad_shape() -> None:
    """Test slicing data block of wrong shape"""
    array = np.array([0, 4, 2])
    with pytest.raises(ValueError) as exc_info:
        slice_block(array, 2.0, 1.0, 1.0, 1.0)
    message = 'Input array must be 2-dimensional. Got shape: (3,)'
    assert str(exc_info.value) == message
