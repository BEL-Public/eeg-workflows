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
    sr = 2.0
    segment = slice_block(array, center=center, padl=padl, padr=padr, sr=sr)
    expected_segment = np.array([[2, 1, 7, 5, 3],
                                 [5, 4, 7, 8, 0]])
    assert segment == pytest.approx(expected_segment)


@pytest.mark.parametrize('padl,padr', [
    (1.0, 1.0),
    (0.9, 1.1),
    (1.8, 1.4)
])
def test_slicing_returns_same_shape(padl: float, padr: float) -> None:
    """Test slicing at different centers with same
    padding returns segments of same shape"""
    array = np.array([[3, 6, 7, 4, 2, 1, 7, 5, 3, 0],
                      [4, 1, 7, 3, 5, 4, 7, 8, 0, 8]])
    sr = 2.0
    sample_seg = slice_block(array, center=2.9, padl=padl, padr=padr, sr=sr)
    if sample_seg is not None:
        sample_shape = sample_seg.shape
    else:
        raise ValueError('NoneType result not expected')
    centers = [2.5, 2.6, 3.0, 3.1, 3.4]
    for center in centers:
        segment = slice_block(array, center=center,
                              padl=padl, padr=padr, sr=sr)
        if segment is not None:
            assert segment.shape == sample_shape
        else:
            raise ValueError('NoneType result not expected')


def test_slice_block_bad_shape() -> None:
    """Test slicing data block of wrong shape"""
    array = np.array([0, 4, 2])
    with pytest.raises(ValueError) as exc_info:
        slice_block(array, 2.0, 1.0, 1.0, 1.0)
    message = f'Input array must be 2-dimensional. Got shape: {array.shape}'
    assert str(exc_info.value) == message


@pytest.mark.parametrize('padl,padr', [
    (7.0, 0.0),
    (0.0, 5.0),
    (7.0, 5.0)
])
def test_out_of_range_slice_returns_none(padl: float, padr: float) -> None:
    """Test requesting an out of range slice returns None"""
    array = np.random.randn(2, 10)
    no_slice = slice_block(array, center=5.0, padl=padl, padr=padr, sr=1.0)
    assert no_slice is None
