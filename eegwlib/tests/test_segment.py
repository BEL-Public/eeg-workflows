import pytest
from os.path import dirname, join
from typing import Optional

import numpy as np

from ..segment import extract_segment_from_array, OutOfRangeError, \
    seconds_to_samples, Segmenter


def test_seconds_to_samples() -> None:
    """Test converting seconds to samples"""
    seconds = 3.49
    sr = 250.0
    expected_samples = 872
    assert seconds_to_samples(seconds, sr) == expected_samples


@pytest.mark.parametrize('center,padl,padr', [
    (3.2, 1.2, 1.1),
    (3.0, 1.0, 1.0),
    (2.9, 0.8, 0.9),
])
def test_extract_segment_from_array(center: float, padl: float,
                                    padr: float) -> None:
    """Test extracting segment from array of signal data

    Three different sets of values for `center`, `padl`, and `padr` are tested.
    Because of the rounding that occurs when converting from seconds to
    samples, all three sets of values should yield the same result.
    """
    array = np.array([[3, 6, 7, 4, 2, 1, 7, 5, 3, 0],
                      [4, 1, 7, 3, 5, 4, 7, 8, 0, 8]])
    sr = 2.0
    segment = extract_segment_from_array(array, center, padl, padr, sr)
    expected_segment = np.array([[2, 1, 7, 5],
                                 [5, 4, 7, 8]])
    assert segment == pytest.approx(expected_segment)


def test_extract_segment_bad_shape() -> None:
    """Test extracting segment from array of wrong shape"""
    array = np.array([0, 4, 2])
    with pytest.raises(ValueError) as exc_info:
        extract_segment_from_array(array, 2.0, 1.0, 1.0, 1.0)
    message = f'Input array must be 2-dimensional. Got shape: {array.shape}'
    assert str(exc_info.value) == message


@pytest.mark.parametrize('padl,padr', [
    (6.0, 0.0),
    (0.0, 6.0),
])
def test_extract_segment_out_of_range(padl: float, padr: float) -> None:
    """Test extracting an out of range segment throws"""
    array = np.random.randn(2, 10)
    center = 5.0
    sr = 1.0
    with pytest.raises(OutOfRangeError) as exc_info:
        extract_segment_from_array(array, center, padl, padr, sr)
    assert str(exc_info.value) == 'Requested segment extends beyond data block'


@pytest.fixture
def segmenter() -> Segmenter:
    """Return example `Segmenter` object"""
    examples_dir = join(dirname(__file__), '..', '..', 'examples')
    return Segmenter(join(examples_dir, 'example_raw.mff'))


def test_is_in_epoch(segmenter: Segmenter) -> None:
    """Test determining whether relative time in epoch"""
    relative_time = 0.3
    epoch = segmenter.epochs[0]
    assert segmenter.is_in_epoch(relative_time, epoch)


def test_data_cache(segmenter: Segmenter) -> None:
    """Test load data block and clear data cache"""
    # Load data block into data cache
    epoch = segmenter.epochs[0]
    order = 1
    fmin = 0.1
    fmax = 50.0
    segmenter.load_filtered_epoch(epoch, order, fmin, fmax)
    loaded_data = segmenter.get_loaded_data()
    assert loaded_data.shape == (257, 224)
    with pytest.raises(AssertionError) as exc_info:
        segmenter.load_filtered_epoch(epoch, order, fmin, fmax)
    assert str(exc_info.value) == 'A data block is already loaded. ' \
                                  'First, clear loaded data.'
    # Clear data cache
    segmenter.clear_loaded_data()
    with pytest.raises(AssertionError) as exc_info:
        segmenter.get_loaded_data()
    assert str(exc_info.value) == 'No data loaded'


@pytest.mark.parametrize('fmin,fmax,data_expected', [
    (None, None, np.array(
        [[-1841.2826, -1841.1616],
         [-1388.2782, -1389.1216],
         [-980.4295, -979.7764]], dtype=np.float32
    )),
    (1.0, 20.0, np.array(
        [[2.10289337e+02, 2.09430405e+02],
         [1.58008759e+02, 1.57409149e+02],
         [1.11948311e+02, 1.11571823e+02]], dtype=np.float32
    ))
])
def test_extract_filtered_segments(segmenter: Segmenter, fmin: Optional[float],
                                   fmax: Optional[float],
                                   data_expected: np.ndarray) -> None:
    """Test extracting filtered segments `Segmenter` object"""
    in_range_time = 0.4
    out_of_range_time = 0.895
    times = {'cat1': [in_range_time, out_of_range_time],
             'cat2': [in_range_time, out_of_range_time]}
    padl = 0.004
    padr = 0.004
    in_range_segs, out_of_range_segs = segmenter.extract_segments(
        times, padl, padr, fmin=fmin, fmax=fmax
    )
    for cat in times:
        assert len(in_range_segs[cat]) == 1
        assert in_range_segs[cat][0][:3] == pytest.approx(data_expected)
        assert len(out_of_range_segs[cat]) == 1
        assert out_of_range_segs[cat][0] == out_of_range_time


def test_extract_segments_negative_padding(segmenter: Segmenter) -> None:
    """Test negative padding throws `ValueError`"""
    negative_pad = -0.1
    with pytest.raises(ValueError) as exc_info:
        segmenter.extract_segments(dict(), padl=negative_pad, padr=0.1)
    message = f'Negative left padding: {negative_pad}'
    assert str(exc_info.value) == message
    with pytest.raises(ValueError) as exc_info:
        segmenter.extract_segments(dict(), padl=0.1, padr=negative_pad)
    message = f'Negative right padding: {negative_pad}'
    assert str(exc_info.value) == message
