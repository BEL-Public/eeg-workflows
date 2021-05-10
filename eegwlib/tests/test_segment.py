import pytest
from os.path import dirname, join

import numpy as np

from ..segment import extract_segment_from_array, seconds_to_samples, Segmenter


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
    with pytest.raises(IndexError) as exc_info:
        extract_segment_from_array(array, center, padl, padr, sr)
    assert str(exc_info.value) == 'Requested segment extends beyond data block'


@pytest.fixture
def example_raw() -> str:
    """Return path to example raw MFF"""
    examples_dir = join(dirname(__file__), '..', '..', 'examples')
    return join(examples_dir, 'example_raw.mff')


@pytest.fixture
def segmenter(example_raw: str) -> Segmenter:
    """Return example `Segmenter` object"""
    return Segmenter(example_raw, 0.004, 0.004, order=4, fmin=1.0, fmax=20.0)


def test_init_segmenter_negative_padding(example_raw: str) -> None:
    """Test negative padding throws `ValueError`"""
    negative_pad = -0.1
    with pytest.raises(ValueError) as exc_info:
        Segmenter(example_raw, padl=negative_pad, padr=0.1)
    message = f'Negative left padding: {negative_pad}'
    assert str(exc_info.value) == message
    with pytest.raises(ValueError) as exc_info:
        Segmenter(example_raw, padl=0.1, padr=negative_pad)
    message = f'Negative right padding: {negative_pad}'
    assert str(exc_info.value) == message


def test_is_in_epoch(segmenter: Segmenter) -> None:
    """Test determining whether relative time in epoch"""
    relative_time = 0.3
    epoch = segmenter.epochs[0]
    assert segmenter.is_in_epoch(relative_time, epoch)


def test_data_cache(segmenter: Segmenter) -> None:
    """Test load data block and clear data cache"""
    # Load data block into data cache
    epoch = segmenter.epochs[0]
    segmenter.load_filtered_epoch(epoch)
    loaded_data = segmenter.get_loaded_data()
    assert loaded_data.shape == (257, 224)
    with pytest.raises(AssertionError) as exc_info:
        segmenter.load_filtered_epoch(epoch)
    assert str(exc_info.value) == 'A data block is already loaded. ' \
                                  'First, clear loaded data.'
    # Clear data cache
    segmenter.clear_loaded_data()
    with pytest.raises(AssertionError) as exc_info:
        segmenter.get_loaded_data()
    assert str(exc_info.value) == 'No data loaded'


def test_sort_category_times_by_epoch(segmenter: Segmenter) -> None:
    """Test sorting times by category into times by epoch"""
    times = {'cat1': [0.0, 0.3, 0.1],
             'cat2': [0.7, 0.2, 0.5]}
    times_by_epoch_expected = {
        0: [('cat1', 0.0), ('cat1', 0.1), ('cat1', 0.3),
            ('cat2', 0.2), ('cat2', 0.5), ('cat2', 0.7)]
    }
    times_by_epoch = segmenter._sort_category_times_by_epoch(times)
    assert times_by_epoch == times_by_epoch_expected


def test_extract_segments(segmenter: Segmenter) -> None:
    """Test extracting segments from `Segmenter` object"""
    in_range_time = 0.4
    out_of_range_time = 0.895
    times = {'cat1': [in_range_time, out_of_range_time],
             'cat2': [in_range_time, out_of_range_time]}
    in_range_segs, out_of_range_segs = segmenter.extract_segments(times)
    data_expected = np.array(
        [[2.10289337e+02, 2.09430405e+02],
         [1.58008759e+02, 1.57409149e+02],
         [1.11948311e+02, 1.11571823e+02]], dtype=np.float32
    )
    for cat in times:
        assert len(in_range_segs[cat]) == 1
        assert in_range_segs[cat][0][:3] == pytest.approx(data_expected)
        assert len(out_of_range_segs[cat]) == 1
        assert out_of_range_segs[cat][0] == out_of_range_time
