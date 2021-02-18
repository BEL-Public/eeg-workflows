import pytest

import numpy as np

from ..average import Average


def test_average_class() -> None:
    """Test creation of `Average` object"""
    category = 'catx'
    num_samples = 10
    num_segments = 5
    segments = [
        np.random.randn(32, num_samples).astype(np.float32)
    ] * num_segments
    center = 5
    sr = 5.0
    bads = [4, 2, 17]
    average = Average(category, segments, center=center, sr=sr, bads=bads)
    assert average.category == category
    assert average.segments == segments
    assert average.center == center
    assert average.sampling_rate == sr
    assert average.bads == bads
    assert average.num_samples() == num_samples
    assert average.num_segments() == num_segments
    expected_content = {
        'status': 'unedited',
        'name': 'Average',
        'beginTime': 2_000_000,
        'endTime': 4_000_000,
        'evtBegin': 3_000_000,
        'evtEnd': 3_000_000,
        'channelStatus': [
            {
                'signalBin': 1,
                'exclusion': 'badChannels',
                'channels': bads
            }
        ],
        'keys': {
            '#seg': {
                'type': 'long',
                'data': num_segments
            }
        }
    }
    content = average.build_category_content(begin_time=2_000_000)
    assert content == expected_content


def test_segments_of_different_shape_throws() -> None:
    shape_1 = (32, 10)
    shape_2 = (64, 12)
    segments = [
        np.random.randn(shape_1[0], shape_1[1]).astype(np.float32),
        np.random.randn(shape_2[0], shape_2[1]).astype(np.float32)
    ]
    with pytest.raises(ValueError) as exc_info:
        Average('catx', segments, center=5, sr=1.0)
    message = f'Segments have different shapes: {shape_2} != {shape_1}'
    assert str(exc_info.value) == message


def test_average_center_out_of_range() -> None:
    """Test creation of `Average` fails when center out of range"""
    num_samples = 10
    center = 11
    segments = [np.random.randn(32, num_samples).astype(np.float32)] * 2
    with pytest.raises(ValueError) as exc_info:
        Average('catx', segments, center=center, sr=5.0)
    message = f'Center ({center}) cannot be larger than ' \
              f'length of data block ({num_samples})'
    assert str(exc_info.value) == message
