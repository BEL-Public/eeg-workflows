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
    expected_content = [
        {
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
    ]
    assert average.build_category_content(idx=1) == expected_content


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
