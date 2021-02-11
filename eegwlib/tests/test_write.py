import pytest
from datetime import datetime, timezone
from os import rmdir, remove
from os.path import join

import numpy as np
from mffpy import Reader

from ..write import write_averaged
from ..average import Average


def test_write_averaged() -> None:
    """Test writing `Averaged` objects to MFF"""
    sampling_rate = 250.0
    startdatetime = datetime(1999, 12, 25, 8, 30, 10, tzinfo=timezone.utc)
    bads = [34, 40, 53]
    data_a = np.random.randn(129, 11).astype(np.float32)
    data_b = np.random.randn(129, 11).astype(np.float32)
    averages = [
        Average('Category A', [data_a], center=5, sr=sampling_rate, bads=bads),
        Average('Category B', [data_b], center=5, sr=sampling_rate, bads=bads)
    ]
    filepath = join('.cache', 'test_averaged.mff')
    W = write_averaged(averages, filepath, startdatetime=startdatetime,
                       device='HydroCel GSN 256 1.0')
    assert W.filename == filepath
    assert W.num_bin_files == 1
    expected_files = [
        'info.xml', 'coordinates.xml', 'sensorLayout.xml', 'signal1.bin',
        'info1.xml', 'epochs.xml', 'categories.xml'
    ]
    for file in expected_files:
        assert file in W.files
    # Check the written file
    R = Reader(filepath)
    assert R.sampling_rates['EEG'] == int(sampling_rate)
    assert R.startdatetime == startdatetime
    expected_signals = [data_a, data_b]
    for i in range(2):
        signals, t0 = R.get_physical_samples_from_epoch(R.epochs[i])['EEG']
        assert signals == pytest.approx(expected_signals[i])
    expected_categories = {
        'Category A': [
            {
                'status': 'unedited',
                'name': 'Average',
                'beginTime': 0,
                'endTime': 44000,
                'evtBegin': 20000,
                'evtEnd': 20000,
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
                        'data': 1
                    }
                }
            }
        ],
        'Category B': [
            {
                'status': 'unedited',
                'name': 'Average',
                'beginTime': 44000,
                'endTime': 88000,
                'evtBegin': 64000,
                'evtEnd': 64000,
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
                        'data': 1
                    }
                }
            }
        ],
    }
    assert R.categories.categories == expected_categories
    # Clean up written files
    try:
        for file in expected_files:
            remove(join(filepath, file))
        rmdir(filepath)
    except BaseException:
        raise AssertionError(f'Clean-up of "{filepath}" failed. '
                             f'Were additional files written?')


def test_write_averaged_bad_input() -> None:
    """Test proper errors are thrown for bad input"""
    # Test non-matching data shapes
    startdatetime = datetime(1999, 12, 25, 8, 30, 10, tzinfo=timezone.utc)
    device = 'HydroCel GSN 256 1.0'
    data_a = np.random.randn(32, 12).astype(np.float32)
    data_b = np.random.randn(32, 11).astype(np.float32)
    averages = [
        Average('a', [data_a], center=5, sr=250.0),
        Average('b', [data_b], center=5, sr=250.0)
    ]
    with pytest.raises(ValueError) as exc_info1:
        write_averaged(averages, 'writeme.mff', startdatetime, device)
    message = 'Averaged data blocks of different shape: (32, 11) != (32, 12)'
    assert str(exc_info1.value) == message

    # Test non-matching sampling rates
    data_c = np.random.randn(32, 12).astype(np.float32)
    averages = [
        Average('a', [data_a], center=5, sr=250.0),
        Average('c', [data_c], center=5, sr=100.0)
    ]
    with pytest.raises(ValueError) as exc_info2:
        write_averaged(averages, 'writeme.mff', startdatetime, device)
    message = 'Averages have different sampling rates: 100.0 != 250.0'
    assert str(exc_info2.value) == message
