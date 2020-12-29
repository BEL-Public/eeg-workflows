import pytest
from datetime import datetime, timezone
from os import rmdir, remove
from os.path import join

import numpy as np
from mne import EvokedArray, create_info
from mffpy import Reader

from ..write import evokeds_to_writer


def test_evokeds_to_writer() -> None:
    """test converting mne.Evoked object to mffpy.Writer"""
    sampling_rate = 250.0
    startdatetime = datetime(1999, 12, 25, 8, 30, 10, tzinfo=timezone.utc)
    ch_names = [f'E{i}' for i in range(1, 258)] + ['ECG', 'EMG']
    ch_types = ['eeg'] * 257 + ['ecg', 'emg']
    info = create_info(ch_names, sampling_rate, ch_types)
    info['meas_date'] = startdatetime
    info['bads'] = ['E3', 'E90', 'EMG']
    data_a = np.random.randn(259, 11)
    data_b = np.random.randn(259, 11)
    evokeds = [
        EvokedArray(data_a, info, tmin=-0.02, comment='Category A', nave=10),
        EvokedArray(data_b, info, tmin=-0.02, comment='Category B', nave=10)
    ]
    filepath = join('.cache', 'write_evokeds.mff')
    W = evokeds_to_writer(evokeds, filepath, 'Hydrocel GSN 256 1.0')
    assert W.filename == filepath
    assert W.num_bin_files == 1
    expected_files = [
        'info.xml', 'coordinates.xml', 'sensorLayout.xml', 'signal1.bin',
        'info1.xml', 'epochs.xml', 'categories.xml'
    ]
    for file in expected_files:
        assert file in W.files
    # Write the .mff and check the result
    W.write()
    R = Reader(filepath)
    assert R.sampling_rates['EEG'] == int(sampling_rate)
    assert R.startdatetime == startdatetime
    expected_signals = [data_a * 1e6, data_b * 1e6]
    for i in range(2):
        signals, t0 = R.get_physical_samples_from_epoch(R.epochs[i])['EEG']
        assert signals == pytest.approx(expected_signals[i][0:257])
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
                        'channels': [3, 90]
                    }
                ],
                'keys': {
                    '#seg': {
                        'type': 'long',
                        'data': 10
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
                        'channels': [3, 90]
                    }
                ],
                'keys': {
                    '#seg': {
                        'type': 'long',
                        'data': 10
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


def test_evokeds_to_writer_bad_input() -> None:
    """test proper errors are thrown for bad input"""
    # Test non-matching infos
    ch_names = [f'E{i}' for i in range(1, 258)]
    ch_types = ['eeg'] * 257
    info1 = create_info(ch_names, 250.0, ch_types)
    info2 = create_info(ch_names, 100.0, ch_types)
    data = np.random.randn(257, 5)
    evokeds = [
        EvokedArray(data, info1, comment='Category A'),
        EvokedArray(data, info2, comment='Category B')
    ]
    with pytest.raises(AssertionError) as exc_info:
        evokeds_to_writer(evokeds, 'writeme.mff', 'HydroCel GSN 256 1.0')
    message = "Measurement info for category Category B different than " \
              "category Category A.\nDifference: ['lowpass'] value mismatch " \
              "(50.0, 125.0)\n['sfreq'] value mismatch (100.0, 250.0)\n"
    assert str(exc_info.value) == message
    # Test no EEG channels
    ch_names = ['EMG', 'ECG']
    ch_types = ['emg', 'ecg']
    info = create_info(ch_names, 250.0, ch_types)
    info['meas_date'] = datetime(1999, 12, 25, 8, 30, 10, tzinfo=timezone.utc)
    data = np.random.randn(2, 5)
    evokeds = [EvokedArray(data, info)]
    with pytest.raises(AssertionError) as exc_info:
        evokeds_to_writer(evokeds, 'writeme.mff', 'HydroCel GSN 256 1.0')
    message = "No EEG channels found in averaged data.\n" \
              "Channels present: ['EMG', 'ECG']"
    assert str(exc_info.value) == message
