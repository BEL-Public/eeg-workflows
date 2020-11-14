from typing import List

import numpy as np

from mne import Evoked
from mne.utils import object_diff
from mne.io.constants import FIFF
from mffpy.writer import Writer, BinWriter


def evokeds_to_writer(evokeds: List[Evoked], outfile: str,
                      recording_device: str) -> Writer:
    """convert list of mne.Evoked objects to mffpy.Writer

    Parameters
    ----------
    evokeds : list[mne.Evoked]
        List of Evoked objects to be written to .mff.
    outfile : str
        Path to output .mff file.
    recording_device : str
        Recording device type (e.g. 'Hydrocel GSN 256 1.0').

    Returns
    -------
    W : instance of mffpy.Writer
        The Writer object has all necessary files added and is ready to write.

    Raises
    ------
    AssertionError
        If Evoked.info is not identical across all Evoked objects in
        ``evokeds``.
    AssertionError
        If no EEG channels are found in the Evoked objects.
    """
    # All evoked objects should have the same info
    evokeds_info = evokeds[0].info
    for evoked in evokeds:
        diff = object_diff(evoked.info, evokeds_info)
        if diff != '':
            raise AssertionError(
                f'Measurement info for category {evoked.comment} different '
                f'than category {evokeds[0].comment}.\nDifference: {diff}'
            )
    sampling_rate = int(evokeds_info['sfreq'])
    W = Writer(outfile)
    W.addxml('fileInfo', recordTime=evokeds_info['meas_date'])
    W.add_coordinates_and_sensor_layout(device=recording_device)

    # Get EEG channel indices
    eeg_channels = []
    for ch in evokeds_info['chs']:
        if ch['kind'] == FIFF.FIFFV_EEG_CH:
            eeg_channels.append(ch['scanno'] - 1)
    assert len(eeg_channels) > 0, \
        'No EEG channels found in averaged data.\n' \
        f'Channels present: {evokeds_info["ch_names"]}'

    # Add EEG data
    eeg_bin = BinWriter(sampling_rate=sampling_rate)
    for evoked in evokeds:
        eeg_bin.add_block(get_data_block(evoked, eeg_channels),
                          offset_us=0)
    W.addbin(eeg_bin)

    # Add category info
    categories_content = {
        evokeds[i].comment: build_category_content(evokeds[i], i)
        for i in range(len(evokeds))
    }
    W.addxml('categories', categories=categories_content)

    return W


def get_data_block(evoked: Evoked, channels) -> np.array:
    """return block of data contained in ``evoked``

    Read signals from ``evoked`` and add return array of signals matching
    indices in ``channels``.
    """
    signals = evoked.data * 1e6  # convert to µV
    signals = np.array([signals[idx] for idx in channels], dtype=np.float32)
    return signals


def build_category_content(evoked: Evoked, idx):
    """construct content dict for evoked category"""
    num_samples = evoked.data.shape[1]
    duration = int(1e6 * num_samples / evoked.info['sfreq'])  # microseconds
    begin_time = duration * idx
    end_time = begin_time + duration
    event_time = begin_time - int(evoked.tmin * 1e6)
    num_segments = evoked.nave
    bad_channels = []
    for ch in evoked.info['chs']:
        if ch['kind'] == FIFF.FIFFV_EEG_CH and \
                ch['ch_name'] in evoked.info['bads']:
            bad_channels.append(ch['scanno'])
    return [
        {
            'status': 'unedited',
            'name': 'Average',
            'beginTime': begin_time,
            'endTime': end_time,
            'evtBegin': event_time,
            'evtEnd': event_time,
            'channelStatus': [
                {
                    'signalBin': 1,
                    'exclusion': 'badChannels',
                    'channels': bad_channels
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
