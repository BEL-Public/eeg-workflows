from typing import Dict, List, Optional, Sequence
from collections import OrderedDict
from datetime import datetime

import numpy as np
from mffpy.writer import BinWriter, Writer


def _average_reference(data: np.ndarray, bads: List[int]) -> np.ndarray:
    """Rereference `data` to an average reference

    Parameters
    ----------
    data
        Block of signal data with shape (channels, samples)
    bads
        Channel to exclude from the average reference. These are interpreted
        as channel numbers starting from 1 (i.e. 1, 2, 3, ... 257).

    Returns
    -------
    The average referenced data

    Raises
    ------
    ValueError
        If `data` is not a 2-dimensional array
    ValueError
        If all channels in `data` are bad
    """
    if len(data.shape) != 2:
        raise ValueError('Input array must be 2-dimensional. '
                         f'Got shape: {data.shape}')
    channels = np.arange(1, data.shape[0] + 1)
    channels_mask = np.isin(channels, bads, invert=True)
    if not channels_mask.any():
        raise ValueError('No good channels from which to build reference')
    return data - np.mean(data[channels_mask], axis=0)


class Averages:
    """Class for generating EEG averages with shared properties

    For all averages that are added to an `Averages` object, the following
    properties must be the same:

    - Sampling rate of the data
    - Shape of the averaged data block
    - Position of the event around which the average is created
    """

    def __init__(self, center: int, sr: float, bads: List[int] = []) -> None:
        """Create instance of `Averages`

        Parameters
        ----------
        center
            The position of the event around which the average is created
            in samples relative to the beginning of the averaged data block.
            This property is shared for all added averages.
        sr
            Sampling rate (cycles/sec)
        bads
            List of channels that are bad across all added averages
        """
        self.center = center
        self.sampling_rate = sr
        self.bads = bads
        self._data: Dict[str, np.ndarray] = OrderedDict()
        self._num_segments: Dict[str, int] = OrderedDict()
        self._average_reference_on = False

    def add(self, label: str, segments: List[np.ndarray]) -> None:
        """Generate an individual average from a list of segments

        `self._data` is updated with {label: averaged data block}.
        `self._num_segments` is updated with {label: number of segments}.

        Parameters
        ----------
        label
            Category label for the average to be added
        segments
            Signal data for the segments going into the average

        Raises
        ------
        ValueError
            If segments are not all 2-dimensional arrays
        ValueError
            If `segments` contains arrays of differing shape
        ValueError
            If `self.center` is larger than the length of the averaged data
        ValueError
            If shape of the averaged data block is different from previously
            added averages
        """
        for segment in segments:
            if len(segment.shape) != 2:
                raise ValueError('Segments must be 2-dimensional. '
                                 f'Got shape: {segment.shape}')
            if segment.shape != segments[0].shape:
                raise ValueError('Segments have different shapes: '
                                 f'{segment.shape} != {segments[0].shape}')
        average = np.mean(np.array(segments), axis=0)
        if len(self.data) == 0:
            # This is the first average being added
            if self.center > average.shape[1]:
                raise ValueError(f'Center ({self.center}) cannot be larger '
                                 'than length of the averaged data block '
                                 f'({average.shape[1]})')
        else:
            # There are previously added averages
            first_average = next(iter(self.data.values()))
            if average.shape != first_average.shape:
                raise ValueError('Attempting to add averaged data block of '
                                 f'different shape [{average.shape}] than '
                                 'previously added blocks '
                                 f'[{first_average.shape}]')
        if self._average_reference_on:
            average = _average_reference(average, self.bads)
        self._data[label] = average
        self._num_segments[label] = len(segments)

    @property
    def data(self) -> Dict[str, np.ndarray]:
        """Get the averaged data blocks"""
        return self._data

    @property
    def num_segments(self) -> Dict[str, int]:
        """Return number of segments going into each average"""
        return self._num_segments

    def set_average_reference(self) -> None:
        """Apply an average reference to each averaged data block

        Future added averages will also have an average reference applied
        """
        for label, data in self.data.items():
            self._data[label] = _average_reference(data, self.bads)
        self._average_reference_on = True

    def build_category_content(self) -> Dict[str, List[Dict[str, object]]]:
        """Construct category content dict for the averages

        Returns
        -------
        The categories content for all averaged data blocks in `self.data`.
        This can be written out as a categories.xml file.

        Raises
        ------
        ValueError
            If no averages have been added
        """
        if len(self.data) == 0:
            raise ValueError('No averages have been added')
        content = {}
        # Times are given in microseconds
        begin_time = 0
        for category, num_segments in self.num_segments.items():
            num_samples = self.data[category].shape[1]
            duration = int(1e6 * num_samples / self.sampling_rate)
            end_time = begin_time + duration
            event_time = begin_time + \
                int(1e6 * self.center / self.sampling_rate)
            content[category] = [
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
                            'channels': self.bads
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
            begin_time += duration

        return content

    def write_to_mff(self, outfile: str, startdatetime: datetime, device: str,
                     history: Optional[List[Dict[str, Sequence[str]]]] = None
                     ) -> None:
        """Write the averaged data to MFF

        Parameters
        ----------
        outfile
            Path to which the averaged MFF will be written
        startdatetime
            Timestamp of recording start for the raw MFF from which the
            averages were generated
        device
            Recording device for the raw MFF from which the averages were
            generated
        history
            Content to be written to `history.xml` file. See method
            `mffpy.xml_files.History.content` for proper format
        """
        W = Writer(outfile)
        W.addxml('fileInfo', recordTime=startdatetime)
        W.add_coordinates_and_sensor_layout(device=device)
        eeg_bin = BinWriter(sampling_rate=int(self.sampling_rate))
        for average in self.data.values():
            eeg_bin.add_block(average, offset_us=0)
        W.addbin(eeg_bin)
        W.addxml('categories', categories=self.build_category_content())
        if history:
            W.addxml('historyEntries', entries=history)
        W.write()
