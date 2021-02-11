from typing import List, Dict

import numpy as np


class Average:
    """Create an EEG average from list of segments"""

    def __init__(self, category: str, segments: List[np.array],
                 center: int, sr: float, bads: List[int] = []):
        """Create instance of `Average`

        Parameters
        ----------
        category
            Category label for the average
        segments
            Signal data for the segments going into the average
        center
            The position of the event for which the average is generated
            in samples relative to the beginning of the averaged data block
        sr
            Sampling rate of the data (cycles/sec)
        bads
            List of bad channels to be marked bad in the averaged data

        Raises
        ------
        ValueError
            If `center` is larger than the length of the averaged data
        """
        self.category = category
        self.segments = segments
        self.center = center
        self.sampling_rate = sr
        self.bads = bads
        if self.center > self.num_samples():
            raise ValueError(f'Center ({self.center}) cannot be larger than '
                             f'length of data block ({self.num_samples()})')

    def num_samples(self) -> int:
        """Return the duration of the averaged data in samples"""
        data = self.data()
        return int(data.shape[1])

    def num_segments(self) -> int:
        """Return number of segments going into average"""
        return len(self.segments)

    def data(self) -> np.array:
        """Return an array with the averaged data"""
        return np.mean(np.array(self.segments), axis=0)

    def build_category_content(self, idx: int) -> List[Dict[str, object]]:
        """construct category content dict for the average

        Parameters
        ----------
        idx
            The position of the average in relation to other averages to be
            written. This is used to determine the timing (begin time, end
            time, event time) described in the category content dict.

        Returns
        -------
        The category content for the average. This can be added to a dictionary
        with [category: category content], which can then be written out as a
        categories.xml file.
        """
        # Times are given in microseconds
        duration = int(1e6 * self.num_samples() / self.sampling_rate)
        begin_time = duration * idx
        end_time = begin_time + duration
        event_time = begin_time + int(1e6 * self.center / self.sampling_rate)
        num_segments = self.num_segments()
        bad_channels = self.bads
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
