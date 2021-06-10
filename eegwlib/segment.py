"""
Copyright 2021 Brain Electrophysiology Laboratory Company LLC

Licensed under the ApacheLicense, Version 2.0(the "License");
you may not use this module except in compliance with the License.
You may obtain a copy of the License at:

http: // www.apache.org / licenses / LICENSE - 2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import mffpy
from mffpy.epoch import Epoch
import numpy as np

from .filter import filtfilt


def extract_segment_from_array(arr: np.ndarray, center: float, padl: float,
                               padr: float, sr: float) -> np.ndarray:
    """Extract a segment from array of signal data

    Parameters
    ----------
    arr
        Array from which to extract segment
    center
        The center of the segment in seconds relative to the beginning of `arr`
    padl
        Left time padding (seconds)
    padr
        Right time padding (seconds)
    sr
        Sampling rate of the signal data

    Returns
    -------
    The extracted segment

    Raises
    ------
    ValueError
        If the loaded data block is not a 2-dimensional array
    IndexError
        If the requested segment extends beyond the data block
    """
    if arr.ndim != 2:
        raise ValueError('Input array must be 2-dimensional. '
                         f'Got shape: {arr.shape}')
    # Start index of right side of segment
    right_start_idx = seconds_to_samples(center, sr)
    left_samples = seconds_to_samples(padl, sr)
    right_samples = seconds_to_samples(padr, sr)
    # Start index of whole segment
    segment_start_idx = right_start_idx - left_samples
    # Stop index of whole segment
    segment_stop_idx = right_start_idx + right_samples
    if segment_start_idx < 0:
        raise IndexError('Requested segment extends beyond data block')
    if segment_stop_idx > arr.shape[1]:
        raise IndexError('Requested segment extends beyond data block')
    return arr[:, segment_start_idx:segment_stop_idx]


def seconds_to_samples(seconds: float, sr: float) -> int:
    """Convert seconds to samples, rounding to the nearest sample"""
    return int(np.round(seconds * sr))


class Segmenter(mffpy.Reader):  # type: ignore
    """Subclass of `mffpy.Reader` that adds functionality to extract
    segments"""

    def __init__(self, filename: str, padl: float, padr: float,
                 order: int = 4, fmin: Optional[float] = None,
                 fmax: Optional[float] = None) -> None:
        """Initialize `Segmenter` object

        `order`, `fmin`, and `fmax` specify filter parameters for the filter
        to be applied to the raw signals before extracting segments. No filter
        will be applied if `fmin` and `fmax` are both set to `None`.

        Parameters
        ----------
        padl
            Left time padding (seconds) of data segments to be extracted
        padr
            Right time padding (seconds) of data segments to be extracted
        order
            Filter order
        fmin
            Lower critical frequency (Hz) for IIR filter
        fmax
            Upper critical frequency (Hz) for IIR filter

        Raises
        ------
        ValueError
            If `padl` or `padr` are negative
        """
        super().__init__(filename)
        for var, value in {'left': padl, 'right': padr}.items():
            if value < 0:
                raise ValueError(f'Negative {var} padding: {value}')
        self.padl = padl
        self.padr = padr
        self.order = order
        self.fmin = fmin
        self.fmax = fmax
        self._data_cache = None

    @property
    def data_cache(self) -> Optional[np.ndarray]:
        """Return value in data cache"""
        return self._data_cache

    def get_loaded_data(self) -> np.ndarray:
        """Retrieve loaded data block from data cache

        Raises
        ------
        AssertionError
            If no data are loaded
        """
        assert self.data_cache is not None, 'No data loaded'
        return self.data_cache

    def load_filtered_epoch(self, epoch: Epoch) -> None:
        """Read and filter all EEG data in `epoch`, load into data cache

        Parameters
        ----------
        epoch
            The epoch to be loaded

        Raises
        ------
        AssertionError
            If a data block is already loaded into data cache
        """
        assert self.data_cache is None, 'A data block is already loaded. ' \
                                        'First, clear loaded data.'
        data = self.get_physical_samples_from_epoch(epoch, channels=['EEG'])
        eeg_data = data['EEG'][0]
        self._data_cache = filtfilt(eeg_data, self.sampling_rates['EEG'],
                                    self.order, self.fmin, self.fmax)

    def clear_loaded_data(self) -> None:
        """Clear data from data cache"""
        self._data_cache = None

    def extract_segments(self, relative_times: Dict[str, List[float]]
                         ) -> Tuple[Dict[str, List[np.ndarray]],
                                    Dict[str, List[float]]]:
        """Extract segments around relative times

        Parameters
        ----------
        relative_times
            Dictionary of {category name: times (seconds)}. Times are relative
            to start of recording.

        Returns
        -------
        segments
            Dictionary of {category name: segments} with the segments that
            were successfully extracted
        out_of_range_segs
            Dictionary of {category name: relative times} with relative times
            for which segments extended beyond the data range and could not
            be extracted
        """
        times_by_epoch = self._sort_category_times_by_epoch(relative_times)
        segments = defaultdict(list)
        out_of_range_segs = defaultdict(list)
        for epoch, category_times in times_by_epoch.items():
            self.clear_loaded_data()
            self.load_filtered_epoch(epoch)
            for category, time in category_times:
                time_relative_to_epoch = time - epoch.t0
                try:
                    segment = self._extract_segment_from_loaded_data(
                        time_relative_to_epoch
                    )
                    segments[category].append(segment)
                except IndexError:
                    out_of_range_segs[category].append(time)
        self.clear_loaded_data()
        return segments, out_of_range_segs

    def _sort_category_times_by_epoch(self,
                                      times_by_category: Dict[str, List[float]]
                                      ) -> Dict[Epoch,
                                                List[Tuple[str, float]]]:
        """Sort dict of {category: times} into dict of {epoch: times}

        Each relative time is converted to a tuple with (category, time)
        to preserve the category names associated with each time
        """
        times_by_epoch = defaultdict(list)
        for cat, times in times_by_category.items():
            times = sorted(times)
            epoch_idx = 0
            for time in times:
                while not self.is_in_epoch(time, self.epochs[epoch_idx]):
                    epoch_idx += 1
                times_by_epoch[self.epochs[epoch_idx]].append((cat, time))
        assert sum(map(len, times_by_category.values())) == \
            sum(map(len, times_by_epoch.values()))
        return times_by_epoch

    def is_in_epoch(self, relative_time: float, epoch: Epoch) -> bool:
        """Return `True` if `epoch` contains `relative_time`"""
        return bool(epoch.t0 <= relative_time < epoch.t1)

    def _extract_segment_from_loaded_data(self, center: float) -> np.ndarray:
        """Extract a segment from data block in data cache"""
        return extract_segment_from_array(self.get_loaded_data(), center,
                                          self.padl, self.padr,
                                          self.sampling_rates['EEG'])
