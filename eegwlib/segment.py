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
    OutOfRangeError
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
        raise OutOfRangeError('Requested segment extends '
                              'beyond data block')
    if segment_stop_idx > arr.shape[1]:
        raise OutOfRangeError('Requested segment extends '
                              'beyond data block')
    segment_indices = np.array(range(segment_start_idx, segment_stop_idx))
    return arr.take(segment_indices, axis=1)


def seconds_to_samples(seconds: float, sr: float) -> int:
    """Convert seconds to samples, rounding to the nearest sample"""
    return int(np.round(seconds * sr))


class OutOfRangeError(Exception):
    """Raised when a requested block slice extends beyond data range"""
    pass


class Segmenter(mffpy.Reader):  # type: ignore
    """Subclass of `mffpy.Reader` that adds functionality to extract
    segments"""

    def __init__(self, filename: str) -> None:
        super().__init__(filename)
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
            If no data is loaded
        """
        assert self.data_cache is not None, 'No data loaded'
        return self.data_cache

    def load_filtered_epoch(self, epoch: Epoch, order: int,
                            fmin: Optional[float],
                            fmax: Optional[float]) -> None:
        """Read and filter all EEG data in `epoch`, load into data cache

        Parameters
        ----------
        epoch
            The epoch to be loaded
        order
            Filter order
        fmin
            Lower critical frequency (Hz) for IIR filter
        fmax
            Upper critical frequency (Hz) for IIR filter

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
                                    order, fmin, fmax)

    def clear_loaded_data(self) -> None:
        """Clear data from data cache"""
        self._data_cache = None

    def extract_segments(self, relative_times: Dict[str, List[float]],
                         padl: float, padr: float, order: int = 4,
                         fmin: Optional[float] = None,
                         fmax: Optional[float] = None
                         ) -> Tuple[Dict[str, List[np.ndarray]],
                                    Dict[str, List[float]]]:
        """Extract segments around relative times

        If `fmin` and/or `fmax` are specified, the raw signals are filtered
        before segments are extracted.

        Parameters
        ----------
        relative_times
            Dictionary of {category name: times (seconds)}. Times are relative
            to start of recording.
        padl
            Left time padding (seconds)
        padr
            Right time padding (seconds)
        order
            Filter order
        fmin
            Lower critical frequency (Hz) for IIR filter
        fmax
            Upper critical frequency (Hz) for IIR filter

        Returns
        -------
        segments
            Dictionary of {category name: segments} with the segments that
            were successfully extracted
        out_of_range_segs
            Dictionary of {category name: relative times} with relative times
            for which segments extended beyond the data range and could not
            be extracted

        Raises
        ------
        ValueError
            If `padl` or `padr` are negative
        """
        for var, value in {'left': padl, 'right': padr}.items():
            if value < 0:
                raise ValueError(f'Negative {var} padding: {value}')
        segments = defaultdict(list)
        out_of_range_segs = defaultdict(list)
        for epoch in self.epochs:
            self.clear_loaded_data()
            for cat, times in relative_times.items():
                for time in times:
                    if self.is_in_epoch(time, epoch):
                        time_relative_to_epoch = time - epoch.t0
                        if self.data_cache is None:
                            self.load_filtered_epoch(epoch, order, fmin, fmax)
                        try:
                            segment = self._extract_segment_from_loaded_data(
                                time_relative_to_epoch, padl, padr)
                            segments[cat].append(segment)
                        except OutOfRangeError:
                            out_of_range_segs[cat].append(time)
        self.clear_loaded_data()
        return segments, out_of_range_segs

    def is_in_epoch(self, relative_time: float, epoch: Epoch) -> bool:
        """Return `True` if `epoch` contains `relative_time`"""
        if epoch.t0 <= relative_time < epoch.t1:
            return True
        else:
            return False

    def _extract_segment_from_loaded_data(self, center: float, padl: float,
                                          padr: float) -> np.ndarray:
        """Extract a segment from data block in data cache"""
        return extract_segment_from_array(self.get_loaded_data(), center, padl,
                                          padr, self.sampling_rates['EEG'])
