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
import pytest
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from os import remove, rmdir
from os.path import join
from typing import List

import numpy as np
from mffpy import Reader, XML
import pytz

from ..average import _average_reference, Averages
from ..segment import seconds_to_samples


@pytest.fixture
def empty_averages() -> Averages:
    """Return `Averages` object with no averages added"""
    center = 0.05
    sr = 100.0
    bads = [4, 2, 17]
    averages = Averages(center=center, sr=sr, bads=bads)
    return averages


@pytest.fixture
def loaded_averages(empty_averages: Averages) -> Averages:
    """Return `Averages` object with averages added"""
    shape = (32, 10)
    num_segments = 5
    for category in ['cat1', 'cat2', 'cat3']:
        segments = [
            np.random.standard_normal(shape).astype(np.float32)
        ] * num_segments
        empty_averages.add(category, segments)
    return empty_averages


@pytest.mark.parametrize('bads,result', [
    ([], np.array([[-3, 32, -3.75, 11, -4.75],
                   [-8, -9, 69.25, -78, -6.75],
                   [15, -10, -2.75, 19, 14.25],
                   [-4, -13, -62.75, 48, -2.75]])),
    ([2, 4], np.array([[-9, 21, -0.5, -4, -9.5],
                       [-14, -20, 72.5, -93, -11.5],
                       [9, -21, 0.5, 4, 9.5],
                       [-10, -24, -59.5, 33, -7.5]])),
    ([1, 2, 3], np.array([[1, 45, 59, -37, -2],
                          [-4, 4, 132, -126, -4],
                          [19, 3, 60, -29, 17],
                          [0, 0, 0, 0, 0]])),
])
def test_average_reference(bads: List[int], result: np.ndarray) -> None:
    """Test applying average reference to signal block"""
    data = np.array(
        [[10, 43, 5, -5, 2],
         [5, 2, 78, -94, 0],
         [28, 1, 6, 3, 21],
         [9, -2, -54, 32, 4]]
    ).astype(np.float32)
    assert _average_reference(data, bads) == pytest.approx(result)


def test_average_reference_bad_input() -> None:
    """Test average reference throws errors for bad inputs"""
    # Test non-2D array
    shape = (2, 2, 2, 2)
    data = np.random.standard_normal(shape)
    with pytest.raises(ValueError) as exc_info:
        _average_reference(data, [])
    message = f'Input array must be 2-dimensional. Got shape: {shape}'
    assert str(exc_info.value) == message
    # Test all channels bad
    num_channels = 10
    data = np.random.rand(num_channels, 5)
    bads = list(range(1, num_channels + 1))
    with pytest.raises(ValueError) as exc_info:
        _average_reference(data, bads)
    message = 'No good channels from which to build reference'
    assert str(exc_info.value) == message


def test_averages_class_init() -> None:
    """Test initializing `Averages` object"""
    center = 1.0
    sr = 250.0
    bads = [33, 21, 2]
    averages = Averages(center=center, sr=sr, bads=bads)
    assert averages.center == center
    assert averages.sampling_rate == sr
    assert averages.bads == bads
    assert averages.data == OrderedDict()
    assert averages.num_segments == OrderedDict()


def test_add_averages(empty_averages: Averages) -> None:
    """Test adding averages to `Averages` object"""
    shape = (5, 10)
    num_segments = 3
    for category in ['cat1', 'cat2', 'cat3']:
        segments = [
            np.random.standard_normal(shape).astype(np.float32)
        ] * num_segments
        empty_averages.add(category, segments)
        average_expected = np.mean(np.array(segments), axis=0)
        assert empty_averages.data[category] == pytest.approx(average_expected)
        assert empty_averages.num_segments[category] == num_segments


def test_add_segments_bad_shape(empty_averages: Averages) -> None:
    """Test adding non-2D shaped segments throws ValueError"""
    shape = (3, 3, 3)
    category = 'catx'
    segments = [np.random.standard_normal(shape).astype(np.float32)]
    with pytest.raises(ValueError) as exc_info:
        empty_averages.add(category, segments)
    message = f'Segments must be 2-dimensional. Got shape: {shape}'
    assert str(exc_info.value) == message


def test_add_segments_of_different_shapes(empty_averages: Averages) -> None:
    """Test adding average from segments of different shapes"""
    shape_1 = (5, 10)
    shape_2 = (shape_1[0] - 1, shape_1[1] - 1)
    category = 'catx'
    segments = [
        np.random.standard_normal(shape_1).astype(np.float32),
        np.random.standard_normal(shape_2).astype(np.float32)
    ]
    with pytest.raises(ValueError) as exc_info:
        empty_averages.add(category, segments)
    message = f'Segments have different shapes: {shape_2} != {shape_1}'
    assert str(exc_info.value) == message


def test_add_length_smaller_than_center(empty_averages: Averages) -> None:
    """Test adding average with length smaller than center"""
    center_samples = seconds_to_samples(empty_averages.center,
                                        empty_averages.sampling_rate)
    length = center_samples - 1
    category = 'catx'
    segments = [np.random.randn(5, length).astype(np.float32)]
    with pytest.raises(ValueError) as exc_info:
        empty_averages.add(category, segments)
    message = f'Center ({center_samples} samples) cannot be larger than ' \
              f'length of the averaged data block ({length} samples)'
    assert str(exc_info.value) == message


def test_add_average_of_different_shape(loaded_averages: Averages) -> None:
    """Test adding average of different shape than previously added"""
    shape = next(iter(loaded_averages.data.values())).shape
    different_shape = (shape[0] + 1, shape[1] + 1)
    category = 'catx'
    segments = [np.random.standard_normal(different_shape).astype(np.float32)]
    with pytest.raises(ValueError) as exc_info:
        loaded_averages.add(category, segments)
    message = 'Attempting to add averaged data block of different shape ' \
              f'[{different_shape}] than previously added blocks ' \
              f'[{shape}]'
    assert str(exc_info.value) == message


def test_set_average_reference(loaded_averages: Averages) -> None:
    """Test applying average reference to `Averages` object"""
    vref_data = deepcopy(loaded_averages.data)
    loaded_averages.set_average_reference()
    for label, signals in loaded_averages.data.items():
        expected_signals = _average_reference(vref_data[label],
                                              loaded_averages.bads)
        assert signals == pytest.approx(expected_signals)
    # Make sure average reference is applied to newly added average
    category = 'catx'
    shape = next(iter(loaded_averages.data.values())).shape
    segment = np.random.standard_normal(shape).astype(np.float32)
    loaded_averages.add(category, [segment])
    expected_signals = _average_reference(segment, loaded_averages.bads)
    assert loaded_averages.data[category] == \
        pytest.approx(expected_signals)


def test_build_categories_content(loaded_averages: Averages) -> None:
    """Test building categories content dict from `Averages` object"""
    expected_content = {
        'cat1': [
            {
                'status': 'unedited',
                'name': 'Average',
                'beginTime': 0,
                'endTime': 100_000,
                'evtBegin': 50_000,
                'evtEnd': 50_000,
                'channelStatus': [
                    {
                        'signalBin': 1,
                        'exclusion': 'badChannels',
                        'channels': loaded_averages.bads
                    }
                ],
                'keys': {
                    '#seg': {
                        'type': 'long',
                        'data': 5
                    }
                }
            }
        ],
        'cat2': [
            {
                'status': 'unedited',
                'name': 'Average',
                'beginTime': 100_000,
                'endTime': 200_000,
                'evtBegin': 150_000,
                'evtEnd': 150_000,
                'channelStatus': [
                    {
                        'signalBin': 1,
                        'exclusion': 'badChannels',
                        'channels': loaded_averages.bads
                    }
                ],
                'keys': {
                    '#seg': {
                        'type': 'long',
                        'data': 5
                    }
                }
            }
        ],
        'cat3': [
            {
                'status': 'unedited',
                'name': 'Average',
                'beginTime': 200_000,
                'endTime': 300_000,
                'evtBegin': 250_000,
                'evtEnd': 250_000,
                'channelStatus': [
                    {
                        'signalBin': 1,
                        'exclusion': 'badChannels',
                        'channels': loaded_averages.bads
                    }
                ],
                'keys': {
                    '#seg': {
                        'type': 'long',
                        'data': 5
                    }
                }
            }
        ]
    }
    assert loaded_averages.build_category_content() == expected_content


def test_build_categories_content_empty(empty_averages: Averages) -> None:
    """Test building categories content fails if no averages added"""
    with pytest.raises(ValueError) as exc_info:
        empty_averages.build_category_content()
    assert str(exc_info.value) == 'No averages have been added'


def test_write_averages_to_mff(loaded_averages: Averages) -> None:
    """Test writing `Averages` objects to MFF"""
    outfile = join('.cache', 'test_averaged.mff')
    startdatetime = datetime(1999, 12, 25, 8, 30, 10, tzinfo=timezone.utc)
    device = 'HydroCel GSN 32 1.0'
    history = [
        {
            'name': 'ERP Workflow Segmentation',
            'method': 'Segmentation',
            'beginTime': pytz.utc.localize(datetime.utcnow()),
            'endTime': pytz.utc.localize(datetime.utcnow()),
            'sourceFiles': ['test.mff'],
            'settings': ['Setting 1', 'Setting 2'],
            'results': ['Result 1', 'Result 2']
        },
        {
            'name': 'ERP Workflow Averaging',
            'method': 'Averaging',
            'beginTime': pytz.utc.localize(datetime.utcnow()),
            'endTime': pytz.utc.localize(datetime.utcnow()),
            'sourceFiles': ['test.mff'],
            'settings': ['Setting 1', 'Setting 2'],
            'results': ['Result 1', 'Result 2']
        }
    ]
    loaded_averages.write_to_mff(outfile, startdatetime=startdatetime,
                                 device=device, history=history)
    R = Reader(outfile)
    assert R.sampling_rates['EEG'] == loaded_averages.sampling_rate
    assert R.startdatetime == startdatetime
    for category, expected_signals in loaded_averages.data.items():
        epoch = R.epochs[category]
        signals = R.get_physical_samples_from_epoch(epoch)['EEG'][0]
        assert signals == pytest.approx(expected_signals)
    expected_categories = loaded_averages.build_category_content()
    assert R.categories.categories == expected_categories
    with R.directory.filepointer('history') as fp:
        assert XML.from_file(fp).entries == history
    # Clean up written files
    try:
        for file in R.directory.listdir():
            remove(join(outfile, file))
        rmdir(outfile)
    except BaseException:
        raise AssertionError(f'Clean-up of "{outfile}" failed. '
                             f'Were additional files written?')
