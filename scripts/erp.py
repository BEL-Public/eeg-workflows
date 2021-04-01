#!/usr/bin/env python
"""Average a raw .mff file according to specified event markers.

The input file is read and searched for all provided labels. An IIR
Butterworth filter is applied to the raw signals if either `highpass`
or `lowpass` are specified by the user.  The order of the filter can be
specified via the `filter-order` argument.  Segments for all
event markers of specified labels with padding intervals `left-padding` and
`right-padding` are extracted from the input file.  If `artifact-detection`
argument is provided, bad segments are dropped based on the specified
peak-to-peak amplitude criterion.  Averaging is performed per label.
Then, the data are re-referenced to an average reference if `average-ref` flag
is present.  These are then written to the output file path as an .mff.
"""

from os.path import splitext, isdir, exists
from functools import partial
from typing import Dict, List, Union
from xml.etree.ElementTree import parse

import numpy as np
from mffpy import Reader, XML
from mffpy.xml_files import EventTrack

from eegwlib.filter import filtfilt
from eegwlib.segment import slice_block
from eegwlib.artifact_detection import detect_bad_channels
from eegwlib.average import Averages

from argparse import ArgumentParser

assert __name__ == "__main__"


def MffType(filepath: str, should_exist: bool = True) -> str:
    """check that filepath is an .mff"""
    filepath = str(filepath)
    base, ext = splitext(filepath)
    assert ext.lower() == '.mff', f"Unknown file type: '{filepath}'"
    if should_exist:
        assert exists(filepath), f"File not found: '{filepath}'"
        assert isdir(filepath), f"Not a folder: '{filepath}'"
    else:
        assert not exists(filepath), f"File exists: '{filepath}'"

    return filepath


def LabelStr(labels: str) -> List[str]:
    """Check that labels are valid and return as list"""
    labels = str(labels)
    labels_list = labels.split(',')
    labels_list = [label.strip() for label in labels_list if label.strip()]
    assert len(labels_list) > 0, "No labels specified"
    return labels_list


def FloatPositive(f: Union[str, float]) -> float:
    """Check that float is not negative"""
    f = float(f)
    assert f >= 0.0, f"Negative value: {f}"
    return f


def Frequency(freq: Union[str, float]) -> float:
    """Check that frequency value positive"""
    freq = float(freq)
    assert freq > 0.0, f"Non-positive frequency: {freq}"
    return freq


def FilterOrder(order: Union[str, int]) -> int:
    """Check that filter order is >= 1"""
    order = int(order)
    assert order >= 1, f"Filter order < 1: {order}"
    return order


parser = ArgumentParser(description=__doc__)
parser.add_argument('input_file', type=MffType,
                    help='Path to input .mff file')
parser.add_argument('output_file', type=partial(MffType, should_exist=False),
                    help='Path to output .mff file')
parser.add_argument('--labels', '-l', type=LabelStr, required=True,
                    help='Comma-separated list of event labels')
parser.add_argument('--left-padding', type=FloatPositive, default=1.0,
                    help='Padding prior to event in sec. (default=1.0)')
parser.add_argument('--right-padding', type=FloatPositive, default=1.0,
                    help='Padding after event in sec. (default=1.0)')
parser.add_argument('--highpass', '-hp', type=Frequency,
                    help='Highpass filter cutoff frequency (Hz)')
parser.add_argument('--lowpass', '-lp', type=Frequency,
                    help='Lowpass filter cutoff frequency (Hz)')
parser.add_argument('--filter-order', type=FilterOrder, default=4,
                    help='Filter order setting for the IIR Butterworth filter '
                         '(default=4). The signals are filtered twice - once '
                         'forward and once backward, so the effective filter '
                         'order is twice the specified value for a high-pass '
                         'or low-pass filter and 4x the specified value for a '
                         'band-pass or band-stop filter.')
parser.add_argument('--artifact-detection', type=FloatPositive,
                    help='Peak-to-peak amplitude criterion for bad segment '
                         'rejection in μV. A segment will be dropped if any '
                         'channels exceed the amplitude criterion. '
                         'Bad segments will be dropped only if this '
                         'argument is provided.')
parser.add_argument('--average-ref', action='store_true',
                    help='Set average reference')
parser.add_argument('--verbose', '-v', action='store_true',
                    help='Print settings and results for each step')
opt = parser.parse_args()

# Read raw input file
raw = Reader(opt.input_file)
sampling_rate = raw.sampling_rates['EEG']

# Get history info if present
if 'history.xml' in raw.directory.listdir():
    with raw.directory.filepointer('history') as fp:
        history = XML.from_file(fp).entries
else:
    history = []

# Get raw signal blocks and apply filter if specified
data = []
for epoch in raw.epochs:
    signals = raw.get_physical_samples_from_epoch(epoch)['EEG'][0]
    filtered_signals = filtfilt(signals, order=opt.filter_order,
                                sr=sampling_rate, fmin=opt.highpass,
                                fmax=opt.lowpass)
    data.append(
        {
            't0': epoch.t0,
            't1': epoch.t1,
            'data': filtered_signals
        }
    )

for filt, freq in {'Highpass': opt.highpass, 'Lowpass': opt.lowpass}.items():
    if freq:
        filter_entry = dict(
            method='Filtering',
            settings=[f'Filter Setting: {freq} Hz {filt}',
                      'Filter Type: IIR Butterworth',
                      f'Filter Order: {opt.filter_order}']
        )
        if opt.verbose:
            print(filter_entry)
        history.append(filter_entry)

# Extract relative times for events of specified labels
all_codes = set()
event_times: Dict[str, List[float]] = {label: [] for label in opt.labels}
for file in raw.directory.files_by_type['.xml']:
    with raw.directory.filepointer(splitext(file)[0]) as fp:
        xml_root = parse(fp).getroot()
        if not xml_root.tag == '{http://www.egi.com/event_mff}eventTrack':
            continue
        events = EventTrack(xml_root).events
        for event in events:
            all_codes.add(event['code'])
            if event['code'] in opt.labels:
                event_times[event['code']].append((
                    event['beginTime'] - raw.startdatetime
                ).total_seconds())

event_times_sorted = {}
for label, times in event_times.items():
    if len(times) == 0:
        raise ValueError(f'Label "{label}" not found among events.\n'
                         f'Valid event labels: {all_codes}')
    event_times_sorted[label] = sorted(times)

# Extract data segments
out_of_bounds_segs: Dict[str, List[float]] = {
    label: [] for label in event_times_sorted
}
segments: Dict[str, List[np.ndarray]] = {
    label: [] for label in event_times_sorted
}
for label, times in event_times_sorted.items():
    block_idx = 0
    block = data[block_idx]
    for time in times:
        while time > block['t1']:
            # Iterate through data blocks until we
            # find the one that contains time
            block_idx += 1
            block = data[block_idx]
        assert block['t0'] < time < block['t1']
        # Extract the segment centered on time
        segment = slice_block(
            block['data'],
            center=time - block['t0'],
            padl=opt.left_padding,
            padr=opt.right_padding,
            sr=sampling_rate
        )
        if segment is not None:
            segments[label].append(segment)
        else:
            out_of_bounds_segs[label].append(time)

# Check if any categories have no segments
for label, segs in segments.items():
    if len(segs) == 0:
        raise ValueError(f'All segments for event type "{label}" '
                         'extended beyond data range')

segmentation_settings = []
segmentation_results = [f'Segmented to {len(segments)} categories and '
                        f'{sum(map(len, segments.values()))} segments']
for category, segs in segments.items():
    segmentation_settings += [
        f'Rules for category "{category}"',
        f'    Milliseconds Before: {opt.left_padding * 1000}',
        f'    Milliseconds After: {opt.right_padding * 1000}',
        '    Milliseconds Offset: 0',
        '    Event 1:',
        f'        Code is "{category}"'
    ]
    segmentation_results += [
        f'Results for category "{category}"',
        f'    {len(segs)} segment(s) created',
    ]
    if len(out_of_bounds_segs[category]) > 0:
        segmentation_results.append(
            f'    {len(out_of_bounds_segs[category])} segment(s) could'
            'not be created because they extended beyond data range'
        )
segmentation_entry = dict(
    method='Segmentation',
    settings=segmentation_settings,
    results=segmentation_results
)
if opt.verbose:
    print(segmentation_entry)
history.append(segmentation_entry)

# Drop bad segments
if opt.artifact_detection is not None:
    clean_segments = {}
    for label, segs in segments.items():
        clean_segments[label] = [
            seg for seg in segs
            if len(detect_bad_channels(seg, opt.artifact_detection)) == 0
        ]
    artifact_results = []
    for label, segs in clean_segments.items():
        if len(segs) == 0:
            raise ValueError('All segments were dropped for event type '
                             f'"{label}" with {opt.artifact_detection} μV '
                             'peak-to-peak amplitude criterion')
        artifact_results += [
            f'Results for category "{label}"',
            f'    {len(segments[label]) - len(segs)} out of '
            f'{len(segments[label])} segments dropped'
        ]
    segments = clean_segments

    artifact_entry = dict(
        method='Artifact Detection',
        settings=[
            f'Bad Channel Threshold: Max - Min > {opt.artifact_detection} μV',
            'Mark segment bad if it contains any bad channels'
        ],
        results=artifact_results
    )
    if opt.verbose:
        print(artifact_entry)
    history.append(artifact_entry)

# Get bad channels from raw MFF
with raw.directory.filepointer('info1') as fp:
    data_info = XML.from_file(fp)
if data_info.generalInformation['channel_type'] != 'EEG':
    raise ValueError('Expected channel type for "info1.xml" is "EEG". Instead '
                     f'got: "{data_info.generalInformation["channel_type"]}"')
channel_info = data_info.find('channels')
if channel_info is not None and \
        channel_info.attrib['exclusion'] == 'badChannels':
    if channel_info.text:
        bad_channels = [int(ch) for ch in channel_info.text.split(' ')]
    else:
        bad_channels = []
else:
    bad_channels = []

# Create averages for each label
averages = Averages(center=int(opt.left_padding * sampling_rate),
                    sr=sampling_rate, bads=bad_channels)
for label, data in segments.items():
    averages.add(label, data)

average_results = [
    f'{nsegs} segments averaged for category "{category}"'
    for category, nsegs in averages.num_segments.items()
]
average_entry = dict(
    method='Averaging',
    settings=['Handle source files separately',
              'Subjects are not averaged together'],
    results=average_results
)
if opt.verbose:
    print(average_entry)
history.append(average_entry)

# Set average reference
if opt.average_ref:
    averages.set_average_reference()

    reference_entry = dict(
        method='Montage Operations Tool',
        settings=['Average Reference']
    )
    if opt.verbose:
        print(reference_entry)
    history.append(reference_entry)

# Write out the averaged data
startdatetime = raw.startdatetime
with raw.directory.filepointer('sensorLayout') as fp:
    sensor_layout = XML.from_file(fp)
device = sensor_layout.name
print(f'\nWriting averaged data to {opt.output_file} ...')
averages.write_to_mff(opt.output_file, startdatetime=startdatetime,
                      device=device, history=history)
