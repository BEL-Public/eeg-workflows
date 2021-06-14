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
is present.  The averages are then written to the output file path as an .mff.

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
from datetime import datetime
from os.path import splitext, isdir, exists
from functools import partial
from typing import List, Union
from xml.etree.ElementTree import parse

from mffpy import XML
from mffpy.xml_files import EventTrack
import pytz

from eegwlib.artifact_detection import detect_bad_channels
from eegwlib.average import Averages
from eegwlib.segment import seconds_to_samples, Segmenter

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


def TimeZone(tzstr: str) -> pytz.BaseTzInfo:
    """Convert timezone string to timezone"""
    assert tzstr in pytz.all_timezones, f'Unknown timezone "{tzstr}". ' \
                                        f'Options: {pytz.all_timezones}'
    return pytz.timezone(tzstr)


parser = ArgumentParser(description=__doc__)
parser.add_argument('input_file', type=MffType,
                    help='Path to input .mff file')
parser.add_argument('output_file', type=partial(MffType, should_exist=False),
                    help='Path to output .mff file')
parser.add_argument('--labels', '-l', type=LabelStr, required=True,
                    help='Comma-separated list of event labels')
parser.add_argument('--categories', type=LabelStr,
                    help='Comma-separated list of category names associated '
                         'with each event label. If category names are not '
                         'provided, the event labels will be used as '
                         'category names.')
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
parser.add_argument('--timezone', type=TimeZone, default='UTC',
                    help='Timezone specification for start/end time '
                         'of each processing step. Must be one of '
                         '`pytz.all_timezones` (default="UTC").')
parser.add_argument('--verbose', '-v', action='store_true',
                    help='Print settings and results for each step')
opt = parser.parse_args()

# Associate category names with event labels
category_names = opt.categories or opt.labels
if len(opt.labels) != len(category_names):
    raise ValueError(f'Number of event labels {opt.labels} does not '
                     f'equal number of categories {category_names}')
categories = dict(zip(category_names, opt.labels))

# Read input file
segmenter = Segmenter(opt.input_file, opt.left_padding, opt.right_padding,
                      order=opt.filter_order, fmin=opt.highpass,
                      fmax=opt.lowpass)
sampling_rate = segmenter.sampling_rates['EEG']

# Get history info if present
if 'history.xml' in segmenter.directory.listdir():
    with segmenter.directory.filepointer('history') as fp:
        history = XML.from_file(fp).entries
else:
    history = []

# Extract relative times for events of specified labels
all_codes = set()
event_times = defaultdict(list)
for file in segmenter.directory.files_by_type['.xml']:
    with segmenter.directory.filepointer(splitext(file)[0]) as fp:
        xml_root = parse(fp).getroot()
        if not xml_root.tag == '{http://www.egi.com/event_mff}eventTrack':
            continue
        events = EventTrack(xml_root).events
        for event in events:
            all_codes.add(event['code'])
            if event['code'] in opt.labels:
                event_times[event['code']].append((
                    event['beginTime'] - segmenter.startdatetime
                ).total_seconds())

times_by_category = {}
for cat, label in categories.items():
    if label not in event_times:
        raise ValueError(f'Label "{label}" not found among events.\n'
                         f'Valid event labels: {all_codes}')
    times_by_category[cat] = sorted(event_times[label])

# Extract data segments from filtered epochs
segment_start = pytz.utc.localize(datetime.utcnow())
segments, out_of_bounds_segs = segmenter.extract_segments(times_by_category)
segment_end = pytz.utc.localize(datetime.utcnow())

# Check if any categories have no segments
for cat in categories:
    if cat not in segments:
        raise ValueError(f'All segments for category "{cat}" '
                         'extended beyond data range')

# Log filtering
for filt, freq in {'Highpass': opt.highpass, 'Lowpass': opt.lowpass}.items():
    if freq:
        filter_entry = dict(
            name=f'ERP Workflow {filt} Filter',
            kind='Transformation',
            method='Filtering',
            beginTime=segment_start.astimezone(opt.timezone),
            endTime=segment_end.astimezone(opt.timezone),
            sourceFiles=[opt.input_file],
            settings=[f'Filter Setting: {freq} Hz {filt}',
                      'Filter Type: IIR Butterworth',
                      f'Filter Order: {opt.filter_order}']
        )
        if opt.verbose:
            print(filter_entry)
        history.append(filter_entry)

# Log segmentation
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
        f'        Code is "{categories[category]}"'
    ]
    segmentation_results += [
        f'Results for category "{category}"',
        f'    {len(segs)} segment(s) created',
    ]
    if category in out_of_bounds_segs:
        segmentation_results.append(
            f'    {len(out_of_bounds_segs[category])} segment(s) could '
            'not be created because they extended beyond data range'
        )
segmentation_entry = dict(
    name='ERP Workflow Segmentation',
    method='Segmentation',
    beginTime=segment_start.astimezone(opt.timezone),
    endTime=segment_end.astimezone(opt.timezone),
    sourceFiles=[opt.input_file],
    settings=segmentation_settings,
    results=segmentation_results
)
if opt.verbose:
    print(segmentation_entry)
history.append(segmentation_entry)

# Drop bad segments
if opt.artifact_detection is not None:
    artifact_start = pytz.utc.localize(datetime.utcnow())
    clean_segments = defaultdict(list)
    for cat, segs in segments.items():
        for seg in segs:
            if len(detect_bad_channels(seg, opt.artifact_detection)) == 0:
                clean_segments[cat].append(seg)
    artifact_results = []
    for cat, segs in segments.items():
        if cat not in clean_segments:
            raise ValueError('All segments were dropped for category '
                             f'"{cat}" with {opt.artifact_detection} μV '
                             'peak-to-peak amplitude criterion')
        artifact_results += [
            f'Results for category "{cat}"',
            f'    {len(segs) - len(clean_segments[cat])} out of '
            f'{len(segs)} segments dropped'
        ]
    segments = clean_segments
    artifact_end = pytz.utc.localize(datetime.utcnow())

    artifact_entry = dict(
        name='ERP Workflow Artifact Detection',
        method='Artifact Detection',
        beginTime=artifact_start.astimezone(opt.timezone),
        endTime=artifact_end.astimezone(opt.timezone),
        sourceFiles=[opt.input_file],
        settings=[
            f'Bad Channel Threshold: Max - Min > {opt.artifact_detection} μV',
            'Mark segment bad if it contains any bad channels'
        ],
        results=artifact_results
    )
    if opt.verbose:
        print(artifact_entry)
    history.append(artifact_entry)

# Get bad channels from input file
with segmenter.directory.filepointer('info1') as fp:
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
average_start = pytz.utc.localize(datetime.utcnow())
# Snap to nearest sample
center = seconds_to_samples(opt.left_padding, sampling_rate) / sampling_rate
averages = Averages(center, sampling_rate, bads=bad_channels)
for cat, data in segments.items():
    averages.add(cat, data)
average_end = pytz.utc.localize(datetime.utcnow())

average_results = [
    f'{nsegs} segments averaged for category "{category}"'
    for category, nsegs in averages.num_segments.items()
]
average_entry = dict(
    name='ERP Workflow Averaging',
    method='Averaging',
    beginTime=average_start.astimezone(opt.timezone),
    endTime=average_end.astimezone(opt.timezone),
    sourceFiles=[opt.input_file],
    settings=['Handle source files separately',
              'Subjects are not averaged together'],
    results=average_results
)
if opt.verbose:
    print(average_entry)
history.append(average_entry)

# Set average reference
if opt.average_ref:
    average_ref_start = pytz.utc.localize(datetime.utcnow())
    averages.set_average_reference()
    average_ref_end = pytz.utc.localize(datetime.utcnow())

    reference_entry = dict(
        name='ERP Workflow Average Reference',
        method='Montage Operations Tool',
        beginTime=average_ref_start.astimezone(opt.timezone),
        endTime=average_ref_end.astimezone(opt.timezone),
        sourceFiles=[opt.input_file],
        settings=['Average Reference']
    )
    if opt.verbose:
        print(reference_entry)
    history.append(reference_entry)

# Write out the averaged data
startdatetime = segmenter.startdatetime
with segmenter.directory.filepointer('sensorLayout') as fp:
    sensor_layout = XML.from_file(fp)
device = sensor_layout.name
print(f'\nWriting averaged data to {opt.output_file} ...')
averages.write_to_mff(opt.output_file, startdatetime=startdatetime,
                      device=device, history=history)
