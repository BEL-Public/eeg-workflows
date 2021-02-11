#!/usr/bin/env python
"""Average a raw .mff file according to specified event markers.

The input file is read and searched for all provided labels. An IIR
Butterworth filter is applied to the raw signals if either `highpass`
or `lowpass` are specified by the user.  The order of the filter can be
specified via the `filter-order` argument.  Segments for all
event markers of specified labels with padding intervals `left-padding` and
`right-padding` are extracted from the input file.  If `artifact-detection`
argument is provided, bad segments are dropped based on the specified
peak-to-peak amplitude criteria.  Averaging is performed per label.
Then, the data are re-referenced to an average reference if `average-ref` flag
is present.  These are then written to the output file path as an .mff.
"""

from os.path import splitext, isdir, exists
from functools import partial
from typing import List, Union
from xml.etree.ElementTree import parse

from mffpy import Reader, XML
from mffpy.xml_files import EventTrack

from eegwlib.filter import filtfilt
from eegwlib.segment import slice_block
from eegwlib.average import Average
from eegwlib.write import write_averaged

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
    labels = labels.split(',')
    labels = [label.strip() for label in labels if label.strip()]
    assert len(labels) > 0, "No labels specified"
    return labels


def FloatPositive(f: Union[str, float]):
    """Check that float is not negative"""
    f = float(f)
    assert f >= 0.0, f"Negative value: {f}"
    return f


def Frequency(freq: Union[str, float]):
    """Check that frequency value positive"""
    freq = float(freq)
    assert freq > 0.0, f"Non-positive frequency: {freq}"
    return freq


def FilterOrder(order: Union[str, int]):
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
                    help='Peak-to-peak amplitude criteria for bad segment '
                         'rejection in μV. Bad segments will be dropped only '
                         'if this argument is provided.')
parser.add_argument('--average-ref', action='store_true',
                    help='Set average reference')
opt = parser.parse_args()

# Read raw input file
raw = Reader(opt.input_file)
sampling_rate = raw.sampling_rates['EEG']

# Get raw signal blocks and apply filter if specified
apply_filter = opt.highpass is not None or opt.lowpass is not None
data = [
    {
        't0': epoch.t0,
        't1': epoch.t1,
        'data': filtfilt(
            raw.get_physical_samples_from_epoch(epoch)['EEG'][0],
            order=opt.filter_order,
            sr=sampling_rate,
            fmin=opt.highpass,
            fmax=opt.lowpass
        ) if apply_filter else
        raw.get_physical_samples_from_epoch(epoch)['EEG'][0]
    }
    for epoch in raw.epochs
]

# Extract relative times for events of specified labels
all_codes = []
event_times = {label: [] for label in opt.labels}
for file in raw.directory.files_by_type['.xml']:
    with raw.directory.filepointer(splitext(file)[0]) as fp:
        xml_root = parse(fp).getroot()
        if xml_root.tag == '{http://www.egi.com/event_mff}eventTrack':
            events = EventTrack(xml_root).events
            for event in events:
                if event['code'] not in all_codes:
                    all_codes.append(event['code'])
                if event['code'] in opt.labels:
                    event_times[event['code']].append((
                        event['beginTime'] - raw.startdatetime
                    ).total_seconds())

event_times_sorted = {}
for label, times in event_times.items():
    if len(times) == 0:
        raise ValueError(f'Label "{label}" not found among events.\n'
                         f'Valid event labels: {all_codes}')
    else:
        print(f'{label}: {len(times)} events found')
        event_times_sorted[label] = sorted(times)

# Extract data segments
out_of_bounds_segs = {label: [] for label in event_times_sorted}
segments = {label: [] for label in event_times_sorted}
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
        try:
            segment = slice_block(
                block['data'],
                center=time - block['t0'],
                padl=opt.left_padding,
                padr=opt.right_padding,
                sr=sampling_rate
            )
        except IndexError:
            out_of_bounds_segs[label].append(time)
            continue
        segments[label].append(segment)

for label in segments.keys():
    if len(segments[label]) == 0:
        raise ValueError(f'All segments for event type "{label}" '
                         'extended beyond data range')
    if len(out_of_bounds_segs[label]) > 0:
        print(f'{len(out_of_bounds_segs[label])} segment(s) extended beyond '
              f'data range for event type "{label}"')
    print(f'{label}: {len(segments[label])} segments created')

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
        print(f'Bad channels: {bad_channels}')
    else:
        bad_channels = []
else:
    bad_channels = []

# Create averages for each label
averages = [
    Average(label, segs, center=int(opt.left_padding * sampling_rate),
            sr=sampling_rate, bads=bad_channels)
    for label, segs in segments.items()
]

# Write out the averaged data
startdatetime = raw.startdatetime
with raw.directory.filepointer('sensorLayout') as fp:
    sensor_layout = XML.from_file(fp)
device = sensor_layout.name
write_averaged(averages, opt.output_file, startdatetime, device)
print(f'Writing averaged data to {opt.output_file} ...')
