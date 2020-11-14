#!/usr/bin/env python
"""Average a raw .mff or .edf file according to specified event markers.

The input file is read and searched for all provided labels.  Segments for all
event markers of specified labels are extracted together with padding intervals
`left-padding` and `right-padding` are extracted from the input file.
Averaging is performed per label.  Then, the data are re-referenced to an
average reference if `average-ref` flag is present.  These are then written to
the output file path as an .mff."""

from os.path import splitext, isdir, isfile, exists, join
from functools import partial

from mne import set_eeg_reference, find_events, events_from_annotations, Epochs
from mne.io import read_raw_egi, read_raw_edf
from mffpy import XML

from eegwlib import evokeds_to_writer

from argparse import ArgumentParser

assert __name__ == "__main__"


def MffType(filepath, should_exist=True):
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


def EEGType(filepath):
    """check that filepath is either an .mff or .edf

    Return (filepath, file_format), either '.mff' or '.edf'."""
    filepath = str(filepath)
    base, ext = splitext(filepath)
    file_format = ext.lower()
    if file_format == '.mff':
        return MffType(filepath), file_format
    elif file_format == '.edf':
        assert exists(filepath), f"File not found: '{filepath}'"
        assert isfile(filepath), f"Not a file: '{filepath}'"
        return filepath, file_format
    else:
        raise ValueError(f"Unknown file type: '{filepath}'")


def LabelStr(labels):
    """Check that labels are valid and return as list"""
    labels = str(labels)
    labels = labels.split(',')
    labels = [label.strip() for label in labels if label.strip()]
    assert len(labels) > 0, "No labels specified"
    return labels


def Padding(f):
    """Check that float is positive"""
    f = float(f)
    assert f >= 0.0, f"Negative padding: {f}"
    return f


devices = [
    'Amp Sample',
    'Geodesic Sensor Net 128 2.1',
    'Geodesic Sensor Net 256 2.1',
    'Geodesic Sensor Net 64 2.0',
    'HydroCel GSN 128 1.0',
    'HydroCel GSN 256 1.0',
    'HydroCel GSN 32 1.0',
    'HydroCel GSN 64 1.0',
    'MicroCel GSN 100 128 1.0',
    'MicroCel GSN 100 256 1.0',
    'MicroCel GSN 100 32 1.0',
    'MicroCel GSN 100 64 1.0',
]

parser = ArgumentParser(description=__doc__)
parser.add_argument('input_file', type=EEGType,
                    help='Path to input file (either .mff or .edf)')
parser.add_argument('output_file', type=partial(MffType, should_exist=False),
                    help='Path to output .mff file')
parser.add_argument('--labels', '-l', type=LabelStr,
                    required=True, help='Comma-separated list of event labels')
parser.add_argument('--left-padding', type=Padding, default=1.0,
                    help='Padding prior to event in sec. (default=1.0)')
parser.add_argument('--right-padding', type=Padding, default=1.0,
                    help='Padding after event in sec. (default=1.0)')
parser.add_argument('--average-ref', action='store_true',
                    help='Set average reference')
parser.add_argument('--device', type=str, choices=devices,
                    help='This argument is required if input file is an .edf. '
                         'Recording device for input data.')
opt = parser.parse_args()

# Read raw input file
path, file_format = opt.input_file
if file_format == '.mff':
    raw = read_raw_egi(path)
    events = find_events(raw, shortest_event=1)
    event_id = raw.event_id
    if opt.device:
        device = opt.device
    else:
        try:
            sensor_layout = XML.from_file(join(path, 'sensorLayout.xml'))
            device = sensor_layout.name
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Could not find recording device info in {path}. Please '
                f'specify a device from the following options: {devices}.'
            )
elif file_format == '.edf':
    if opt.device:
        device = opt.device
    else:
        raise ValueError(
            f'Input file {path} is an .edf. A recording device must be '
            f'specified from the following options: {devices}.'
        )
    raw = read_raw_edf(path)
    events, event_id = events_from_annotations(raw)
else:
    raise TypeError(f"Unknown file type: '{file_format}'")

# Extract events of specified labels
segmentation_events = {}
for label in opt.labels:
    if label in event_id:
        segmentation_events[label] = event_id[label]
    else:
        raise ValueError(f"Label '{label}' not found among events\n"
                         f"Valid events: {event_id.keys()}")

epochs = Epochs(raw, events, event_id=segmentation_events,
                tmin=-opt.left_padding, tmax=opt.right_padding, baseline=None)

# Average across all events by label and re-reference
averages = []
for label in epochs.event_id.keys():
    average = epochs[label].average()
    if opt.average_ref:
        average, _ = set_eeg_reference(average, ref_channels='average')
    averages.append(average)

# Write result
W = evokeds_to_writer(averages, opt.output_file, device)
W.write()
