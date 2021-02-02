#!/usr/bin/env python
"""Average a raw .mff file according to specified event markers.

The input file is read and searched for all provided labels.  Segments for all
event markers of specified labels with padding intervals `left-padding` and
`right-padding` are extracted from the input file.  If `artifact-detection`
argument is provided, bad segments are dropped based on the specified
peak-to-peak amplitude criteria. Averaging is performed per label.
Then, the data are re-referenced to an average reference if `average-ref` flag
is present.  These are then written to the output file path as an .mff.
"""

from os.path import splitext, isdir, exists, join
from functools import partial
from typing import List, Union

from mne import set_eeg_reference, find_events, Epochs
from mne.io import read_raw_egi
from mffpy import XML

from eegwlib import evokeds_to_writer

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
    """Check that float is positive"""
    f = float(f)
    assert f >= 0.0, f"Negative value: {f}"
    return f


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
parser.add_argument('--artifact-detection', type=FloatPositive,
                    help='Peak-to-peak amplitude criteria for bad segment '
                         'rejection in μV. Bad segments will be dropped only '
                         'if this argument is provided.')
parser.add_argument('--average-ref', action='store_true',
                    help='Set average reference')
opt = parser.parse_args()

# Read raw input file
raw = read_raw_egi(opt.input_file)
events = find_events(raw, shortest_event=1)
event_id = raw.event_id
sensor_layout = XML.from_file(join(opt.input_file, 'sensorLayout.xml'))
device = sensor_layout.name

# Get event IDs of specified labels
segmentation_events = {}
for label in opt.labels:
    if label in event_id:
        segmentation_events[label] = event_id[label]
    else:
        raise ValueError(f"Label '{label}' not found among events.\n"
                         f"Valid event labels: {event_id.keys()}")

# Segment according to specified event labels
epochs = Epochs(raw, events, event_id=segmentation_events,
                tmin=-opt.left_padding, tmax=opt.right_padding, baseline=None)

# Drop bad segments
if opt.artifact_detection is not None:
    criteria = opt.artifact_detection / 1e6  # convert to volts
    num_epochs_before_drop = len(epochs.selection)
    epochs = epochs.drop_bad(reject=dict(eeg=criteria), verbose='WARNING')
    if len(epochs.selection) == 0:
        raise RuntimeError('All segments were rejected.')
    print(f'{num_epochs_before_drop - len(epochs.selection)} segments were '
          f'rejected based on {opt.artifact_detection} μV peak-to-peak '
          'amplitude criteria.')

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
