#!/usr/bin/env python
"""Average a raw .mff file according to specified event markers.

The input file is read and searched for all provided labels.  Segments for all
event markers of specified labels are extracted together with padding intervals
`left-padding` and `right-padding` are extracted from the .mff file .
Averaging is performed per label.  Then, the data are re-referenced to an
average reference.  These are then written to the output file path."""

from os.path import splitext, isdir, exists
from functools import partial
from warnings import warn
from mne import set_eeg_reference, find_events, Epochs, write_evokeds
from mne.io import read_raw_egi

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


parser = ArgumentParser(description=__doc__)
parser.add_argument('input_file', type=MffType, help='Path to input .mff file')
parser.add_argument('output_file', type=partial(
    MffType, should_exist=False), help='Path to output .mff file')
parser.add_argument('--labels', '-l', type=LabelStr,
                    required=True, help='Comma-separated list of event labels')
parser.add_argument('--left-padding', type=Padding, default=1.0,
                    help='Padding prior to event in sec. (default=1.0)')
parser.add_argument('--right-padding', type=Padding, default=1.0,
                    help='Padding after event in sec. (default=1.0)')
opt = parser.parse_args()

# Read raw .mff file
mff = read_raw_egi(opt.input_file)
events = find_events(mff, shortest_event=1)

# Extract events of specified labels
for label in opt.labels:
    if label not in mff.event_id:
        warn(f"Label '{label}' not found among events\n"
             f"Valid events: {mff.event_id.keys()}")

event_id = {
    label: mff.event_id[label]
    for label in opt.labels
    if label in mff.event_id
}
epochs = Epochs(mff, events, event_id=event_id,
                tmin=-opt.left_padding, tmax=opt.right_padding, baseline=None)

# Average across all events by label and re-reference
averages = []
for label in epochs.event_id.keys():
    average = epochs[label].average()
    average, _ = set_eeg_reference(average, ref_channels='average')
    averages.append(average)

# Write result
write_evokeds(opt.output_file, averages)
