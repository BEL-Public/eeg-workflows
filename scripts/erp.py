#!/usr/bin/env python
"""Average and .mff file according to specified event markers"""

from os.path import splitext, isdir, exists
from functools import partial
from mne import set_eeg_reference, find_events, Epochs, write_evokeds
from mne.io import read_raw_egi

from argparse import ArgumentParser

assert __name__ == "__main__"

def MffType(filepath, exist_ok=True):
    """check that filepath is an .mff"""
    filepath = str(filepath)
    assert exist_ok or not exists(filepath), f"File exists: '{filepath}'"
    assert isdir(filepath), f"Not a folder: '{filepath}'"
    base, ext = splitext(filepath)
    assert ext.lower() == '.mff', f"Unknown file type: '{filepath}'"
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
parser.add_argument('output_file', type=partial(MffType, exist_ok=False), help='Path to output .mff file')
parser.add_argument('--labels', '-l', type=LabelStr, required=True, help='Comma-separated list of event labels')
parser.add_argument('--tminus', type=Padding, default=1.0, help='Padding prior to event in sec. (default=1.0)')
parser.add_argument('--tplus', type=Padding, default=1.0, help='Padding after event in sec. (default=1.0)')
opt = parser.parse_args()

mff = read_raw_egi(opt.input_file)
events = find_events(mff, shortest_event=1)
event_id = {
    label: mff.event_id[label]
    for label in opt.labels
    if label in mff.event_id
}
epochs = Epochs(mff, events, event_id=event_id,
                tmin=-opt.tminus, tmax=opt.tplus, baseline=None)

averages = []
for label in epochs.event_id.keys():
    average = epochs[label].average()
    average, _ = set_eeg_reference(average, ref_channels='average')
    averages.append(average)

write_evokeds(opt.output_file, averages)
