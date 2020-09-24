#!/usr/bin/env python
"""Generate Slow Wave averages for topographic
categories from MFF with marked slow waves"""

from mne import set_eeg_reference, find_events, Epochs, write_evokeds
from mne.io import read_raw_egi

from argparse import ArgumentParser

parser = ArgumentParser(description='Generates averages of Slow Wave events \
                                    marked in input .mff file and writes \
                                    them as .fif evoked file.')
parser.add_argument('mff', type=str, help='Path to input .mff file.')
parser.add_argument('out', type=str, help='Path to output .fif file. \
                    Output filename should end with -ave.fif.')
opt = parser.parse_args()

if __name__ == "__main__":
    # Read in raw signals and metadata from .mff
    mff = read_raw_egi(opt.mff)

    # Get events from trigger channels
    events = find_events(mff, shortest_event=1)

    # Map IDs of slow wave events
    sw_labels = ['lplf', 'lpcf', 'lprf', 'lplc', 'lpct',
                 'lprc', 'lplt', 'lprt', 'lpoc']
    sw_label_ids = {}
    for label in mff.event_id.keys():
        if label in sw_labels:
            sw_label_ids[label] = mff.event_id[label]

    # Load segments of data around SW peak events
    segments = Epochs(mff, events, event_id=sw_label_ids,
                      tmin=-1.0, tmax=1.0, baseline=None)

    # Average segments for each SW type
    sw_averages = []
    for label in segments.event_id.keys():
        ave = segments[label].average()
        # Average reference
        ave, ref_data = set_eeg_reference(ave)
        sw_averages.append(ave)

    # Write averaged data out as .fif
    write_evokeds(opt.out, sw_averages)
