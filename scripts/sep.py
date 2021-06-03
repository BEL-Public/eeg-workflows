#!/usr/bin/env python
"""Somatosensory evoked potentials

This script demonstrates a comparison of averaged evoked potentials from
vibratory stimulation of four different fingers.
"""
from os.path import splitext

import mne

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('input_file', type=str, help='Path to the input MFF file.')

# Read the raw MFF
fname = parser.parse_args().input_file
raw = mne.io.read_raw_egi(fname, preload=True)
events = mne.find_events(raw)

# Highpass filter
raw.filter(l_freq=0.1, h_freq=None, picks=['eeg'])

# Create segments for the 4 DIN conditions
din_codes = ['DIN1', 'DIN2', 'DIN3', 'DIN4']
din_ids = {
    code: val for code, val in raw.event_id.items() if code in din_codes
}
segments = mne.Epochs(raw, events, din_ids, tmin=-0.2, tmax=0.5,
                      baseline=None, picks='eeg')
del raw

# Artifact detection
segments.load_data()
segments.drop_bad(reject=dict(eeg=200e-6))

# Equalize event counts for each condition
segments.equalize_event_counts(din_codes)

# Average
averages = [segments[code].average().apply_baseline() for code in din_codes]
del segments

# Plot comparison of averaged data
figure = mne.viz.plot_compare_evokeds(averages, show_sensors='upper right',
                                      show=False)
output = splitext(fname)[0] + '_compare_ave.png'
figure[0].savefig(output)
