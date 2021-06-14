#!/usr/bin/env python
"""Somatosensory evoked potentials

This script averages evoked potentials from vibratory stimulation of
fingertips. This particular analysis goes with the SEP experiment in FLOW. In
this experiment, subjects received vibratory stimulation of four different
fingers on a single hand while EEG was recorded. The events "DIN1", "DIN2",
"DIN3", and "DIN4" correspond to stimulation the four different fingers.
The evoked potentials from stimulation of each finger are averaged and written
to a single MFF file.

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
from os.path import splitext

import mne

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('input_file', type=str, help='Path to the input MFF file.')
parser.add_argument('--output_file', '-o', type=str,
                    help='Path to the output MFF file. Defaults to same '
                         'path as the input file with "_ave" appended.')
opt = parser.parse_args()

assert __name__ == '__main__'

# Read the raw MFF
fname = opt.input_file
raw = mne.io.read_raw_egi(fname, preload=True)
events = mne.find_events(raw, shortest_event=1)

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
output = opt.output_file or splitext(fname)[0] + '_ave.mff'
mne.export.export_evokeds(output, averages, fmt='mff')
