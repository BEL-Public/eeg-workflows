#!/usr/bin/env python
"""Measure the effect of TES with power spectral density

This script analyzes the percent change in power spectral density (PSD) in the
EEG signals after applying transcranial electrical stimulation (TES). This
analysis script is specifically meant to be run on data from the "Power Nap"
experiment in FLOW. In this experiment, a single participant was stimulated
with low frequency TES targeting ventral-limbic brain regions during a day-time
nap for several sessions. TES was delivered in a variable number of 5 minute
blocks (marked with "CLIP" events) with 1 minute rest blocks in between. The
goal of the study was to determine whether the TES would enhance slow wave
sleep. Relative PSD can serve as an indirect measure of slow wave activity by
indicating whether power in the slow wave and spindle bands was increased.
This script plots the percent change in PSD by frequency for left, right, and
medial frontal scalp regions.

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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('input_file', type=str, help='Path to the input MFF file.')
parser.add_argument('--output_file', '-o', type=str,
                    help='Path to the output plot. Defaults to same path as '
                         'the input file with "_relative_psd.pdf" appended.')
opt = parser.parse_args()

assert __name__ == '__main__'

channel_regions = {
    'Left Frontal': [
        'E241', 'E242', 'E243', 'E244', 'E245', 'E246', 'E247', 'E248', 'E249',
        'E250', 'E252', 'E253', 'E254', 'E46', 'E54', 'E61', 'E67', 'E47',
        'E48', 'E55', 'E56', 'E62'
    ],
    'Right Frontal': [
        'E238', 'E239', 'E240', 'E234', 'E235', 'E236', 'E237', 'E230', 'E231',
        'E232', 'E226', 'E227', 'E225', 'E10', 'E1', 'E219', 'E2', 'E221',
        'E220', 'E222', 'E212', 'E211'
    ],
    'Medial Frontal': [
        'E23', 'E15', 'E6', 'E24', 'E16', 'E7', 'E207', 'E17', 'E8', 'E198',
        'E9', 'E186'
    ],
}

# Read the raw MFF
raw = mne.io.read_raw_egi(opt.input_file, preload=True)
events = mne.find_events(raw, shortest_event=1)

# Bandpass filter
raw.filter(l_freq=0.1, h_freq=50.0, picks=['eeg'])

# Get the relative start time of the stimulation
stim_start = None
for event in events:
    if event[2] == raw.event_id['CLIP']:
        stim_start = event[0] / raw.info['sfreq']
        break
if not stim_start:
    raise ValueError('No "CLIP" events found marking stimulation blocks')

# Calculate periodograms for a 60 second window
# preceding first stimulation block
pre_stim_psd = dict()
for region, channels in channel_regions.items():
    psd, freqs = mne.time_frequency.psd_welch(raw, fmax=40.0,
                                              tmin=stim_start - 62.0,
                                              tmax=stim_start - 2.0,
                                              n_overlap=128, picks=channels)
    pre_stim_psd['freqs'] = freqs
    # Average PSD across channels for each region
    pre_stim_psd[region] = np.mean(psd, axis=0)

# Create epochs for time windows following each stimulation block
post_stim_epochs = mne.Epochs(raw, events, raw.event_id['CLIP'], tmin=320.0,
                              tmax=358.0, baseline=(None, None), preload=True)

# Calculate periodograms for post-stimulation epochs
post_stim_psd = dict()
for region, channels in channel_regions.items():
    psd, freqs = mne.time_frequency.psd_welch(post_stim_epochs, fmax=40.0,
                                              n_overlap=128, picks=channels)
    # Average PSD across epochs and channels for each region
    post_stim_psd[region] = np.mean(psd, axis=(0, 1))

# Plot post-stimulation relative to pre-stimulation PSD
fig = plt.figure()
ax = fig.add_subplot()
regions = list(post_stim_psd.keys())
freqs = pre_stim_psd['freqs']
for region in regions:
    relative_psd = post_stim_psd[region] / pre_stim_psd[region]
    ax.plot(freqs, relative_psd)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Relative PSD')
ax.grid()
ax.legend(regions)
ax.set_title('Percent Change in PSD\nfrom Pre to Post Stimulation')

# Save the figure
output = opt.output_file or splitext(opt.input_file)[0] + '_relative_psd.pdf'
plt.savefig(output)
