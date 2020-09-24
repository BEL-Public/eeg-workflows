# eeg-workflows
A catalogue of Python EEG analysis workflows to be implemented in BEL Cloud.

This repository contains template Python scripts for ERP and other EEG
analytic workflows based primarily on the MNE library. These scripts will
be adopted as analysis tools in BEL Cloud, providing users options for
processing EEG data and generating outputs. Eventually, we could transition
this repository to BEL-Public github account where BEL Cloud users could
view our analysis scripts and submit their own custom scripts.

### Installation
```
# Set up the conda environment
$ conda create -n eegw python=3.6 pip
$ conda activate eegw
$ pip install -r requirements.txt
# Install python scripts
$ python setup.py install
```
