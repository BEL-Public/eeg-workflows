# eeg-workflows
A catalogue of Python EEG analysis workflows to be implemented in BEL Cloud.

This repository contains template Python scripts for ERP and other EEG
analytic workflows based primarily on the MNE library. These scripts will
be adopted as analysis tools in BEL Cloud, providing users options for
processing EEG data and generating outputs. Eventually, we could transition
this repository to BEL-Public github account where BEL Cloud users could
view our analysis scripts and submit their own custom scripts.

## Installation
```
# Set up the conda environment
$ conda create -n eegw python=3.6 pip
$ conda activate eegw
$ pip install -r requirements.txt
# Install python scripts
$ python setup.py install
# Run the tests
$ make test
```

## Containerization

We containerize all scripts with docker.  `make docker-build` builds the docker
image able to execute any of the scripts.

To execute a script in a container, you need to map a folder with the input
data into the container at "/app/volume".  This folder will be set as root for
the execution of the script.  Here's an example that will print the options of
"scripts/sws-pilot-workflow.py".

```bash
docker run -it \
       -v `pwd`/volume:/app/volume \
       docker.belco.tech/eegworkflow:latest \
       sws-pilot-workflow.py -h
```

## Contribute

Run:
```bash
$ pip install -r requirements-dev.txt
$ pre-commit install
```
