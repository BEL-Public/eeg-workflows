# eeg-workflows
A library of containerized EEG analysis Python scripts.

This repository includes a generalized [ERP script](scripts/erp.py) for
generating ERP averages from a raw MFF marked with events. It also includes
template scripts for more specific EEG analysis pipelines.

## QMS
All the code in this repository is controlled by the standard operating
procedures 0050-SOP Rev. C, 0051-SOP Rev. A, 0052-SOP Rev. A, and
0053-SOP Rev. A.

We keep a [changelog](CHANGELOG.md).

### Code Style
The code adheres to [PEP-0008](https://www.python.org/dev/peps/pep-0008/)
with exceptions detailed in the code.

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
