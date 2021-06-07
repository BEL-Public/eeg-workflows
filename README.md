# eeg-workflows
A library of containerized EEG analysis Python scripts.

This repository contains EEG analysis workflows in the form of Python scripts,
which are intended to be containerized as workflows in FLOW, an EEG database
management solution. FLOW makes it easy to add Python scripts accompanied by
a [dockerfile](https://docs.docker.com/engine/reference/builder/) which can
be run as workflows with I/O directly to the FLOW database. We created this
public repository to encourage sharing of EEG analysis scripts, as we hope to
expand the variety of workflows that FLOW has to offer. This repository
includes a generalized [ERP script](scripts/erp.py) for generating ERP averages
from a raw MFF marked with events. It also includes template scripts for more
specific EEG analysis pipelines.

## Installation
Python scripts in this repository can be run and tested locally by following
these install instructions.

Create a local copy of the repository by running the following in the
terminal. Make sure your are in the directory to where you want the copy to
live.
```
$ git clone https://github.com/BEL-Public/eeg-workflows.git
$ cd eeg-workflows
```
We recommend creating an anaconda environment to which to install the
dependencies. If you have not installed anaconda on your machine, follow
[these instructions](https://docs.anaconda.com/anaconda/install) to do so.
Once anaconda is installed, run the following commands in the terminal to
set up the environment.
```
# Set up the conda environment
$ conda create -n eeg-wf python=3.8 pip
$ conda activate eeg-wf
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

## Data
Sample data to be used with each script can be downloaded directly from an
instance of FLOW we have deployed for OHBM BrainHack. The data are organized
into different "experiments", each corresponding to one of the workflow
scripts.

## Contribute
We are happy to receive contributions of existing or new EEG analysis workflow
Python scripts to be containerized in FLOW.

We recommend making your additions to your own fork of this repository and then
submitting a pull request to the `main` branch of the upstream repository.
First, fork the repository to your own github account and create a local clone
of the fork. You can add the upstream repository as a remote with
`git remote add upstream https://github.com/BEL-Public/eeg-workflows.git` in
order to pull in any upstream changes. At this point you should have two
remotes - the fork you created on your own github account (most likely called
`origin`) and the `upstream` repository from `BEL-Public`. Create a new branch
to which you make your additions and push it to your remote fork. Make sure to
run:
```bash
$ pip install -r requirements-dev.txt
$ pre-commit install
```
in your conda environment before committing changes. Finally, submit a pull
request to merge your new branch with the `main` branch of the upstream
repository.

If you are new to Python scripting for EEG analysis, it is a good idea to check
out the existing [scripts](scripts) for some examples of how to construct one.
A powerful library of tools for EEG analysis in Python is
[MNE-Python](https://mne.tools/stable/index.html). MNE-Python has an extensive,
growing suite of functions and methods for cleaning, transforming, and
analyzing EEG and MEG data in its native data structures. It also supports
reading data from several different file formats and functionality to export to
various file formats is slowly being added. MNE-Python is already added as a
dependency to this repository. Add `import mne` to a script to import the
entire namespace. If you add a script which requires additional packages, make
sure to add these to the [requirements.txt](requirements.txt) document.

In FLOW, we use the MFF file format, and support converting files from EDF to
MFF. Raw MFF data can be read into the MNE framework as a `Raw` object with
function `mne.io.read_raw_egi`. Once in the form of a `Raw` object, the data
can be transformed and analyzed with a typical MNE-Python workflow, segmenting
into an `Epochs` object and averaging into an `Evoked` object. MNE-Python
includes a
[tutorial](https://mne.tools/stable/auto_tutorials/intro/10_overview.html)
for a quick overview of some of the functions and methods one might use when
creating an analysis workflow. More in-depth tutorials on specific types of
analyses are also available.

## QMS
All the code in this repository is controlled by the standard operating
procedures 0050-SOP Rev. C, 0051-SOP Rev. A, 0052-SOP Rev. A, and
0053-SOP Rev. A.

We keep a [changelog](CHANGELOG.md).

### Code Style
The code adheres to [PEP-0008](https://www.python.org/dev/peps/pep-0008/)
with exceptions detailed in the code.
