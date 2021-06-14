# eeg-workflows
A library of containerized EEG analysis Python scripts.

This repository contains EEG analysis workflows in the form of Python scripts,
which are intended to be containerized as workflows in FLOW, an EEG database
management solution. FLOW makes it easy to add Python scripts accompanied by
a [dockerfile](https://docs.docker.com/engine/reference/builder/) which can
be run as workflows with I/O directly to the FLOW database. We created this
public repository to encourage sharing of EEG analysis scripts, as we hope to
expand the variety of workflows that FLOW has to offer.

This repository includes a generalized [ERP workflow](workflows/erp/erp.py) for
generating ERP averages from a raw MFF marked with events. The script for this
workflow takes several arguments, making it useful for most ERP-type analyses.
Other [workflows](workflows) in this repository are specific to particular EEG
experiments.

## Installation
Python scripts in this repository can be containerized and run locally by
following these install instructions. It is assumed you have
[git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) installed
and have a decent understanding of working with a local clone of a remote
repository.

Create a local copy of the repository by running the following in the
terminal. Make sure your are in the directory where you want the copy to live.
```bash
$ git clone https://github.com/BEL-Public/eeg-workflows.git
$ cd eeg-workflows
```
We recommend creating an anaconda environment to which to install the
dependencies. If you have not installed anaconda on your machine, follow
[these instructions](https://docs.anaconda.com/anaconda/install) to do so.
Once anaconda is installed, run the following commands in the terminal to
set up the environment.
```bash
$ conda create -n eeg-wf python=3.8 pip
$ conda activate eeg-wf
$ python setup.py install
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
The experiment-specific scripts in this repository each correspond to an
"experiment" in this instance of [FLOW](brainhack2021.bel.company/login) we
have deployed for OHBM BrainHack. Data from these experiments can be downloaded
from FLOW if you wish to containerize and run these scripts locally. Take a
look at the docstring for each script to know which experiment it goes with.
Please be aware that these are large EEG files that consume a lot of storage.

## Contribute
We are happy to receive contributions of existing or new EEG analysis workflow
Python scripts to be containerized in FLOW.

We recommend making your additions to your own fork of this repository and then
submitting a pull request to the `brainhack` branch of the upstream repository.

First, fork the repository to your own github account and create a local clone
of the fork.
```bash
$ git clone https://github.com/<your username here>/eeg-workflows.git
```
Follow the regular [install instructions](#installation) to set up your
anaconda environment, then run
```bash
$ pip install -r requirements-dev.txt
$ pre-commit install
```
Make sure you are on the `brainhack` branch and pull in any changes from the
upstream branch. This will require you to add the upstream repository as a
remote.
```bash
$ git checkout brainhack
$ git remote add upstream https://github.com/BEL-Public/eeg-workflows.git
$ git pull upstream brainhack
```
Then you can create a new branch to which you will commit your additions.
```bash
$ git checkout -b <branch name here>
```
You are now ready to add your workflow. Each workflow is comprised of its own
directory in the main [workflows](workflows) directory with the following
structure. The `requirements.txt` should contain list all packages required
by the workflow script, the `setup.py` script specifies the version and the
script to be containerized, and the `Dockerfile` contains commands to build the
workflow image.
```bash
├── workflow
│   ├── __init__.py
│   ├── workflow.py
│   ├── requirements.txt
│   ├── setup.py
│   ├── Dockerfile
```
We have included an [example script](workflows/example_script), which simply
copies an input file path to an output file path. This provides a very simple
example of what a workflow should contain.

Once you have added and committed your additions, you will want to push your
branch to your remote fork.
```bash
git push --set-upstream origin <branch name here>
```
Finally, submit a pull request to merge your new branch with the `brainhack`
branch of the upstream repository.

## Getting Started with EEG Analysis Python Scripting
If you are new to Python scripting for EEG analysis, it is a good idea to check
out the existing [workflows](workflows) for some examples of how to construct
one. A powerful library of tools for EEG analysis in Python is
[MNE-Python](https://mne.tools/stable/index.html). MNE-Python has an extensive,
growing suite of functions and methods for cleaning, transforming, and
analyzing EEG and MEG data in its native data structures. It also supports
reading data from several different file formats and functionality to export to
various file formats is in the process of being implemented. Check out this
[tutorial](https://mne.tools/stable/auto_tutorials/intro/10_overview.html)
for a quick overview of some of the functions and methods one might use when
creating an analysis workflow. More in-depth tutorials on specific types of
analyses are also available. If you wish to use MNE-Python's functionality in
a script, make sure to add `mne` to your `requirements.txt` and add
`import mne` to a script to import the entire namespace.

In FLOW, we use the MFF file format. If working within the MNE framework, Raw
MFF data can be converted to an `MNE.io.Raw` object with function
`mne.io.read_raw_egi`. Once in the form of a `Raw` object, the data can be
transformed and analyzed with a typical MNE-Python workflow, segmenting into an
`Epochs` object and averaging into an `Evoked` object. An `Evoked` object can
then be exported as an MFF with `mne.export.export_evokeds_mff` (this has not
been released yet, so add `git+https://github.com/mne-tools/mne-python@main` to
your `requirements.txt` to use it).

More robust MFF file I/O can be achieved with
[mffpy](https://github.com/BEL-Public/mffpy), a Python MFF reader/writer
developed by our own team. `mffpy` can be used as a versatile analysis tool, as
demonstrated in this [library](eegwlib) of functions and methods which are
employed in the main [ERP workflow](workflows/erp/erp.py). We encourage you to
use this library as well as `mffpy` for analysis scripting in addition or in
lieu of the MNE-Python suite. To utilize mffpy, add `mffpy` to your
`requirements.txt` and add `import mffpy` to your script. `eegwlib` is
also published as a `pypi` package. Simply add `eegwlib` to your
`requirements.txt` and add `import eegwlib` to your script.

## QMS
All the code in this repository is controlled by the standard operating
procedures 0050-SOP Rev. C, 0051-SOP Rev. A, 0052-SOP Rev. A, and
0053-SOP Rev. A.

We keep a [changelog](CHANGELOG.md).

## Code Style
The code adheres to [PEP-0008](https://www.python.org/dev/peps/pep-0008/)
with exceptions detailed in the code.

## License and Copyright
Copyright 2021 Brain Electrophysiology Laboratory Company LLC

Licensed under the ApacheLicense, Version 2.0(the "License");
you may not use this module except in compliance with the License.
You may obtain a copy of the License at:

http: // www.apache.org / licenses / LICENSE - 2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied.
