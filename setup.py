from distutils.core import setup

setup(
    name='eeg-workflows',
    version='0.1.0-develop',
    packages=['eegwlib'],
    scripts=['scripts/sws-pilot-workflow.py', 'scripts/erp.py'],
)
