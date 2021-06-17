from distutils.core import setup

setup(
    name='example_workflow',
    description='Copies an input file to an output file. This script is used '
    'purely as demonstration for containerization',
    version='0.1.0',
    scripts=['example.py'],
)
