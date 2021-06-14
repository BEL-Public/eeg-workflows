import setuptools

requirements = open('requirements.txt').read().split('\n')

setuptools.setup(
    name='eegwlib',
    version='0.1.3',
    author='Evan Hathaway',
    author_email='evan.hathaway@bel.company',
    description='Helper functions for EEG workflows',
    install_requires=requirements,
    packages=['eegwlib'],
)
