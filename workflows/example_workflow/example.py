#!/usr/bin/env python
from argparse import ArgumentParser
import shutil

parser = ArgumentParser(description=__doc__)
parser.add_argument('--input-file', type=str, required=True)
parser.add_argument('--output-file', type=str, required=True)

opts = parser.parse_args()

shutil.copyfile(opts.input_file, opts.output_file)
