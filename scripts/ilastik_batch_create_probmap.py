#!/usr/bin/env python
import glob
import os
from modules import utils

""" Creates Pixel Probability Map from Raw Data.

The workflow runs a given Ilastik Pixel Classifier on an input data set in headless mode.
For more information to run ilastik in headless mode,
see: http://ilastik.org/documentation/pixelclassification/headless.html

Attributes:
    project_file: Ilastik project file with a trained pixel classifier.
    inputdirectory: Directory containing .png image stack
    outputdirectory: Directory in which the probability maps are stored
    ilastik_source_directory: main directory of ilastik installation (in which run_ilastik.sh is located)
"""

base_dir = os.path.dirname(os.getcwd())

project_file = os.path.join(base_dir, 'data/ilastik_projectfile.ilp')
inputdirectory = os.path.join(base_dir, 'data/raw_stack/')
outputdirectory = os.path.join(base_dir, 'data/prob_map/')
ilastik_source_directory = '/mnt/drive2015/home/julia/src/ilastik-1.1.7-Linux/'

if not os.path.exists(outputdirectory):
    os.mkdir(outputdirectory)

input_files = glob.glob(inputdirectory + '*.png')

# Extract Probability Map in headless mode
batch_command = ilastik_source_directory + "/run_ilastik.sh --headless --project="
batch_command += project_file + " "
for input_file in input_files:
    batch_command += input_file + " "
batch_command += " --output_filename_format="
batch_command += outputdirectory + "{nickname}.h5"
os.system(batch_command)

# Write single h5 files to h5 stack
outputfile = outputdirectory + 'stack/'
if not os.path.exists(outputfile):
    os.mkdir(outputfile)
outputfile += "stack.h5"
utils.from_h5_to_h5stack(outputdirectory, outputfile)
