# Tracking of Microtubules in Electron Microscopy Data

## Introduction

This projects implements a pipeline for the tracking of microtubules in electron microscopy data of neural tissue.
For more information see the corresponding [publication](http://ieeexplore.ieee.org/document/7493275/?arnumber=7493275).







## Installation

Only installation from source is available:

    $ git clone https://github.com/juliabuhmann/microtubules.git
    $ cd microtubules
    $ git submodule update --recursive


## Dependencies

 * python: scipy, h5py, networkx
 * Gurobi

  On Ubuntu, you can get these packages via

  ```
  sudo apt-get install libboost-all-dev liblapack-dev libfftw3-dev libx11-dev libx11-xcb-dev libxcb1-dev libxrandr-dev libxi-dev freeglut3-dev libglew1.6-dev libpng12-dev libtiff4-dev libhdf5-serial-dev
  ```

  Get the gurobi solver from http://www.gurobi.com. Academic licenses are free.

  * XQuartz (only on OS X, http://xquartz.macosforge.org/)

## Getting Started
The repos comes with a small example dataset. To run the pipeline on this example dataset:

    $ cd microtubules
    $ BASEDIR=$(pwd)
    $ ./run_pipeline.sh

Get help:

    $ python scripts/extract_candidates.py --help
    $ python scripts/run_ILP_model.py --help

Intermediate and final results are output in form of knossos skeletons and can be visualized with http://knossostool.org/. 
