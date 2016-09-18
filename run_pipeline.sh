#!/bin/bash

# Specify here location to git repos directory in which this shell script resides
# BASEDIR=/PATH/TO/MAINDIRECTORY

# Dataset Parameters
VOXEL_SIZEX=4.6
VOXEL_SIZEY=4.6
VOXEL_SIZEZ=50.0

# IO Parameters
CONFIG_PATH=$BASEDIR/data/ILP_model.conf
DATA_INPUTFILENAME=$BASEDIR/data/candidates/0_50_52Dpoints.nml
OUTPUTDIRECTORY=$BASEDIR/data/

# Candidate Extraction Parameters
THRESHOLD=0.5

# ILP Hyperparameters
COMB_ANGLE_COST=3.0
DISTANCE_COST=1.0
ANGLE_COST_FACTOR=30.0
PRIOR=60


# Extract candidates
python scripts/extract_candidates.py \
    -prob_map_inputfilename $BASEDIR/data/prob_map/stack/stack.h5 \
    -knossos_outputdirectory=$BASEDIR/data/candidates/ \
    -voxel_size $VOXEL_SIZEX $VOXEL_SIZEY $VOXEL_SIZEZ \
    -preprocessing_threshold $THRESHOLD


# Run ILP model
python scripts/run_ILP_model.py @${CONFIG_PATH} -v \
    -outputdirectory $OUTPUTDIRECTORY \
    -comb_angle_cost $COMB_ANGLE_COST \
    -angle_cost_factor $ANGLE_COST_FACTOR \
    -dummy_edge_cost $PRIOR \
    -selection_cost $((-2*PRIOR)) \
    -data_inputfilename $DATA_INPUTFILENAME