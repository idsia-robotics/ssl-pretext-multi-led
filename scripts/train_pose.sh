#!/bin/bash

python3.11 training.py -d data/robomaster_ds_training.h5 -t pose\
    -n pose_test --experiment-id 0\
    --device $1 -a --visible\
    -v data/robomaster_ds_validation.h5
