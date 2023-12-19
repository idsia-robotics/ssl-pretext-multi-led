#!/bin/bash
python3.11 -m training -d data/robomaster_ds_train.h5 -t position\
    -n position_test --experiment-id 0\
    --device $1 -a --visible\
    -v data/robomaster_ds_validation.h5
