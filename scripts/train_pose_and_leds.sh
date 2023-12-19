python3.11 training.py -d data/robomaster_ds_training.h5 -t pose_and_led\
    -n pose_and_led --experiment-id 0\
    --device $1 -a --visible\
    -v data/robomaster_ds_validation.h5 --epochs 100
