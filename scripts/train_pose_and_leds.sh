RUN_NAME=pose_and_led_big_rf_small_orb_actual_2

#python3.11 training.py -d data/robomaster_ds_training.h5 -t pose_and_led\
#    -n pose_and_led_big_rf_small_orb_no_mask --experiment-id 0\
#    --device $1 -a --visible\
#    -v data/robomaster_ds_validation.h5 --epochs 100
python3.11 -m training -d data/robomaster_ds_training.h5 -t pose_and_led\
    -n ${RUN_NAME} --experiment-id 0\
    --device $1 -a --visible\
    -v data/robomaster_ds_validation.h5 --epochs 200
python3.11 -m testing_led --checkpoint-id 199 --run-name ${RUN_NAME} --experiment-id 0 --task pose_and_led --dataset data/robomaster_ds_validation.h5 --device $1
