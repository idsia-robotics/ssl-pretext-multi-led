RUN_NAME=multiscale_led_masked

python3.11 -m training -d data/robomaster_ds_training.h5 -t pose_and_led\
    -n ${RUN_NAME} --experiment-id 0\
    --device $1 -a --visible\
    --model-type multiscale_model_s
    -v data/robomaster_ds_validation.h5 --epochs 100 --learning-rate 0.002
python3.11 -m testing_led --checkpoint-id 99 --run-name ${RUN_NAME} --experiment-id 0 --task pose_and_led --dataset data/robomaster_ds_validation.h5 --device $1
