RUN_NAME=upperbound_ms

python3.11 -m training -d data/robomaster_ds_training.h5 -t pose_and_led\
    -n ${RUN_NAME} --experiment-id 0\
    --device $1 -a --visible\
    -v data/robomaster_ds_validation.h5 --epochs 100 --learning-rate 0.002\
    --w-proj .25 --w-dist .25 --w-ori .25 --w-led .25 --model-type multiscale_model_s
python3.11 -m testing --checkpoint-id 99 --run-name ${RUN_NAME}\
        --experiment-id 0 --task pose_and_led --visible\
        --dataset data/robomaster_ds_validation.h5 --device $1
