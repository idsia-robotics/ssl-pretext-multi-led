python3.11 -m training -d data/robomaster_ds_full_on_training.h5 -t pose_and_led \
-n visible_robot --experiment-id 0 \
--device cuda:0 -a -v data/robomaster_ds_full_on_validation.h5 \
--epochs 100 --learning-rate 0.001 \
--w-proj 0.0 --w-dist 0.0 --w-ori 0.0 --w-led 1. --visible \
--labeled-count 0 --labeled-count-seed 0 --model-type model_s

python3.11 -m testing --checkpoint-id 99 --run-name visible_robot \
--experiment-id 0 --task pose_and_led --visible \
--dataset data/robomaster_ds_full_on_validation.h5 --device cuda:0 --inference-dump out/visible_robot.pkl

python3.11 -m training -d data/robomaster_ds_full_on_training.h5 -t pose_and_led \
-n non_visible_robot --experiment-id 0 \
--device cuda:0 -a -v data/robomaster_ds_full_on_validation.h5 \
--epochs 100 --learning-rate 0.001 \
--w-proj 0.0 --w-dist 0.0 --w-ori 0.0 --w-led 1. \
--labeled-count 0 --labeled-count-seed 0 --model-type model_s

python3.11 -m testing --checkpoint-id 99 --run-name non_visible_robot \
--experiment-id 0 --task pose_and_led --visible \
--dataset data/robomaster_ds_full_on_validation.h5 --device cuda:0 --inference-dump out/non_visible_robot.pkl