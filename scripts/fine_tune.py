# import subprocess

# val_ds = ""
# train_ds = ""
# run_name = "fine_tune"
# epochs = 100
# task_weights = .3
# pretext_weight = 0.
# led = "amax"
# size = 64
# count = 100
# fine_tune_id="four_leds_model_s_wide_l_1.0_c_1"
# lr_sch = "cosine"


# cmd = f"""
# python3.11 -m training -d data/{train_ds} -t pose_and_led \
# -n {run_name} --experiment-id 0 \
# --device cuda:0 -a \
# --epochs {epochs} --learning-rate 0.001 -v data/{val_ds} \
# --w-proj {task_weights} --w-dist {task_weights} --w-ori {task_weights} --w-led {pretext_weight} \
# --labeled-count-seed 0 --led-inference {led} --lr-schedule {lr_sch} --batch-size {size} --visible \
# --checkpoint-id 99 --run-name {fine_tune_id}"""
# subprocess.run(cmd.strip().split(' '))

import subprocess

val_ds = "four_leds_ds_validation.h5"
train_ds = "four_leds_ds_training.h5"
run_name = "baseline"
epochs = 100
task_weights = .3
pretext_weight = 0.
led = "amax"
size = 64
count = 100
fine_tune_id="four_leds_model_s_wide_l_1.0_c_1"
lr_sch = "cosine"


cmd = f"""
python3.11 -m training -d data/{train_ds} -t pose_and_led \
-n pretext_four_leds --experiment-id 0 \
--device cuda:0 -a \
--epochs {epochs} --learning-rate 0.001 -v data/{val_ds} \
--w-proj 0 --w-dist 0 --w-ori 0 --w-led 1 --model model_s_wide \
--labeled-count-seed 0 --labeled-count 1 --led-inference {led} --lr-schedule {lr_sch} --batch-size {size} --visible"""
subprocess.run(cmd.strip().split(' '))

cmd = f"""
python3.11 -m training -d data/{train_ds} -t pose_and_led \
-n fine_tune_four_leds --experiment-id 0 \
--device cuda:0 -a \
--epochs {epochs} --learning-rate 0.001 -v data/{val_ds} \
--w-proj .3 --w-dist .3 --w-ori .3 --w-led 0 \
-c {count} -cseed 0 --led-inference {led} --lr-schedule {lr_sch} --batch-size {size} --visible \
--checkpoint-id 99 --weights-run-name pretext_four_leds"""
subprocess.run(cmd.strip().split(' '))


cmd = f"""
python3.11 -m training -d data/{train_ds} -t pose_and_led \
-n baseline_four_leds --experiment-id 0 \
--device cuda:0 -a \
--epochs {epochs} --learning-rate 0.001 -v data/{val_ds} \
--w-proj .3 --w-dist .3 --w-ori .3 --w-led 0 --model model_s_wide \
-c {count} -cseed 0--led-inference {led} --lr-schedule {lr_sch} --batch-size {size} --visible"""
subprocess.run(cmd.strip().split(' '))


cmd = f"""
python3.11 -m training -d data/{train_ds} -t pose_and_led \
-n baseline_four_leds_long --experiment-id 0 \
--device cuda:0 -a \
--epochs {epochs * 2} --learning-rate 0.001 -v data/{val_ds} \
--w-proj .3 --w-dist .3 --w-ori .3 --w-led 0 --model model_s_wide \
-c {count} -cseed 0--led-inference {led} --lr-schedule {lr_sch} --batch-size {size} --visible"""
subprocess.run(cmd.strip().split(' '))

