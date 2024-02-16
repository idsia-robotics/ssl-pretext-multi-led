import subprocess

val_ds = ""
train_ds = ""
run_name = ""
epochs = 100
task_weights = .3
pretext_weight = 0.
led = "amax"
size = 64
count = 100
run_id="dc073d899ea74a63b0c564c2b8c04b56"
lr_sch = "cosine"


cmd = f"""
python3.11 -m training -d data/{train_ds} -t pose_and_led \
-n {run_name} --experiment-id 0 \
--device cuda:0 -a \
--epochs {epochs} --learning-rate 0.001 -v data/{val_ds} \
--w-proj {task_weights} --w-dist {task_weights} --w-ori {task_weights} --w-led {pretext_weight} \
--labeled-count-seed 0 --led-inference {led} --lr-schedule {lr_sch} --batch-size {size} --visible"""
subprocess.run(cmd.strip().split(' '))
