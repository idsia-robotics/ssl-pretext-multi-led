import io
import subprocess
from collections import OrderedDict
from itertools import product

def main():

    device = "cuda:0"

    params = OrderedDict({
        'sample-count' : [-1, 10, 100, 1000, None],
        'train_dataset' : ["real_four_ds_training.h5"],
        'val_dataset' : ["real_four_ds_validation.h5"],
        "model" : ['model_s_wide'],
        'replicas' : ["1", "2", "3", "4"],
        "use_pre_trained" : [True, False]
    })

    for comb in product(*params.values()):
        count, train_ds, val_ds, model, rpl, using_pre_trained = comb

        pre_trained_name = "pretext_real_four_leds_attention_pretext"
        led_inference = "amax"
        tasks_w = 0.3
        led_w = 0

        if int(rpl) > 1 and int(rpl) < 4:
            pre_trained_name = pre_trained_name + f"_rpl{rpl}"
        elif int(rpl) == 4:
            pre_trained_name = "pretext_None_rpl4"

        if not count == -1:
            continue
        if count == None:
            if using_pre_trained:
                continue
            run_name = f"upperbound_{rpl}"
            task = "tuning"
        elif count == -1:
            if using_pre_trained:
                continue
            approach_name = "pretext"
            task = "pretext"
            led_inference = "pred"
            count = None
            led_w = 1.
            tasks_w = 0.
#            run_name = f"{approach_name}_{count}_rpl{rpl}"
            run_name = pre_trained_name
        else:
            approach_name = "fine_tune" if using_pre_trained else "baseline"
            task = "tuning"
            run_name = f"{approach_name}_{count}_rpl{rpl}"

        print(run_name)

        cmd = f"""
python3.11 -m training -d data/{train_ds} -t {task} \
-n {run_name} --experiment-id 0 \
--device {device} -a \
--epochs 100 --learning-rate 0.001 -v data/{val_ds} \
--w-proj {tasks_w} --w-dist {tasks_w} --w-ori {tasks_w} --w-led {led_w} \
--model-type {model} --visible --led-inference {led_inference} -bs 50"""
        
        if count is not None:
            cmd += f" -c {count} -cseed 0 --labeled-count 1"

        if using_pre_trained:
            cmd += f" --checkpoint-id 99 --weights-run-name {pre_trained_name}"
#        subprocess.run(cmd.strip().split(' '))
        testing_checkpoint = 99 if "upperbound" not in run_name else 40
        testing_cmd = f"""
python3.11 -m testing --checkpoint-id {testing_checkpoint} --run-name {run_name} \
--experiment-id 0 --task pose_and_led --visible \
--dataset data/{val_ds} --device {device} --inference-dump out/real_four_grid/{run_name}.pkl\
        """
        subprocess.run(testing_cmd.strip().split(' '))


if __name__ == "__main__":
    main()





    
