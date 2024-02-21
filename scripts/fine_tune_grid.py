import io
import subprocess
from collections import OrderedDict
from itertools import product

def main():

    device = "cuda:0"

    params = OrderedDict({
        'sample-count' : [10, 100, 1000, None],
        'train_dataset' : ["real_four_ds_training.h5"],
        'val_dataset' : ["real_four_ds_validation.h5"],
        "model" : ['model_s_wide'],
        'replicas' : ["1", "2", "3"],
        "use_pre_trained" : [True, False]
    })

    for comb in product(*params.values()):
        count, train_ds, val_ds, model, rpl, using_pre_trained = comb

        pre_trained_name = "pretext_real_four_leds_attention_pretext"
        if int(rpl) > 1:
            pre_trained_name = pre_trained_name + f"_rpl{rpl}"

        if count == None:
            if int(rpl) == 1:
                continue
            if using_pre_trained:
                continue
            run_name = f"upperbound_{rpl}"
            task = "tuning"
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
--w-proj .3 --w-dist .3 --w-ori .3 --w-led 0 \
--model-type {model} --visible"""
        
        if count is not None:
            cmd += " -c {count} -cseed 0"
        subprocess.run(cmd.strip().split(' '))

        testing_cmd = f"""
python3.11 -m testing --checkpoint-id 99 --run-name {run_name} \
--experiment-id 0 --task pose_and_led --visible \
--dataset data/{val_ds} --device {device} --inference-dump out/real_four_1000/{run_name}.pkl\
        """
        subprocess.run(testing_cmd.strip().split(' '))


if __name__ == "__main__":
    main()





    
