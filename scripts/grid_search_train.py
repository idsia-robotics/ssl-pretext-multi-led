import io
import subprocess
from collections import OrderedDict
from itertools import product

def main():
    params = OrderedDict({
        'lambda' : [.5,],
        'sample-count' : [1000,],
        'train_dataset' : ["top_bottom_ds_training.h5"],
        'val_dataset' : ["top_bottom_ds_validation.h5"],
        "model" : ['mobile_net']
    })

    for comb in product(*params.values()):
        print(comb)
        lmb, count, train_ds, val_ds, model = comb
        run_name = f"{train_ds}_{model}_sync_l_{lmb}_c_{count}"
        pretext_weight = lmb
        task_weights = (1 - lmb) / 3

        cmd = f"""
python3.11 -m training -d data/{train_ds} -t pose_and_led \
-n {run_name} --experiment-id 0 \
--device cuda:0 -a \
--epochs 100 --learning-rate 0.001 -v data/{val_ds} \
--w-proj {task_weights} --w-dist {task_weights} --w-ori {task_weights} --w-led {pretext_weight} \
--labeled-count {count} --labeled-count-seed 0 --model-type {model}"""
        
        subprocess.run(cmd.strip().split(' '))

        testing_cmd = f"""
python3.11 -m testing --checkpoint-id 99 --run-name {run_name} \
--experiment-id 0 --task pose_and_led --visible \
--dataset data/{val_ds} --device cuda:0 \
        """
        subprocess.run(testing_cmd.strip().split(' '))


if __name__ == "__main__":
    main()





    
