import io
import subprocess
from collections import OrderedDict
from itertools import product

def main():
    params = OrderedDict({
        'lambda' : [0, .1, .5, .9, 1],
        'sample-count' : [10, 100, 1000]
    })

    for comb in product(*params.values()):
        print(comb)
        lmb, count = comb
        run_name = f"model_s_l_{lmb}_c_{count}"
        pretext_weight = lmb
        task_weights = (1 - lmb) / 3

        cmd = f"""
python3.11 -m training -d data/robomaster_ds_training.h5 -t pose_and_led \
-n {run_name} --experiment-id 0 \
--device cuda:3 -a \
--epochs 100 --learning-rate 0.001 \
--w-proj {task_weights} --w-dist {task_weights} --w-ori {task_weights} --w-led {pretext_weight} \
--labeled-count {count} --labeled-count-seed 0 --model-type model_s"""
        
        subprocess.run(cmd.strip().split(' '))

        testing_cmd = f"""
python3.11 -m testing --checkpoint-id 99 --run-name {run_name} \
--experiment-id 0 --task pose_and_led --visible \
--dataset data/robomaster_ds_validation.h5 --device cuda:3 \
        """
        subprocess.run(testing_cmd.strip().split(' '))


if __name__ == "__main__":
    main()





    
