import io
import subprocess
from collections import OrderedDict
from itertools import product

def main():

    device = "cuda:0"
    train_ds = "real_four_ds_training.h5"
    val_ds = "real_four_ds_validation.h5"


    params = OrderedDict({
        "model" : ['cam'],
        'replicas' : ["1", "2", "3"],
    })

    for comb in product(*params.values()):
        model, rpl = comb
        run_name = f"gradcam_rpl{rpl}"

        cmd = f"""
python3.11 -m training -d data/{train_ds} -t pose_and_led \
-n {run_name} --experiment-id 0 \
--device {device} -a \
--epochs 100 --learning-rate 0.001 -v data/{val_ds} \
--w-proj .0 --w-dist .0 --w-ori .0 --w-led 1. \
--model-type {model} --visible --led-inference amax"""
        
        subprocess.run(cmd.strip().split(' '))

        testing_cmd = f"""
python3.11 -m testing --checkpoint-id 99 --run-name {run_name} \
--experiment-id 0 --task pose_and_led --visible \
--dataset data/{val_ds} --device {device} --inference-dump out/gradcam/{run_name}.pkl\
        """
        subprocess.run(testing_cmd.strip().split(' '))


if __name__ == "__main__":
    main()





    
