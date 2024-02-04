import argparse
from pathlib import Path
from datetime import datetime
from numpy import Inf

def parse_args(*config):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='filename of the dataset',
                        default='data/robomaster_ds.h5')
    
    parser.add_argument('--device', type=str, help='filename of the dataset',
                        default='cuda')
    parser.add_argument("-a", "--augmentations", action='store_true')

    parser.add_argument("-c", "--sample-count", type=int, default=None)
    parser.add_argument("-cseed", "--sample-count-seed", type=int, default=None)
    parser.add_argument("--visible", action='store_true')
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("-t", "--task", type=str)
    parser.add_argument("--dist-range", type=float, nargs="+", default = [-Inf, Inf])

    if 'inference' in config:
        group = parser.add_mutually_exclusive_group()

        mlflow_group = group.add_argument_group()
        mlflow_group.add_argument("--checkpoint-id", type=str)
        mlflow_group.add_argument("--run-name", type=str, default=datetime.today().isoformat())
        mlflow_group.add_argument("--run-id", type=str, default=None)
        group.add_argument("--checkpoint-path", type=str)

        parser.add_argument("--inference-dump", default=None, type=Path)
        parser.add_argument("--led-inference", type=str, choices=["gt", "pred", "hybrid", "amax"])


    if 'vis' in config:
        parser.add_argument("--fps", default=3, type=float)
        parser.add_argument("--save", default = None, type= Path)
        parser.add_argument("-r", "--robot-id", type=str, default="RM1")
        parser.add_argument("-tr", "--target-robot-id", type=str, default="RM2")
        parser.add_argument("--receptive-field", default=None, type=float)
        parser.add_argument("--range", default=None, nargs="+", type=int)



    if 'train' in config:
        parser.add_argument("-m", "--model-type", type=str, default='model_s')
        parser.add_argument("-n", "--run-name", type=str, default=datetime.now().isoformat())
        parser.add_argument("-v", "--validation-dataset", type=Path, default=None)
        parser.add_argument("-e", "--epochs", type=int, default=100)
        parser.add_argument("-lr", "--learning-rate", type=float, default=.002)
        parser.add_argument("--dry-run", action='store_true')
        parser.add_argument("--w-proj", default=.25, type = float)
        parser.add_argument("--w-dist", default=.25, type = float)
        parser.add_argument("--w-ori", default=.25, type = float)
        parser.add_argument("--w-led", default=.25, type = float)
        parser.add_argument("--labeled-count", default = None, type=int)
        parser.add_argument("--labeled-count-seed", default = 0, type=int)
        parser.add_argument("--led-inference", type=str, choices=["gt", "pred", "hybrid", "amax"])



    if "comparison" in config:
        parser.add_argument("--dump-files", type=Path, nargs="+")
        parser.add_argument("--out-dir", type=Path, required=True)




    args = parser.parse_args()
    return args