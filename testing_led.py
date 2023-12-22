import mlflow
import numpy as np
from src.config.argument_parser import parse_args
from src.dataset.dataset import H5Dataset, get_dataset
import torch
from torch.utils.data import DataLoader
import tqdm
import pandas as pd
from src.metrics import angle_difference, mse, binary_auc

from src.models import load_model_mlflow, load_model_raw
from src.viz.plots import orientation_error_distribution, theta_scatter_plot, proj_scatter_plot, proj_error_distribution, custom_scatter, orientation_error_by_orientation
from matplotlib import pyplot as plt


def main():
    args = parse_args('vis', 'inference')
    ds = get_dataset(args.dataset, camera_robot=args.robot_id, target_robots=[args.target_robot_id],
                     augmentations=args.augmentations, only_visible_robots=True,
                     sample_count=args.sample_count, sample_count_seed=args.sample_count_seed,
                     compute_led_visibility=True)
    dataloader = DataLoader(ds, batch_size = 64, shuffle = False)


    if args.checkpoint_id:
        model, run_id = load_model_mlflow(experiment_id=args.experiment_id, mlflow_run_name=args.run_name, checkpoint_idx=args.checkpoint_id,
                        model_task=args.task, return_run_id=True)
        using_mlflow = True
    else:
        model = load_model_raw(args.checkpoint_path, model_task=args.task)

    model = model.to(args.device)
    model.eval()

    
    led_preds = []
    led_trues = []
    led_visibility = []

    for batch in tqdm.tqdm(dataloader):
        image = batch['image'].to(args.device)
        # plt.imshow(image[0, ...].numpy().transpose(1, 2, 0))
        # print(batch["led_visibility_mask"][0])
        # plt.show()
        outs = model(image)
        led_preds.extend(model.predict_leds_with_gt_pos(batch, image))
        led_trues.extend(batch['led_mask'])
        led_visibility.extend(batch['led_visibility_mask'])

    led_visibility = np.stack(led_visibility, axis = 0)
    led_trues = np.stack(led_trues, axis = 0)
    led_preds = np.stack(led_preds, axis = 0)
    aucs = []
    for i, led_label in enumerate(H5Dataset.LED_TYPES):
        vis = led_visibility[:, i]
        auc = binary_auc(led_preds[vis, i], led_trues[vis, i])
        print(f"AUC for led {led_label}: {auc}")
        aucs.append(auc)

    
    if using_mlflow:
        with mlflow.start_run(run_id=run_id) as run:
            for i, led_label in enumerate(H5Dataset.LED_TYPES):
                mlflow.log_metric(f"testing/led/{led_label}", aucs[i])

if __name__ == "__main__":
    main()