#!/bin/python3

from pathlib import Path
import mlflow
import numpy as np
from src.config.argument_parser import parse_args
from src.dataset.dataset import H5Dataset, get_dataset
from torch.utils.data import DataLoader
import tqdm
import pandas as pd
from src.metrics import angle_difference, binary_auc
from src.models import load_model_mlflow, load_model_raw
from src.viz.plots import orientation_error_distribution, plot_multi_add, theta_scatter_plot, proj_scatter_plot, proj_error_distribution, custom_scatter, orientation_error_by_orientation, distance_error_distribution, pose_add_jointplot
from matplotlib import pyplot as plt
from src.inference import reconstruct_position

def main():
    args = parse_args('vis', 'inference')
    ds = get_dataset(args.dataset, camera_robot=None,
                     augmentations=args.augmentations, only_visible_robots=args.visible,
                     sample_count=args.sample_count, sample_count_seed=args.sample_count_seed,
                     compute_led_visibility=True,
                     distance_range=args.dist_range)
    dataloader = DataLoader(ds, batch_size = 32, shuffle = False)

    using_mlflow = False
    params = None

    if args.checkpoint_id:
        model, run_id, params = load_model_mlflow(experiment_id=args.experiment_id, mlflow_run_name=args.run_name, checkpoint_idx=args.checkpoint_id,
                        model_kwargs={'task' : args.task, 'led_inference' : args.led_inference, 'cam' : True}, return_run_id=True, return_run_params=True)
        using_mlflow = True
    else:
        model = load_model_raw(args.checkpoint_path, model_kwargs={'task' : args.task, 'led_inference' : args.led_inference, 'cam' : True})

    model = model.to(args.device)
    model.eval()

    data = {
        'proj_true': [],
        'proj_pred' : [],
        'led_true' : [],
        'led_pred' : [],
        'timestamp' : [],
        'led_visibility_mask' : [],
    }

    for batch in tqdm.tqdm(dataloader):
        image = batch['image'].to(args.device)
        outs = model(image)
        proj_pred = model.predict_pos(image)
        led_pred = model.predict_leds(image).detach().cpu().numpy()
        
        data['proj_pred'].extend(proj_pred)
        data["led_pred"].extend(led_pred)

        data['proj_true'].extend(batch['proj_uvz'][:, :2].numpy())
        data["led_true"].extend(batch["led_mask"])
        data["timestamp"].extend(batch["timestamp"])
        data["led_visibility_mask"].extend(batch["led_visibility_mask"])


    for k, v in data.items():
        data[k] = np.stack(v)

    data["proj_error"] = np.linalg.norm(data["proj_true"] - data["proj_pred"], axis = 1)

    under_30 = np.linalg.norm(data["proj_true"] - data["proj_pred"], axis = 1) < 30
    precision_30 = under_30.sum() / under_30.shape[0]


    aucs = []
    for i, led_label in enumerate(H5Dataset.LED_TYPES):
        visible_mask = data["led_visibility_mask"][:, i]
        auc, thr = binary_auc(data["led_pred"][visible_mask, i], data["led_true"][visible_mask, i], return_optimal_threshold=True)
        print(f"AUC for led {led_label}: {auc}\t thr:{thr:.2f}")

    if using_mlflow:
        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_metric("testing/proj/p30", precision_30)
            
            for a, l in zip(aucs, H5Dataset.LED_TYPES):
                mlflow.log_metric(f"testing/led/{l}/auc", a)

    if args.inference_dump:
        for k in data.keys():
            data[k] = data[k].tolist()
        df = pd.DataFrame(data)
        if params is not None:
            df.attrs = params
        df.to_pickle(args.inference_dump)

if __name__ == "__main__":
    main()