import mlflow
import numpy as np
from src.config.argument_parser import parse_args
from src.dataset.dataset import get_dataset
import torch
from torch.utils.data import DataLoader
import tqdm
import pandas as pd
from src.metrics import angle_difference, mse

from src.models import load_model_mlflow, load_model_raw
from src.viz.plots import orientation_error_distribution, theta_scatter_plot, proj_scatter_plot, proj_error_distribution, custom_scatter, orientation_error_by_orientation
from matplotlib import pyplot as plt


def main():
    args = parse_args('vis', 'inference')
    ds = get_dataset(args.dataset, camera_robot=args.robot_id, target_robots=[args.target_robot_id],
                     augmentations=args.augmentations, only_visible_robots=args.visible,
                     sample_count=args.sample_count, sample_count_seed=args.sample_count_seed)
    dataloader = DataLoader(ds, batch_size = 1, shuffle = False)

    using_mlflow = False

    if args.checkpoint_id:
        model, run_id = load_model_mlflow(experiment_id=args.experiment_id, mlflow_run_name=args.run_name, checkpoint_idx=args.checkpoint_id,
                        model_task=args.task)
        using_mlflow = True
    else:
        model = load_model_raw(args.checkpoint_path, model_task=args.task)

    model = model.to(args.device)
    model.eval()

    
    data = {
        'proj_true': [],
        'proj_pred' : [],
        'dist_true': [],
        'dist_pred' : [],
        'theta_true': [],
        'theta_pred' : [],
        'cos_pred' : [],
        'sin_pred' : [],
    }

    for batch in tqdm.tqdm(dataloader):
        image = batch['image'].to(args.device)
        outs = model(image)
        proj_pred = model.predict_pos_from_out(image, outs)
        dist_pred = model.predict_dist_from_outs(outs)
        theta_pred, cos_pred, sin_pred = model.predict_orientation_from_outs(outs, return_cos_sin = True)
        
        data['proj_pred'].extend(proj_pred)
        data['dist_pred'].extend(dist_pred)
        data['theta_pred'].extend(theta_pred)
        data['cos_pred'].extend(cos_pred)
        data['sin_pred'].extend(sin_pred)

        data['proj_true'].extend(batch['proj_uvz'][:, :2].numpy())
        data['dist_true'].extend(batch['distance_rel'].numpy())
        data['theta_true'].extend(batch['pose_rel'][:, -1].numpy())
    
    
    for k, v in data.items():
        data[k] = np.stack(v)

    data['cos_true'] = np.cos(data['theta_true'])
    data['sin_true'] = np.sin(data['theta_true'])
    data['theta_error'] = angle_difference(data["theta_true"], data["theta_pred"])

    # ds = pd.DataFrame(data)
    mean_dist_error = np.abs(data["dist_true"] - data["dist_pred"]).mean()
    mean_angle_error = np.mean(data['theta_error'])
    mean_proj_erorr = np.median(np.linalg.norm(data["proj_true"] - data["proj_pred"], axis = 1))
    print(f"Median proj error: {mean_proj_erorr}")
    print(f"Mean distance error: {mean_dist_error}")
    print(f"Mean angle error (rads): {mean_angle_error}")
    print(f"Mean angle error (degs): {np.rad2deg(mean_angle_error)}")


    figures = [
        theta_scatter_plot,
        proj_scatter_plot,
        proj_error_distribution,
        orientation_error_distribution,
        custom_scatter('cos_true', 'cos_pred', 'Cos scatter', xlim = [-1, 1], ylim=[-1,1]),
        custom_scatter('sin_true', 'sin_pred', 'Sin scatter', xlim = [-1, 1], ylim=[-1,1]),
        orientation_error_by_orientation,
        custom_scatter('theta_error', 'dist_true', 'Theta error vs Distance', xlabel = "Theta Error [rad]", ylabel = "Distance [m]", correlation=True)
    ]

    if using_mlflow:
        with mlflow.start_run(run_id=run_id) as run:
            for fig_fn in figures:
                fig = fig_fn(data)
                mlflow.log_figure(fig, fig_fn.__name__)
    else:
        for fig_fn in figures:
            fig = fig_fn(data)
            fig.show()
        plt.show()
            



if __name__ == "__main__":
    main()