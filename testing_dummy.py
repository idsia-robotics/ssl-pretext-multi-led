#!/bin/python3

from pathlib import Path
from scipy.stats import circmean
import mlflow
import numpy as np
from src.config.argument_parser import parse_args
from src.dataset.dataset import H5Dataset, get_dataset
from torch.utils.data import DataLoader
import tqdm
import pandas as pd
from src.metrics import angle_difference, binary_auc
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

    data = {
        'proj_true': [],
        # 'proj_pred' : [],
        'dist_true': [],
        # 'dist_pred' : [],
        'theta_true': [],
        # 'theta_pred' : [],
        # 'cos_pred' : [],
        # 'sin_pred' : [],
        'pose_rel_true' : [],
        'led_true' : [],
        # 'led_pred' : [],
        'timestamp' : [],
        'led_visibility_mask' : [],
    }

    for batch in tqdm.tqdm(dataloader):
        image = batch['image'].to(args.device)
        

        data['proj_true'].extend(batch['proj_uvz'][:, :2].numpy())
        data['dist_true'].extend(batch['distance_rel'].numpy())
        data['theta_true'].extend(batch['pose_rel'][:, -1].numpy())
        data["pose_rel_true"].extend(batch["pose_rel"])
        data["led_true"].extend(batch["led_mask"])
        data["timestamp"].extend(batch["timestamp"])
        data["led_visibility_mask"].extend(batch["led_visibility_mask"])

    

    for k, v in data.items():
        data[k] = np.stack(v)

    data['cos_true'] = np.cos(data['theta_true'])
    data['sin_true'] = np.sin(data['theta_true'])

    rows = data['dist_true'].shape[0]

    data['proj_pred'] = np.tile(np.mean(data["proj_true"], axis = 0), (rows,1))
    data['dist_pred'] = np.tile(np.mean(data["dist_true"], axis = 0), (rows,))

    data['theta_pred'] = np.tile(circmean(data["theta_true"], axis = 0), (rows,))
    data['cos_pred'] = np.tile(np.mean(data["cos_true"], axis = 0), (rows,))
    data['sin_pred'] = np.tile(np.mean(data["sin_true"], axis = 0), (rows,))
    data["led_pred"] = np.tile(np.mean(data["led_true"], axis = 0), (rows,1))

    data['theta_error'] = angle_difference(data["theta_true"], data["theta_pred"])
    data["proj_error"] = np.linalg.norm(data["proj_true"] - data["proj_pred"], axis = 1)

    # ds = pd.DataFrame(data)
    data["dist_abs_error"] = np.abs(data["dist_true"] - data["dist_pred"])
    mean_dist_error = data["dist_abs_error"].mean()
    mean_angle_error = np.mean(data['theta_error'])
    median_proj_error = np.median(data["proj_error"])

    under_30 = np.linalg.norm(data["proj_true"] - data["proj_pred"], axis = 1) < 30
    precision_30 = under_30.sum() / under_30.shape[0]


    pose_rel_pred = reconstruct_position(data["proj_pred"].T, data["dist_pred"]).T
    position_rel_true = np.concatenate((
        data["pose_rel_true"][:, :-1],
        np.zeros((data["pose_rel_true"].shape[0], 1))),
        axis = 1)
    # breakpoint()
    pose_rel_err = np.linalg.norm(position_rel_true - pose_rel_pred, axis = 1)
    data["pose_rel_err"] = pose_rel_err
    
    pose_rel_err_add = pose_rel_err < .1
    ori_err_add = data["theta_error"] < np.deg2rad(10)
    pose_add = pose_rel_err_add & ori_err_add
    data["pose_add_10_10"] = pose_add

    pose_rel_err_add = pose_rel_err < .2
    ori_err_add = data["theta_error"] < np.deg2rad(20)
    pose_add = pose_rel_err_add & ori_err_add
    data["pose_add_20_20"] = pose_add

    pose_rel_err_add = pose_rel_err < .3
    ori_err_add = data["theta_error"] < np.deg2rad(30)
    pose_add = pose_rel_err_add & ori_err_add
    data["pose_add_30_30"] = pose_add

    print(f"Median proj error: {median_proj_error}")
    print(f"Mean distance error: {mean_dist_error}")
    print(f"Mean angle error (rads): {mean_angle_error}")
    print(f"Mean angle error (degs): {np.rad2deg(mean_angle_error)}")
    print(f"Pose ADD(10cm, 10deg): {data['pose_add_10_10'].mean()}")
    print(f"Pose ADD(20cm, 20deg): {data['pose_add_20_20'].mean()}")
    print(f"Pose ADD(30cm, 30deg): {data['pose_add_30_30'].mean()}")
    print(data["timestamp"].min())

    aucs = []
    for i, led_label in enumerate(H5Dataset.LED_TYPES):
        visible_mask = data["led_visibility_mask"][:, i]
        auc, thr = binary_auc(data["led_pred"][visible_mask, i], data["led_true"][visible_mask, i], return_optimal_threshold=True)
        print(f"AUC for led {led_label}: {auc}\t thr:{thr:.2f}")

    figures = [
        theta_scatter_plot,
        proj_scatter_plot,
        proj_error_distribution,
        orientation_error_distribution,
        custom_scatter('cos_true', 'cos_pred', 'Predicted Cos vs True Cos', xlim = [-1, 1], ylim=[-1,1], plot_name="Cos scatter", xlabel="True [rad]", ylabel = "Pred [rad]"),
        custom_scatter('sin_true', 'sin_pred', 'Predicted Sin vs True Sin', xlim = [-1, 1], ylim=[-1,1], plot_name="Sin scatter", xlabel="True [rad]", ylabel = "Pred [rad]"),
        orientation_error_by_orientation,
        custom_scatter('theta_error', 'dist_true', 'Predicted Orientation Error vs True Distance', xlabel = "Theta Error [rad]", ylabel = "Distance [m]", correlation=True, plot_name="Distance-Theta error scatter"),
        custom_scatter('proj_error', 'dist_true', 'Predicted Image Position Error vs True Distance', xlabel = "Proj error [px]", ylabel = "Distance [m]", correlation=True, plot_name="Distance-Proj error scatter"),
        custom_scatter('dist_pred', 'dist_true', 'Predicted Distance vs True Distance', xlabel = "Predicted Distance [m]", ylabel = "True Distance [m]", correlation=True, plot_name="Distance scatter"),
        distance_error_distribution,
        pose_add_jointplot,
        plot_multi_add
    ]

    run_name = args.run_name
    out_folder = Path("plots/") / run_name
    out_folder.mkdir(exist_ok=True, parents=True)
    for fig_fn in figures:
        fig = fig_fn(data)
        fig.savefig(out_folder / fig_fn.__name__)
            
    if args.inference_dump:
        params = {"w_led" : "1.", "sample_count" : None}
        for k in data.keys():
            data[k] = data[k].tolist()
        df = pd.DataFrame(data)
        if params is not None:
            df.attrs = params
        df.to_pickle(args.inference_dump)



if __name__ == "__main__":
    main()