
import numpy as np
from src.config.argument_parser import parse_args
from src.dataset.dataset import get_dataset
import torch
from torch.utils.data import DataLoader
import tqdm
import pandas as pd
from src.metrics import angle_difference, mse

from src.models import load_model_mlflow, load_model_raw


def main():
    args = parse_args('vis', 'inference')
    ds = get_dataset(args.dataset, camera_robot=args.robot_id, target_robots=[args.target_robot_id],
                     augmentations=args.augmentations, only_visible_robots=args.visible,
                     sample_count=args.sample_count, sample_count_seed=args.sample_count_seed)
    dataloader = DataLoader(ds, batch_size = 1, shuffle = False)

    if args.checkpoint_id:
        model = load_model_mlflow(experiment_id=args.experiment_id, mlflow_run_name=args.run_name, checkpoint_idx=args.checkpoint_id,
                        model_task=args.task)
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
    }
    for batch in tqdm.tqdm(dataloader):
        image = batch['image'].to(args.device)
        outs = model(image)
        proj_pred = model.predict_pos_from_out(image, outs)
        dist_pred = model.predict_dist_from_outs(outs)
        theta_pred = model.predict_orientation_from_outs(outs)
        
        data['proj_pred'].extend(proj_pred)
        data['dist_pred'].extend(dist_pred)
        data['theta_pred'].extend(theta_pred)

        data['proj_true'].extend(batch['proj_uvz'][:, :2].numpy())
        data['dist_true'].extend(batch['pose_rel'][:, 0].numpy())
        data['theta_true'].extend(batch['pose_rel'][:, -1].numpy())
    
    ds = pd.DataFrame(data)
    
    mean_dist_error = mse(ds["dist_true"], ds["dist_pred"])
    mean_angle_error = np.mean(angle_difference(ds["theta_true"], ds["theta_pred"]))
    print(f"Mean distance error: {mean_dist_error}")
    print(f"Mean angle error (rads): {mean_angle_error}")
    print(f"Mean angle error (degs): {np.rad2deg(mean_angle_error)}")



if __name__ == "__main__":
    main()