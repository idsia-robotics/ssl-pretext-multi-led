from matplotlib import pyplot as plt
from src.dataset.dataset import get_dataset
from src.config.argument_parser import parse_args
import matplotlib.animation as animation
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.viz import RobotOrientationWidget, LedStatusWidget, ImageWidget, ProjectionGTWidget, PositionGTWidget, DistanceWidget

def main():
    args = parse_args('vis')
    ds = get_dataset(args.dataset, camera_robot=args.robot_id, target_robots=[args.target_robot_id],
                     augmentations=args.augmentations, only_visible_robots=args.visible,
                     sample_count=args.sample_count, sample_count_seed=args.sample_count_seed)
    dataloader = DataLoader(ds, batch_size = 1, shuffle = False)

    fig = plt.figure(figsize=(1920 / 150, 1080 / 150), dpi=150)
    axs = []
    axs.append(
        plt.subplot(231)
    )
    axs.append(
        plt.subplot(232)
    )
    axs.append(
        plt.subplot(233)
    )
    axs.append(
        plt.subplot(234, projection = 'polar')
    )
    axs.append(
        plt.subplot(235)
    )
    axs.append(
        plt.subplot(236)
    )

    distance_widget = DistanceWidget(axs[0], title="Relative distance")
    image_widget = ImageWidget(axs[1], title = f"{args.robot_id}: Camera feed")
    led_status_widget = LedStatusWidget(axs[2], f"{args.target_robot_id}: Led status")
    orientation_widget = RobotOrientationWidget(axs[3], title = f"{args.target_robot_id}: Relative orientation w.r.t. camera")
    proj_gt_widget = ProjectionGTWidget(axs[4])
    position_gt_widget = PositionGTWidget(axs[5])



    widgets = [distance_widget, orientation_widget,led_status_widget, image_widget, proj_gt_widget, position_gt_widget]

    fig.tight_layout()
    anim = animation.FuncAnimation(
        fig, lambda data: [w.update(data) for w in widgets], dataloader,
        blit=False,
        interval=1000 // args.fps,
        cache_frame_data=False)
    if args.save:
        anim.save(args.save, writer='ffmpeg', fps=args.fps, bitrate=12000, dpi=150)
    else:
        plt.show()

    


if __name__ == "__main__":
    main()