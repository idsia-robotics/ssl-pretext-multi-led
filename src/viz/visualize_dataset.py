from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import torch
from src.dataset.dataset import get_dataset
from src.config.argument_parser import parse_args
import matplotlib.animation as animation
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.viz import RobotOrientationWidget, LedStatusWidget, ImageWidget, ProjectionGTWidget, PositionGTWidget, DistanceWidget, PoseWidget

def main():
    args = parse_args('vis')
    ds = get_dataset(args.dataset, camera_robot=args.robot_id, target_robots=[args.target_robot_id],
                     augmentations=args.augmentations, only_visible_robots=args.visible,
                     sample_count=args.sample_count, sample_count_seed=args.sample_count_seed,
                     compute_led_visibility=True)
    if args.range is not None:
        start = args.range[0]
        end = args.range[1] if len(args.range) > 1 else len(ds)
        ds = torch.utils.data.Subset(
            ds, torch.arange(start, end))

    dataloader = DataLoader(ds, batch_size = 1, shuffle = False)

    fig = plt.figure(figsize=(1920/150,1080/150), dpi = 150)
    axs = np.array([
        [plt.subplot(121),
        plt.subplot(122)],
    ]).flatten()


    image_widget = PoseWidget(axs[0], title = f"Ground truth pose", mode = 'true')
    position_gt_widget = ProjectionGTWidget(axs[1], title= "Ground truth position map")

    def save_frame(ev):
        if ev.key == 'e':
            name = f"/tmp/{datetime.today().isoformat()}.svg"
            fig.savefig(name, format="svg")
    fig.canvas.mpl_connect('key_press_event', save_frame)


    widgets = [image_widget, position_gt_widget,]

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