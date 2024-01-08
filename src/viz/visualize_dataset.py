from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
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
                     sample_count=args.sample_count, sample_count_seed=args.sample_count_seed,
                     compute_led_visibility=True)
    dataloader = DataLoader(ds, batch_size = 1, shuffle = False)

    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios' : [2, 2]}, figsize=(1920 / 150, 1080 / 150), dpi=150)
    

    axs[1].tick_params(labelbottom=False, labelleft=False, left = False, bottom = False)
    axs[1].spines[['right', 'top', 'left', 'bottom']].set_visible(False)\

    inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=axs[1], wspace=0.1, hspace=0.1)
    ax_inner1 = fig.add_subplot(inner_gs[0])
    ax_inner2 = fig.add_subplot(inner_gs[1])

    # distance_widget = DistanceWidget(axs[0], title="Relative distance")
    image_widget = ImageWidget(axs[0], title = f"{args.robot_id}: Camera feed", receptive_field= args.receptive_field)
    led_status_widget = LedStatusWidget(ax_inner1, f"{args.target_robot_id}: Led status")
    # orientation_widget = RobotOrientationWidget(axs[3], title = f"{args.target_robot_id}: Relative orientation w.r.t. camera")
    # proj_gt_widget = ProjectionGTWidget(axs[4], receptive_field = args.receptive_field)
    position_gt_widget = PositionGTWidget(ax_inner2)



    widgets = [led_status_widget, image_widget, position_gt_widget]

    # fig.tight_layout()
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