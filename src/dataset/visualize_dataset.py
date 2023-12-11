from matplotlib import pyplot as plt
from src.dataset.dataset import get_dataset
from src.config.argument_parser import parse_args
import matplotlib.animation as animation
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import numpy as np


def get_led_indicator_color(led_status):
    return 'green' if led_status else 'white'

def vis(sample, image_plot, gt_pos_plot, led_front, led_right, led_back, led_left, led_top_left, led_top_right, odom_plot):
    image_plot.set_data(sample['image'].squeeze().cpu().numpy().transpose(1, 2, 0))
    led_front.set(facecolor = get_led_indicator_color(sample["led_bf"]))
    led_right.set(facecolor = get_led_indicator_color(sample["led_br"]))
    led_back.set(facecolor = get_led_indicator_color(sample["led_bb"]))
    led_left.set(facecolor = get_led_indicator_color(sample["led_bl"]))
    led_top_left.set(facecolor = get_led_indicator_color(sample["led_tl"]))
    led_top_right.set(facecolor = get_led_indicator_color(sample["led_tr"]))
    print(sample['robot_visible'])

    gt_pos_plot.set_offsets(sample['proj_uvz'][:2])
    # current_data = odom_plot.get_xydata()
    # if current_data[-1, 0] - current_data[0,0] >= 5 * 1e9:
        # current_data = current_data[1:, ...]
    # breakpoint()
    # new_datapoint = np.array([[sample.timestamp, sample.odom[0]]])
    # current_data = np.concatenate([current_data, new_datapoint])
    # odom_plot.set_xdata(current_data[:, 0])
    # odom_plot.set_ydata(current_data[:, 1])
    # breakpoint()
    # odom_plot._axes.set_xlim([current_data[0, 0], current_data[-1, 0]])



def main():
    args = parse_args('vis')
    ds = get_dataset(args.dataset, camera_robot=args.robot_id, target_robots=[args.target_robot_id],
                     augmentations=args.augmentations)
    dataloader = DataLoader(ds, batch_size = 1, shuffle = False)

    fig, axs = plt.subplots(2,2, gridspec_kw={"width_ratios": [.7, .3], "height_ratios" : [.6, .4]})
    axs = axs.flatten()
    image_plot = axs[0].imshow(np.zeros((360, 640, 3), dtype=np.uint8))
    axs[0].set_title(f"{args.robot_id}: Camera feed")

    data_plot = axs[1].imshow(np.ones((300, 300, 3), dtype=np.uint8))
    axs[1].set_title(f"{args.target_robot_id}: Led status")
    led_front = patches.Rectangle((110, 230), 80, 20, linewidth=1, edgecolor = 'black', facecolor = 'white')
    led_right = patches.Rectangle((100, 150), 20, 80, linewidth=1, edgecolor = 'black', facecolor = 'white')
    led_left = patches.Rectangle((180, 150), 20, 80, linewidth=1, edgecolor = 'black', facecolor = 'white')
    led_back = patches.Rectangle((110, 130), 80, 20, linewidth=1, edgecolor = 'black', facecolor = 'white')
    led_top_left = patches.Rectangle((45, 50), 80, 20, linewidth=1, edgecolor = 'black', facecolor = 'white')
    led_top_right = patches.Rectangle((175, 50), 80, 20, linewidth=1, edgecolor = 'black', facecolor = 'white')

    axs[1].invert_yaxis()

    axs[1].add_patch(led_front)
    axs[1].add_patch(led_right)
    axs[1].add_patch(led_left)
    axs[1].add_patch(led_back)
    axs[1].add_patch(led_top_left)
    axs[1].add_patch(led_top_right)

    odom_plot = axs[2].plot(0, 0)[0]
    axs[2].set_ylim([-2, 2])
    axs[2].set_title(f"{args.robot_id}: Odometry X")

    pos_scatter = axs[0].scatter(0, 0,s=1001,
                          facecolor='none', edgecolors='blue')


    axs[0].tick_params(bottom = False, left = False, top = False, right = False, labelleft = False, labelbottom = False) 
    axs[1].tick_params(bottom = False, left = False, top = False, right = False, labelleft = False, labelbottom = False) 

    fig.tight_layout()
    anim = animation.FuncAnimation(
        fig, vis, dataloader,
        blit=False, fargs=(image_plot,pos_scatter, led_front, led_right, led_back, led_left, led_top_left, led_top_right, odom_plot),
        interval=1000 // args.fps,
        cache_frame_data=False)
    if args.save:
        anim.save(args.save, writer='ffmpeg', fps=args.fps, bitrate=12000, dpi=200)
    else:
        plt.show()




if __name__ == "__main__":
    main()