from matplotlib import pyplot as plt
from src.dataset.dataset import get_dataset
from src.config.argument_parser import parse_args
import matplotlib.animation as animation
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_led_indicator_color(led_status):
    return 'green' if led_status else 'white'

def vis(sample, image_plot, gt_pos_plot, led_front, led_right, led_back, led_left, led_top_left, led_top_right, pose_rel_theta_plot,
        pos_map_plot, pos_scatter_in_map, gt_abs_pose_scatter,  gt_abs_orientation_rm1, gt_abs_orientation_rm2,
        gt_abs_pose_text):
    image_plot.set_data(sample['image'].squeeze().cpu().numpy().transpose(1, 2, 0))
    led_front.set(facecolor = get_led_indicator_color(sample["led_bf"]))
    led_right.set(facecolor = get_led_indicator_color(sample["led_br"]))
    led_back.set(facecolor = get_led_indicator_color(sample["led_bb"]))
    led_left.set(facecolor = get_led_indicator_color(sample["led_bl"]))
    led_top_left.set(facecolor = get_led_indicator_color(sample["led_tl"]))
    led_top_right.set(facecolor = get_led_indicator_color(sample["led_tr"]))

    gt_pos_plot.set_offsets(sample['proj_uvz'][:2])
    pos_scatter_in_map.set_offsets(sample['proj_uvz'][:2])
    pos_map_plot.set_data(sample['pos_map'].squeeze().cpu().numpy())

    
    rm1_pose = sample["RM1_pose"].squeeze()[:3]
    rm2_pose = sample["RM2_pose"].squeeze()[:3]
    gt_positions = np.stack([rm1_pose, rm2_pose], axis = 0) # [[x1, y1], [x2, y2]]
    gt_abs_pose_scatter.set_offsets(gt_positions)
    
    theta_1 = sample["RM1_pose"].squeeze()[-1] # (2,)
    theta_2 = sample["RM2_pose"].squeeze()[-1] # (2,)

    
    theta = sample['pose_rel'].squeeze()[-1]
    pose_rel_theta_plot.set_offsets([[theta, 1.5]])
    gt_abs_pose_text[0].set_position(gt_positions[0])
    gt_abs_pose_text[1].set_position(gt_positions[1])



    direction_draw_length = .4

    end_p1 = np.array([rm1_pose[0] + np.cos(theta_1) * direction_draw_length, rm1_pose[1] + np.sin(theta_1) * direction_draw_length ])
    end_p2 = np.array([rm2_pose[0] + np.cos(theta_2) * direction_draw_length, rm2_pose[1] + np.sin(theta_2) * direction_draw_length ])
    # direction_end_coordinates = gt_positions + offsets
    print(sample["pose_rel"][0])
    gt_abs_orientation_rm1.set_xdata((rm1_pose[0], end_p1[0]))
    gt_abs_orientation_rm1.set_ydata((rm1_pose[1], end_p1[1]))

    gt_abs_orientation_rm2.set_xdata((rm2_pose[0], end_p2[0]))
    gt_abs_orientation_rm2.set_ydata((rm2_pose[1], end_p2[1]))
    



def main():
    args = parse_args('vis')
    ds = get_dataset(args.dataset, camera_robot=args.robot_id, target_robots=[args.target_robot_id],
                     augmentations=args.augmentations, only_visible_robots=args.visible,
                     sample_count=args.sample_count, sample_count_seed=args.sample_count_seed)
    dataloader = DataLoader(ds, batch_size = 1, shuffle = False)

    # fig, axs = plt.subplots(2,2, gridspec_kw={"width_ratios": [.7, .3], "height_ratios" : [.6, .4]})
    fig = plt.figure(figsize=(1920 / 150, 1080 / 150), dpi=150)
    axs = []
    axs.append(
        plt.subplot(231)
    )
    axs.append(
        plt.subplot(232)
    )
    axs.append(
        plt.subplot(233, projection = 'polar')
    )
    axs.append(
        plt.subplot(234)
    )
    axs.append(
        plt.subplot(235)
    )

    # axs = axs.flatten()
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
    pos_map_plot = axs[3].imshow(np.ones((360, 640, 1), dtype=np.uint8), cmap = 'viridis', vmin = 0, vmax = 1)

    axs[1].invert_yaxis()
    

    axs[1].add_patch(led_front)
    axs[1].add_patch(led_right)
    axs[1].add_patch(led_left)
    axs[1].add_patch(led_back)
    axs[1].add_patch(led_top_left)
    axs[1].add_patch(led_top_right)

    pose_rel_theta_plot = axs[2].scatter(0, 1.5, s = 250)
    axs[2].set_ylim([-2, 2])
    axs[2].set_title(f"{args.target_robot_id}: Relative orientation w.r.t. camera")

    pos_scatter = axs[0].scatter(0, 0,s=1001,
                          facecolor='none', edgecolors='blue')
    
    pos_scatter_in_map = axs[3].scatter(0, 0,s=1001,
                          facecolor='none', edgecolors='blue')
    



    axs[0].tick_params(bottom = False, left = False, top = False, right = False, labelleft = False, labelbottom = False) 
    axs[1].tick_params(bottom = False, left = False, top = False, right = False, labelleft = False, labelbottom = False) 
    
    # Absolute pose
    axs[4].set_xlim([-2, 2])
    axs[4].set_ylim([-2, 2])
    gt_abs_pose_scatter = axs[4].scatter([0, 1, ], [0, 1], s = 250, color=["red", "blue"])
    #                                   x_l1      x_l2          y_l1      y_l2
    gt_abs_orientation_rm1 = axs[4].plot([], color='black')[0]
    # gt_abs_orientation_rm2 = axs[4].axline((0, 0), (-1, -1), color='black')
    gt_abs_orientation_rm2 = axs[4].plot([], color='black')[0]

    gt_abs_l1 = axs[4].text(0, 0, "RM1")
    gt_abs_l2 = axs[4].text(0, 0, "RM2")



    fig.tight_layout()
    anim = animation.FuncAnimation(
        fig, vis, dataloader,
        blit=False, fargs=(image_plot,pos_scatter, led_front, led_right, led_back, led_left, led_top_left, led_top_right, pose_rel_theta_plot,
                           pos_map_plot, pos_scatter_in_map, gt_abs_pose_scatter, gt_abs_orientation_rm1, gt_abs_orientation_rm2,
                           [gt_abs_l1, gt_abs_l2]),
        interval=1000 // args.fps,
        cache_frame_data=False)
    if args.save:
        anim.save(args.save, writer='ffmpeg', fps=args.fps, bitrate=12000, dpi=150)
    else:
        plt.show()

    


if __name__ == "__main__":
    main()