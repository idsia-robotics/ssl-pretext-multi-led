from matplotlib import pyplot as plt
from src.dataset.dataset import get_dataset
from src.config.argument_parser import parse_args
import matplotlib.animation as animation
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import numpy as np
from src.models import load_model_mlflow, load_model_raw


def get_led_indicator_color(led_status):
    return 'green' if led_status else 'white'

def vis(sample, model, image_plot, model_output_plot, pos_pred_plot, pos_map_plot):
    image_plot.set_data(sample['image'].squeeze().cpu().numpy().transpose(1, 2, 0))
    out = model(sample['image']).squeeze().detach().cpu().numpy()

    predicted_pos = model.predict_pos(sample['image'])
    pos_pred_plot.set_offsets(predicted_pos)
    model_output_plot.set_data(out)
    print(out.mean())
    pos_map_plot.set_data(sample['pos_map'].squeeze().cpu().numpy())

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
                     augmentations=args.augmentations, only_visible_robots=args.visible,
                     sample_count=args.sample_count, sample_count_seed=args.sample_count_seed)
    dataloader = DataLoader(ds, batch_size = 1, shuffle = False)

    fig, axs = plt.subplots(1,3)
    
    image_plot = axs[0].imshow(np.zeros((360, 640, 3), dtype=np.uint8))
    pos_map_plot = axs[2].imshow(np.zeros((360, 640, 3), dtype=np.uint8), vmin = 0, vmax = 1, cmap = 'viridis')
    pos_pred_plot = axs[0].scatter(0, 0, marker = 'o', s=1001, facecolor = 'none', edgecolors = 'red')
    axs[0].set_title(f"Camera feed")

    model_output_plot = axs[1].imshow(np.zeros((360, 640, 1), dtype=np.uint8), cmap = 'viridis', vmin = 0, vmax = 1)
    axs[1].set_title(f"Model output")
    
    axs[0].tick_params(bottom = False, left = False, top = False, right = False, labelleft = False, labelbottom = False) 
    axs[1].tick_params(bottom = False, left = False, top = False, right = False, labelleft = False, labelbottom = False) 

    fig.tight_layout()

    if args.checkpoint_id:
        model = load_model_mlflow(experiment_id=args.experiment_id, mlflow_run_name=args.run_name, checkpoint_idx=args.checkpoint_id,
                        model_task=args.task)
    else:
        model = load_model_raw(args.checkpoint_path, model_task=args.task)

    model.eval()

    anim = animation.FuncAnimation(
        fig, vis, dataloader,
        blit=False, fargs=(model, image_plot,model_output_plot, pos_pred_plot, pos_map_plot),
        interval=1000 // args.fps,
        cache_frame_data=False)
    if args.save:
        anim.save(args.save, writer='ffmpeg', fps=args.fps, bitrate=12000, dpi=200)
    else:
        plt.show()

    


if __name__ == "__main__":
    main()