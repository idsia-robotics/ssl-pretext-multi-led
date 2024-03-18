from datetime import datetime
import torch
from matplotlib import pyplot as plt
from src.dataset.dataset import get_dataset
from src.config.argument_parser import parse_args
import matplotlib.animation as animation
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import numpy as np
from src.models import load_model_mlflow, load_model_raw
from src.viz import ImageInferenceWidget, ModelProjOutput, DistanceInferenceWidget, ModelDistanceOuput, RobotOrientationInferenceWidget, PoseWidget


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



def main():
    args = parse_args('vis', 'inference')
    ds = get_dataset(args.dataset, camera_robot=args.robot_id, target_robots=[args.target_robot_id],
                     augmentations=args.augmentations, only_visible_robots=args.visible,
                     sample_count=args.sample_count, sample_count_seed=args.sample_count_seed,
                     distance_range=args.dist_range)
    if args.range is not None:
        start = args.range[0]
        end = args.range[1] if len(args.range) > 1 else len(ds)
        ds = torch.utils.data.Subset(
            ds, torch.arange(start, end))
    ds = torch.utils.data.Subset(ds, torch.arange(0, len(ds), 3))

    dataloader = DataLoader(ds, batch_size = 1, shuffle = False)

    
    fig, axs = plt.subplots(1, 3, figsize=(1920 / 150, 1080 / 150), dpi=150,
                            gridspec_kw={
                                'width_ratios' : [.01, .45, .45],
                                'height_ratios' : [.5],
                                'top' : .7,
                                'bottom' : .3
                            },
                            )
    # axs = []
    # axs.append(
    #     plt.subplot(131)
    # )
    # axs.append(
    #     plt.subplot(132)
    # )
    # axs.append(
    #     plt.subplot(133)
    # )
    # axs.append(
    #     plt.subplot(234, projection = 'polar')
    # )

    widgets = [
        DistanceInferenceWidget(axs[0], "Predicted Distance"),
        PoseWidget(axs[1], "Predicted Pose"),
        ModelProjOutput(axs[2], "Predicted Projection"),
        # RobotOrientationInferenceWidget(axs[3], "Relative orientation prediction")
    ]
    
    bbox = axs[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    print(f"{width}-{height}")

    bbox = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    print(f"{width}-{height}")

    fig.tight_layout()

    if args.checkpoint_id:
        model = load_model_mlflow(experiment_id=args.experiment_id, mlflow_run_name=args.run_name, checkpoint_idx=args.checkpoint_id,
                        model_kwargs={'task' : args.task, 'led_inference' : args.led_inference})
    else:
        model = load_model_raw(args.checkpoint_path, model_kwargs={'task' : args.task, 'led_inference' : args.led_inference})

    model.eval()

    def save_frame(ev):
        if ev.key == 'e':
            name = f"/tmp/{datetime.today().isoformat()}.svg"
            fig.savefig(name, format="svg")
    fig.canvas.mpl_connect('key_press_event', save_frame)

    anim = animation.FuncAnimation(
        fig, lambda data: [w.update(data, model) for w in widgets], dataloader,
        blit=False,
        interval=1000 // args.fps,
        cache_frame_data=False)
    if args.save:
        anim.save(args.save, writer='ffmpeg', fps=args.fps, bitrate=12000, dpi=200)
    else:
        plt.show()

    


if __name__ == "__main__":
    main()