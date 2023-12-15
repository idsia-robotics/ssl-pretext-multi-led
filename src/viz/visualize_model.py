from matplotlib import pyplot as plt
from src.dataset.dataset import get_dataset
from src.config.argument_parser import parse_args
import matplotlib.animation as animation
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import numpy as np
from src.models import load_model_mlflow, load_model_raw
from src.viz import ImageInferenceWidget, ModelProjOutput, DistanceInferenceWidget, ModelDistanceOuput


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
                     sample_count=args.sample_count, sample_count_seed=args.sample_count_seed)
    dataloader = DataLoader(ds, batch_size = 1, shuffle = False)

    fig, axs = plt.subplots(1,4)

    widgets = [
        ImageInferenceWidget(axs[0], "Camera feed"),
        ModelProjOutput(axs[1], "Model output"),
        DistanceInferenceWidget(axs[2]),
        ModelDistanceOuput(axs[3])
    ]
    

    fig.tight_layout()

    if args.checkpoint_id:
        model = load_model_mlflow(experiment_id=args.experiment_id, mlflow_run_name=args.run_name, checkpoint_idx=args.checkpoint_id,
                        model_task=args.task)
    else:
        model = load_model_raw(args.checkpoint_path, model_task=args.task)

    model.eval()

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