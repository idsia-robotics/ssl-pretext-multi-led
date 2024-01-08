import pandas as pd
from functools import reduce
from matplotlib import pyplot as plt
from src.dataset.dataset import get_dataset
from src.config.argument_parser import parse_args
import matplotlib.animation as animation
from torch.utils.data import DataLoader
from src.viz.comparison_widgets import DistanceInferenceWidget, ImageInferenceWidget, RobotOrientationInferenceWidget


def get_led_indicator_color(led_status):
    return 'green' if led_status else 'white'

def vis(sample, model, image_plot, model_output_plot, pos_pred_plot, pos_map_plot):
    image_plot.set_data(sample['image'].squeeze().cpu().numpy().transpose(1, 2, 0))
    out = model(sample['image']).squeeze().detach().cpu().numpy()

    predicted_pos = model.predict_pos(sample['image'])
    pos_pred_plot.set_offsets(predicted_pos)
    model_output_plot.set_data(out)
    pos_map_plot.set_data(sample['pos_map'].squeeze().cpu().numpy())



def main():
    args = parse_args('vis', 'inference', "comparison")

    ds = get_dataset(args.dataset, camera_robot=args.robot_id, target_robots=[args.target_robot_id],
                     augmentations=args.augmentations, only_visible_robots=args.visible,
                     sample_count=args.sample_count, sample_count_seed=args.sample_count_seed)
    dataloader = DataLoader(ds, batch_size = 1, shuffle = False)


    dfs = []
    names = []
    for file in args.dump_files:
        fname = file.stem
        df = pd.read_pickle(file)
        dfs.append(df)
        names.append(fname)
    
    
    df = reduce(lambda  left,right: (pd.merge(left[0],right[0],on=['timestamp'],
                                            how='outer', suffixes=["_" + left[1], "_" + right[1]]), right[1]), zip(dfs, names))[0]
    
    fig, axs = plt.subplots(1, 3, figsize=(1920 / 150, 1080 / 150), dpi=150, gridspec_kw={"width_ratios" : [.5, .05, .45]})

    axs = axs.flatten()
    axs[-1].remove()
    axs[-1] = fig.add_subplot(1, 3, 3, projection='polar')

    widgets = [
        DistanceInferenceWidget(axs[1], dfs, names, "Relative distance"),
        ImageInferenceWidget(axs[0], dfs, names, "Predicted proj UV"),
        RobotOrientationInferenceWidget(axs[2], dfs, names, "Relative orientation prediction")
    ]

    fig.tight_layout()

    anim = animation.FuncAnimation(
        fig, lambda data: [w.update(data) for w in widgets], dataloader,
        blit=False,
        interval=1000 // args.fps,
        cache_frame_data=False)
    if args.save:
        anim.save(args.save, writer='ffmpeg', fps=args.fps, bitrate=12000, dpi=200)
    else:
        plt.show()

    


if __name__ == "__main__":
    main()