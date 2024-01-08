from matplotlib import pyplot as plt
import numpy as np
from src.viz import ImageWidget


def get_cmap(n, name='rainbow'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

class DistanceInferenceWidget:

    def __init__(self, plt_axis, models, model_names, title = '') -> None:
        self.axis = plt_axis

        self.axis.set_title(title)

        color_ids = np.concatenate([np.arange(17), np.arange(23, 30)])
        np.random.seed(0)
        
        self.model_colors = np.random.choice(color_ids, len(models), replace=False)
        self.model_colors = get_cmap(len(models) + 1)
        self.model_colors = [self.model_colors(i) for i in range(len(models) + 1)]

        self.scatter_plots = [self.axis.scatter(0, 0, marker='o', s=51,
                             facecolor='none', c=[c], label = n) for c,n in zip(self.model_colors[:-1], model_names)]
        
        self.gt_scatter_plot = self.axis.scatter(0, 0, marker='o', s=51,
                             facecolor='none', c = [self.model_colors[-1]], label = "GT")
        

        self.models = models

        self.axis.set_xlim([-0.1, 0.1])
        self.axis.set_ylim([-0.1, 3.1])
        self.axis.set_xticks([])
        # self.axis.legend()

    def update(self, data):
        timestamp = data["timestamp"].item()
        for i, m in enumerate(self.models):
            prediction = m[m["timestamp"] == timestamp].iloc[0]["dist_pred"]

            self.scatter_plots[i].set_offsets([0, prediction])

        gt_dist = self.models[0][self.models[0]["timestamp"] == timestamp].iloc[0]["dist_true"]
        self.gt_scatter_plot.set_offsets([0, gt_dist])

class ImageInferenceWidget(ImageWidget):

    def __init__(self, plt_axis, models, model_names, title) -> None:
        super().__init__(plt_axis, title)
        self.axis = plt_axis

        color_ids = np.concatenate([np.arange(17), np.arange(23, 30)])
        np.random.seed(0)
        
        self.model_colors = np.random.choice(color_ids, len(models), replace=False)
        self.model_colors = get_cmap(len(models) + 1)
        self.model_colors = [self.model_colors(i) for i in range(len(models) + 1)]

        self.scatter_plots = [self.axis.scatter(0, 0, marker='o', s=1001,
                             facecolor='none', color=[c], label = n) for c,n in zip(self.model_colors[:-1], model_names)]
        
        self.gt_scatter_plot = self.axis.scatter(0, 0, marker='o', s=1001,
                             facecolor='none', color = [self.model_colors[-1]], label = "GT")


        self.models = models
        legend = self.axis.legend()
        for lh in legend.legendHandles:
            # breakpoint()
            lh.set_sizes([6.0])
            lh.set_facecolor(lh._edgecolors)


    def update(self, data):
        super().update(data)
        timestamp = data["timestamp"].item()
        for i, m in enumerate(self.models):
            prediction = np.stack(m[m["timestamp"] == timestamp].iloc[0]["proj_pred"])
            self.scatter_plots[i].set_offsets(prediction)

        gt_proj = self.models[0][self.models[0]["timestamp"] == timestamp].iloc[0]["proj_true"]
        self.gt_scatter_plot.set_offsets(gt_proj)

class RobotOrientationInferenceWidget:

    def __init__(self, plt_axis, models, model_names, title = '') -> None:
        self.axis = plt_axis
        self.axis.set_title(title)

        color_ids = np.concatenate([np.arange(17), np.arange(23, 30)])
        np.random.seed(0)
        
        self.model_colors = np.random.choice(color_ids, len(models), replace=False)
        self.model_colors = get_cmap(len(models) + 1)
        self.model_colors = [self.model_colors(i) for i in range(len(models) + 1)]

        self.scatter_plots = [self.axis.scatter(0, 0, marker='o', s=51,
                             facecolor='none', c=[c], label = n) for c,n in zip(self.model_colors[:-1], model_names)]
        
        self.gt_scatter_plot = self.axis.scatter(0, 0, marker='o', s=51,
                             facecolor='none', c = [self.model_colors[-1]], label = "GT")
        

        self.models = models

        self.axis.set_ylim([0, 1.2])

        self.axis.legend()

    def update(self, data):
        timestamp = data["timestamp"].item()
        for i, m in enumerate(self.models):
            prediction = m[m["timestamp"] == timestamp].iloc[0]["theta_pred"]

            self.scatter_plots[i].set_offsets([prediction, 1])

        gt_theta = self.models[0][self.models[0]["timestamp"] == timestamp].iloc[0]["theta_true"]
        self.gt_scatter_plot.set_offsets([gt_theta, 1])


    