import matplotlib.patches as patches
import numpy as np
import torch
from src.dataset.dataset import H5Dataset

class RobotOrientationWidget:

    def __init__(self, plt_axis, title) -> None:
        self.axis = plt_axis
        self.plot_obj = plt_axis.scatter(0, 1.5, s = 250)
        self.axis.set_title(title)

    
    def update(self, data):
        theta = data["pose_rel"].squeeze()[-1]
        self.plot_obj.set_offsets([theta, 1.5])


class ImageWidget:

    def __init__(self, plt_axis, title,  receptive_field = None) -> None:
        self.axis = plt_axis
        self.plot_obj = plt_axis.imshow(np.zeros((360, 640, 3), dtype=np.uint8))
        self.axis.set_title(title)
        self.axis.tick_params(bottom = False, left = False,
                              top = False, right = False,
                              labelleft = False, labelbottom = False) 
        self.gt_pos_scatter = self.axis.scatter(0, 0, s = 1001, facecolor = 'none', edgecolors = 'blue')

        self.receptive_field = receptive_field
        if receptive_field:
            self.rf_plot = patches.Rectangle((0, 0), receptive_field, receptive_field, linewidth=1, edgecolor = 'blue', facecolor = 'none')   
            self.axis.add_patch(self.rf_plot)

        # self.x_axis_scatters = [self.axis.scatter(0, 0, s = 10, facecolor='red') for _ in range(100)]
    
    def update(self, data):
        self.plot_obj.set_data(data['image'].squeeze().cpu().numpy().transpose(1, 2, 0))
        self.gt_pos_scatter.set_offsets(data['proj_uvz'][:2])
        if data['proj_uvz'][:, -1] < 0:
            self.gt_pos_scatter.set_offsets([-1000, -1000])
        self.axis.set_title(data["timestamp"].item())

        # print(data['proj_x'].shape)
        
        # for i, p in enumerate(self.x_axis_scatters):
        #     p.set_offsets(data['proj_x'][0, i, :-1])

        if self.receptive_field:
            self.rf_plot.set_xy((data['proj_uvz'][0][:2] - self.receptive_field / 2))


class LedStatusWidget:

    def __init__(self, plt_axis, title) -> None:
        self.axis = plt_axis
        self.axis.invert_yaxis()
        self.plot_obj = plt_axis.imshow(np.zeros((300, 300, 3), dtype=np.uint8))
        self.axis.set_title(title)
        self.led_back = patches.Rectangle((110, 230), 80, 20, linewidth=1, edgecolor = 'black', facecolor = 'white')
        self.led_left = patches.Rectangle((100, 150), 20, 80, linewidth=1, edgecolor = 'black', facecolor = 'white')
        self.led_right = patches.Rectangle((180, 150), 20, 80, linewidth=1, edgecolor = 'black', facecolor = 'white')
        self.led_front = patches.Rectangle((110, 130), 80, 20, linewidth=1, edgecolor = 'black', facecolor = 'white')
        self.led_top_left = patches.Rectangle((45, 50), 80, 20, linewidth=1, edgecolor = 'black', facecolor = 'white')
        self.led_top_right = patches.Rectangle((175, 50), 80, 20, linewidth=1, edgecolor = 'black', facecolor = 'white')
        self.front_label = self.axis.text(125, 270, "BACK", color = "white")
        self.front_label = self.axis.text(125, 110, "FRONT", color = "white")
        self.front_label = self.axis.text(220, 175, "RIGHT", color = "white")
        self.front_label = self.axis.text(40, 175, "LEFT", color = "white")

        direction_arrow = patches.Polygon(((130,180), (150,160), (170, 180)), color = 'red')

        self.axis.add_patch(self.led_front)
        self.axis.add_patch(self.led_right)
        self.axis.add_patch(self.led_left)
        self.axis.add_patch(self.led_back)
        self.axis.add_patch(self.led_top_left)
        self.axis.add_patch(self.led_top_right)
        self.axis.add_patch(direction_arrow)
        self.axis.tick_params(bottom = False, left = False,
                              top = False, right = False,
                              labelleft = False, labelbottom = False) 

    def update(self, data):
        # print(H5Dataset.LED_TYPES)
        # print(data["led_visibility_mask"])
        # print("=" * 50)
        led_bb,led_bl,led_br,led_bf,led_tl,led_tr = data["led_mask"].squeeze()

        self.led_front.set(facecolor = self.get_led_indicator_color(led_bf))
        self.led_right.set(facecolor = self.get_led_indicator_color(led_br))
        self.led_back.set(facecolor = self.get_led_indicator_color(led_bb))
        self.led_left.set(facecolor = self.get_led_indicator_color(led_bl))
        self.led_top_left.set(facecolor = self.get_led_indicator_color(led_tl))
        self.led_top_right.set(facecolor = self.get_led_indicator_color(led_tr))
    
    def get_led_indicator_color(self, led_status):
        return 'green' if led_status else 'white'


class ProjectionGTWidget:

    def __init__(self, plt_axis, title = '', receptive_field = None) -> None:
        self.axis = plt_axis
        self.plot_obj = plt_axis.imshow(np.zeros((360, 640, 3), dtype=np.uint8),
                                        vmin = 0, vmax = 1, cmap = 'viridis')
        self.axis.set_title(title)
        orb_size = H5Dataset.POS_ORB_SIZE
        scatter_size = (np.sqrt(2) * orb_size / 2) ** 2 * np.pi / 72
        self.pos_scatter = self.axis.scatter(0, 0, s = scatter_size,
                                             facecolor = 'none', edgecolors = 'blue')
        
        self.receptive_field = receptive_field
        if receptive_field:
            self.rf_plot = patches.Rectangle((0, 0), receptive_field, receptive_field, linewidth=1, edgecolor = 'blue', facecolor = 'none')   
            self.axis.add_patch(self.rf_plot)

    def update(self, data):
        self.plot_obj.set_data(data['pos_map'].squeeze().cpu().numpy())
        self.pos_scatter.set_offsets(data['proj_uvz'][:2])

        if self.receptive_field:
            self.rf_plot.set_xy((data['proj_uvz'][0][:2] - self.receptive_field / 2))

        
class PositionGTWidget:
    def __init__(self, plt_axis, title = '') -> None:
        self.axis = plt_axis
        self.axis.set_title(title)
        self.position_scatter = self.axis.scatter([0, 1, ], [0, 1], s = 250, color=["red", "blue"])
        self.gt_abs_orientation_rm1 = self.axis.plot([], color='black')[0]
        self.gt_abs_orientation_rm2 = self.axis.plot([], color='black')[0]
        self.gt_abs_l1 = self.axis.text(0, 0, "RM1")
        self.gt_abs_l2 = self.axis.text(0, 0, "RM2")
        self.axis.set_xlim([-2, 2])
        self.axis.set_ylim([-2, 2])
        self.axis.set_aspect('equal')
        

    def update(self, data):
        rm1_pose = data["RM1_pose"].squeeze()[:3]
        rm2_pose = data["RM2_pose"].squeeze()[:3]
        gt_positions = np.stack([rm1_pose, rm2_pose], axis = 0) # [[x1, y1], [x2, y2]]
        self.position_scatter.set_offsets(gt_positions)
        
        theta_1 = data["RM1_pose"].squeeze()[-1] # (2,)
        theta_2 = data["RM2_pose"].squeeze()[-1] # (2,)
        self.gt_abs_l1.set_position(gt_positions[0])
        self.gt_abs_l2.set_position(gt_positions[1])
        direction_draw_length = .4

        end_p1 = np.array([rm1_pose[0] + np.cos(theta_1) * direction_draw_length, rm1_pose[1] + np.sin(theta_1) * direction_draw_length ])
        end_p2 = np.array([rm2_pose[0] + np.cos(theta_2) * direction_draw_length, rm2_pose[1] + np.sin(theta_2) * direction_draw_length ])
        self.gt_abs_orientation_rm1.set_xdata((rm1_pose[0], end_p1[0]))
        self.gt_abs_orientation_rm1.set_ydata((rm1_pose[1], end_p1[1]))

        self.gt_abs_orientation_rm2.set_xdata((rm2_pose[0], end_p2[0]))
        self.gt_abs_orientation_rm2.set_ydata((rm2_pose[1], end_p2[1]))


class ImageInferenceWidget(ImageWidget):

    def __init__(self, plt_axis, title, color = 'red') -> None:
        super().__init__(plt_axis, title)
        self.prediction_scatter = self.axis.scatter(0, 0, marker = 'o', s=1001, facecolor = 'none', edgecolors = color)

    def update(self, data, model):
        super().update(data)
        predicted_proj = model.predict_pos(data['image'])
        self.prediction_scatter.set_offsets(predicted_proj)


class ModelProjOutput:
    def __init__(self, plt_axis, title = '') -> None:
        self.axis = plt_axis
        self.axis.set_title(title)
        self.plot_obj = self.axis.imshow(np.zeros((360, 640, 1), dtype=np.uint8), cmap = 'viridis', vmin = 0, vmax = 1)
        self.axis.tick_params(bottom = False, left = False, top = False, right = False, labelleft = False, labelbottom = False) 

    def update(self, data, model):
        out = model(data['image'])[..., :1, :, :].squeeze().detach().cpu().numpy()
        self.plot_obj.set_data(out)


class DistanceWidget:
    
    def __init__(self, plt_axis, title) -> None:
        self.axis = plt_axis
        self.axis.set_title(title)
        # dis_plot_pred = self.axis.scatter(0, 0, marker='+', s=21,
        #                        facecolor='black')
        self.dis_plot_gt = self.axis.scatter(0, 0, marker='o', s=51,
                             facecolor='none', edgecolor='blue')
        self.axis.set_xlim([-0.1, 0.1])
        self.axis.set_ylim([-0.1, 3.1])
        self.axis.set_xticks([])

    def update(self, data):
        self.dis_plot_gt.set_offsets([0, data["distance_rel"].squeeze()])

        
class DistanceInferenceWidget(DistanceWidget):

    def __init__(self, plt_axis, title = '', color = 'red') -> None:
        super().__init__(plt_axis, title)
        self.dis_plot_pred = self.axis.scatter(0, 0, marker='o', s=51,
                             facecolor='none', edgecolor=color)
        
    def update(self, data, model):
        super().update(data)
        dist_pred = model.predict_dist(data['image']).squeeze()
        self.dis_plot_pred.set_offsets([0, dist_pred])

class ModelDistanceOuput:

    def __init__(self, plt_axis, title = '') -> None:
        self.axis = plt_axis
        self.axis.set_title(title)
        self.plot_obj = self.axis.imshow(np.zeros((360, 640, 1), dtype=np.uint8), cmap = 'viridis', vmin = 0, vmax = 1)
        self.axis.tick_params(bottom = False, left = False, top = False, right = False, labelleft = False, labelbottom = False) 

    def update(self, data, model):
        out = model(data['image'])[..., 2:3, :, :].squeeze().detach().cpu().numpy()
        self.plot_obj.set_data(out)


class RobotOrientationInferenceWidget(RobotOrientationWidget):
    def __init__(self, plt_axis, title, color = 'red') -> None:
        super().__init__(plt_axis, title)
        self.pred_plot = plt_axis.scatter(0, 1.5, s = 250, color = color)

    
    def update(self, data, model):
        super().update(data)
        ori = model.predict_orientation(data["image"]).squeeze()
        self.pred_plot.set_offsets([ori, 1.5])


class RobotLedInferenceWidget:
    def __init__(self, plt_axes, title = '') -> None:
        self.axes = plt_axes
        self.image_plot = self.axes[4].imshow(np.zeros((360, 640, 3)))
        led_out_idx = [0, 1, 2, 3, 5, 7]
        self.led_plots = [self.axes[i].imshow(np.zeros((360, 640, 1)), cmap ='viridis', vmin = 0, vmax = 1) for i in led_out_idx]

        self.axes[0].set_title("TOP LEFT")
        self.axes[1].set_title("FRONT")
        self.axes[2].set_title("TOP RIGHT")
        self.axes[3].set_title("LEFT")
        self.axes[5].set_title("RIGHT")
        self.axes[7].set_title("BACK")

        self.thresholds = np.array([0.70342696, 0.852335, 0.7073505, 0.76392925, 0.52581376, 0.66283405])


        orb_size = H5Dataset.POS_ORB_SIZE
        scatter_size = (np.sqrt(2) * orb_size / 2) ** 2 * np.pi / 72

        self.pos_scatters = [self.axes[i].scatter(0,0, facecolor = 'none', color = 'red', s = scatter_size, ) for i in led_out_idx]

        self.led_status_widget = LedStatusWidget(self.axes[-1], title='LED status')
        self.proj_gt_widget = ProjectionGTWidget(self.axes[-3], 'GT pos') 

        # self.axes.set_title(title)
        for ax in self.axes:
            ax.tick_params(bottom = False, left = False, top = False, right = False, labelleft = False, labelbottom = False) 

    def update(self, data, model):
        out = model(data['image'])[..., 4:, :, :].squeeze().detach().cpu().numpy()
        preds = model.predict_leds_with_gt_pos(data, data['image'])[0]

        # out[out < self.thresholds[:, None, None]] = 0
        self.image_plot.set_data(data['image'].squeeze().numpy().transpose(1, 2, 0))
        self.led_plots[0].set_data(out[-2])
        self.led_plots[1].set_data(out[3])
        self.led_plots[2].set_data(out[-1])
        self.led_plots[3].set_data(out[1])
        self.led_plots[4].set_data(out[2])
        self.led_plots[5].set_data(out[0])

        self.led_plots[0].axes.set_title(f"TOP LEFT {preds[-2]:.2f}")
        self.led_plots[1].axes.set_title(f"FRONT {preds[3]:.2f}")
        self.led_plots[2].axes.set_title(f"TOP RIGHT {preds[-1]:.2f}")
        self.led_plots[3].axes.set_title(f"LEFT {preds[1]:.2f}")
        self.led_plots[4].axes.set_title(f"RIGHT {preds[2]:.2f}")
        self.led_plots[5].axes.set_title(f"BACK {preds[0]:.2f}")



        for pos_scatter in self.pos_scatters:
            proj_uv = data['proj_uvz'][:2]
            pos_scatter.set_offsets(proj_uv)
        self.led_status_widget.update(data)
        self.proj_gt_widget.update(data)

