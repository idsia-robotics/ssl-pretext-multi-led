import matplotlib.patches as patches
import numpy as np


class RobotOrientationWidget:

    def __init__(self, plt_axis, title) -> None:
        self.axis = plt_axis
        print(plt_axis.properties())
        # assert plt_axis.properties()['projection'] == "polar"
        self.plot_obj = plt_axis.scatter(0, 1.5, s = 250)
        self.axis.set_title(title)

    
    def update(self, data):
        theta = data["pose_rel"].squeeze()[-1]
        self.plot_obj.set_offsets([theta, 1.5])


class ImageWidget:

    def __init__(self, plt_axis, title) -> None:
        self.axis = plt_axis
        self.plot_obj = plt_axis.imshow(np.zeros((360, 640, 3), dtype=np.uint8))
        self.axis.set_title(title)
        self.axis.tick_params(bottom = False, left = False,
                              top = False, right = False,
                              labelleft = False, labelbottom = False) 
        self.gt_pos_scatter = self.axis.scatter(0, 0, s = 1001, facecolor = 'none', edgecolors = 'blue')
    
    def update(self, data):
        self.plot_obj.set_data(data['image'].squeeze().cpu().numpy().transpose(1, 2, 0))
        self.gt_pos_scatter.set_offsets(data['proj_uvz'][:2])


class LedStatusWidget:

    def __init__(self, plt_axis, title) -> None:
        self.axis = plt_axis
        self.axis.invert_yaxis()
        self.plot_obj = plt_axis.imshow(np.zeros((300, 300, 3), dtype=np.uint8))
        self.axis.set_title(title)
        self.led_front = patches.Rectangle((110, 230), 80, 20, linewidth=1, edgecolor = 'black', facecolor = 'white')
        self.led_right = patches.Rectangle((100, 150), 20, 80, linewidth=1, edgecolor = 'black', facecolor = 'white')
        self.led_left = patches.Rectangle((180, 150), 20, 80, linewidth=1, edgecolor = 'black', facecolor = 'white')
        self.led_back = patches.Rectangle((110, 130), 80, 20, linewidth=1, edgecolor = 'black', facecolor = 'white')
        self.led_top_left = patches.Rectangle((45, 50), 80, 20, linewidth=1, edgecolor = 'black', facecolor = 'white')
        self.led_top_right = patches.Rectangle((175, 50), 80, 20, linewidth=1, edgecolor = 'black', facecolor = 'white')
        self.axis.add_patch(self.led_front)
        self.axis.add_patch(self.led_right)
        self.axis.add_patch(self.led_left)
        self.axis.add_patch(self.led_back)
        self.axis.add_patch(self.led_top_left)
        self.axis.add_patch(self.led_top_right)
        self.axis.tick_params(bottom = False, left = False,
                              top = False, right = False,
                              labelleft = False, labelbottom = False) 

    def update(self, data):
        self.led_front.set(facecolor = self.get_led_indicator_color(data["led_bf"]))
        self.led_right.set(facecolor = self.get_led_indicator_color(data["led_br"]))
        self.led_back.set(facecolor = self.get_led_indicator_color(data["led_bb"]))
        self.led_left.set(facecolor = self.get_led_indicator_color(data["led_bl"]))
        self.led_top_left.set(facecolor = self.get_led_indicator_color(data["led_tl"]))
        self.led_top_right.set(facecolor = self.get_led_indicator_color(data["led_tr"]))
    
    def get_led_indicator_color(self, led_status):
        return 'green' if led_status else 'white'


class ProjectionGTWidget:

    def __init__(self, plt_axis, title = '') -> None:
        self.axis = plt_axis
        self.plot_obj = plt_axis.imshow(np.zeros((360, 640, 3), dtype=np.uint8),
                                        vmin = 0, vmax = 1, cmap = 'viridis')
        self.axis.set_title(title)
        self.pos_scatter = self.axis.scatter(0, 0, s = 1001,
                                             facecolor = 'none', edgecolors = 'blue')

    def update(self, data):
        self.plot_obj.set_data(data['pos_map'].squeeze().cpu().numpy())
        self.pos_scatter.set_offsets(data['proj_uvz'][:2])
        
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

    def __init__(self, plt_axis, title) -> None:
        super().__init__(plt_axis, title)
        self.prediction_scatter = self.axis.scatter(0, 0, marker = 'o', s=1001, facecolor = 'none', edgecolors = 'red')

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

    def __init__(self, plt_axis, title = '') -> None:
        super().__init__(plt_axis, title)
        self.dis_plot_pred = self.axis.scatter(0, 0, marker='o', s=51,
                             facecolor='none', edgecolor='red')
        
    def update(self, data, model):
        super().update(data)
        dist_pred = model.predict_dist(data['image']).squeeze()
        print(dist_pred)
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

