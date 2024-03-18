from itertools import combinations
import matplotlib as mpl
import matplotlib.patches as patches
import numpy as np
import torch
from src.dataset.dataset import H5Dataset
from src.inference import reconstruct_position, point_rel_to_camera
from scipy.spatial.transform.rotation import Rotation as R
from src.inference import gimbal_to_base

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
        self.axis.set_title(title.title())
        orb_size = H5Dataset.POS_ORB_SIZE
        scatter_size = (np.sqrt(2) * orb_size / 2) ** 2 * np.pi / 72
        # self.pos_scatter = self.axis.scatter(0, 0, s = scatter_size,
                                            #  facecolor = 'none', edgecolors = 'blue')
        
        self.receptive_field = receptive_field
        if receptive_field:
            self.rf_plot = patches.Rectangle((0, 0), receptive_field, receptive_field, linewidth=1, edgecolor = 'blue', facecolor = 'none')   
            self.axis.add_patch(self.rf_plot)

        self.axis.tick_params(bottom = False, left = False,
                              top = False, right = False,
                              labelleft = False, labelbottom = False) 


    def update(self, data):
        self.plot_obj.set_data(data['pos_map'].squeeze().cpu().numpy())
        # self.pos_scatter.set_offsets(data['proj_uvz'][:2])

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
        rm1_pose = data["RM6_pose"].squeeze()[:3]
        rm2_pose = data["RM2_pose"].squeeze()[:3]
        gt_positions = np.stack([rm1_pose, rm2_pose], axis = 0) # [[x1, y1], [x2, y2]]
        self.position_scatter.set_offsets(gt_positions)
        
        theta_1 = data["RM6_pose"].squeeze()[-1] # (2,)
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
        # out = out / out.sum()
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
        alpha = 1. if data["robot_visible"] else 0.
        self.dis_plot_gt.set_alpha(alpha)
        self.dis_plot_gt.set_offsets([0, data["distance_rel"].squeeze()])

        
class DistanceInferenceWidget(DistanceWidget):

    def __init__(self, plt_axis, title = '', color = 'red') -> None:
        super().__init__(plt_axis, title)
        self.dis_plot_pred = self.axis.scatter(0, 0, marker='o', s=11,
                             facecolor=color, edgecolor=color)
        
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
        ori = model.predict_orientation(data["image"])[0].squeeze()
        self.pred_plot.set_offsets([ori, 1.5])


class RobotLedInferenceWidget:
    def __init__(self, plt_axes, title = '') -> None:
        self.axes = plt_axes
        self.image_plot = self.axes[4].imshow(np.zeros((360, 640, 3)))
        led_out_idx = [0, 1, 2, 3, 5, 7]

        cmap = mpl.colormaps.get_cmap('coolwarm')  # viridis is the default colormap for imshow
        cmap.set_bad(color='green')
        self.led_plots = [self.axes[i].imshow(np.zeros((360, 640, 1)), cmap =cmap, vmin = 0, vmax = 1, interpolation = 'none') for i in led_out_idx]

        self.axes[0].set_title("TOP LEFT")
        self.axes[1].set_title("FRONT")
        self.axes[2].set_title("TOP RIGHT")
        self.axes[3].set_title("LEFT")
        self.axes[5].set_title("RIGHT")
        self.axes[7].set_title("BACK")

        self.thresholds = np.array([0.63, 0.852335, 0.7073505, 0.65392925, 0.65581376, 0.71283405])


        orb_size = H5Dataset.POS_ORB_SIZE
        scatter_size = (np.sqrt(2) * orb_size / 2) ** 2 * np.pi / 72

        self.pos_scatters = [self.axes[i].scatter(0,0, facecolor = 'none', color = 'red', s = scatter_size, ) for i in led_out_idx]

        self.led_status_widget = LedStatusWidget(self.axes[-1], title='LED status')
        self.proj_gt_widget = PoseWidget(self.axes[-3], 'GT pos') 
#        self.proj_gt_widget = ModelProjOutput(self.axes[-3], 'GT pos') 

        # self.axes.set_title(title)
        for ax in self.axes:
            ax.tick_params(bottom = False, left = False, top = False, right = False, labelleft = False, labelbottom = False) 

    def update(self, data, model):
        out = model(data['image']).squeeze().detach().cpu()
        preds = model.predict_leds(out, data)
        pos_out = out[:1, :, :]# / torch.sum(out[:1, :, :], dim = (-1, -2), keepdim=True)

        
        out = (out[..., -6:, :, :] * pos_out * 100).squeeze()
        out[:, pos_out.squeeze() < .1] = np.nan

        self.image_plot.set_data(data['image_with_gun'].squeeze().numpy().transpose(1, 2, 0))
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
            z = data['proj_uvz'][-1][-1]
            if z > 0:
                proj_uv = data['proj_uvz'][:2]
                pos_scatter.set_offsets(proj_uv)
            else:
                pos_scatter.set_offsets((-1000, -1000))
        self.led_status_widget.update(data)
        self.proj_gt_widget.update(data, model)

class PoseWidget:

    def __init__(self, ax, title = '', mode = 'predicted') -> None:
        self.axis = ax

        self.image_plot = self.axis.imshow(np.zeros((360, 640, 3)))
        self.axis.tick_params(bottom = False, left = False,
                              top = False, right = False,
                              labelleft = False, labelbottom = False) 
        
        self.linspace_count = 2

        self.x_axis_proj_line = self.axis.plot((0, 0), (1,1), color = 'magenta')[0]
        self.y_axis_proj_line = self.axis.plot((0, 0), (1,1), color = 'green')[0]
        self.z_axis_proj_line = self.axis.plot((0, 0), (1,1), color = 'blue')[0]

        self.axis.set_xlim((0, 640))
        self.axis.set_ylim((0, 360))
        self.axis.invert_yaxis()
        self.bb_lines = [self.axis.plot((0, 0), (0,0), color = 'red')[0] for i in range(12)]



        self.mode = mode
        self.axis.set_title(title.title())

    def target_to_camera_transform(self, data, model):
        if self.mode == 'predicted':
            proj_pred = model.predict_pos(data['image'])
            dist_pred = model.predict_dist(data['image'])
            ori_pred = model.predict_orientation(data['image'])[0]
            position_pred = reconstruct_position(proj_pred.T, dist_pred, np.linalg.inv(data["base_to_camera"].numpy()))
            rot_mat = R.from_euler('z', -ori_pred).as_matrix().squeeze() 
            T = np.eye(4)
            T[:-1, :-1] = rot_mat
            T[0, -1] = position_pred[0]
            T[1, -1] = position_pred[1]
            T[2, -1] = .118
            return T
        elif self.mode == "true":
            proj_pred = data['proj_uvz'][:, :2].numpy()
            dist_pred = data["pose_rel"][..., 0].numpy()
            ori_pred = data["pose_rel"].squeeze().numpy()[-1] 

            position_pred = data["pose_rel"][..., :2].numpy().squeeze()
            rot_mat = R.from_euler('z', -ori_pred).as_matrix().squeeze() 

            T = np.eye(4)
            T[:-1, :-1] = rot_mat
            T[0, -1] = position_pred[0]
            T[1, -1] = position_pred[1]
            T[2, -1] = .118
            return T
    def update(self, data, model = None):

        assert (not self.mode == 'predicted') or model is not None


        # line_alpha = 1. if data["robot_visible"] else 0.
        line_alpha = 1.

        T = self.target_to_camera_transform(data, model)
        
        x_vec = (T @ np.linspace((0., 0., 0., 1.), (.25, 0., 0., 1.), self.linspace_count).T).T
        y_vec = (T @  np.linspace((0., 0., 0., 1.), (.0, .25, 0., 1.), self.linspace_count).T).T
        z_vec = (T @  np.linspace((0., 0., 0., 1.), (.0, 0., .25, 1.), self.linspace_count).T).T

        x_point = x_vec[..., :-1]
        y_point = y_vec[..., :-1]
        z_point = z_vec[..., :-1]
        x_point_proj = point_rel_to_camera(x_point, camera_pose = data["base_to_camera"].numpy().squeeze())
        y_point_proj = point_rel_to_camera(y_point, camera_pose = data["base_to_camera"].numpy().squeeze())
        z_point_proj = point_rel_to_camera(z_point, camera_pose = data["base_to_camera"].numpy().squeeze())


        self.x_axis_proj_line.set_xdata(x_point_proj[:, 0])
        self.x_axis_proj_line.set_ydata(x_point_proj[:, 1])
        self.x_axis_proj_line.set_alpha(line_alpha)

        self.y_axis_proj_line.set_xdata(y_point_proj[:, 0])
        self.y_axis_proj_line.set_ydata(y_point_proj[:, 1])
        self.y_axis_proj_line.set_alpha(line_alpha)


        self.z_axis_proj_line.set_xdata(z_point_proj[:, 0])
        self.z_axis_proj_line.set_ydata(z_point_proj[:, 1])
        self.z_axis_proj_line.set_alpha(line_alpha)



        # BB
        # The lazy man's guide to getting all adjacent vertexes in a cube
        ls = np.linspace(-.27/2, .27/2, 2)
        bb_x, bb_y, bb_z = np.meshgrid(ls, ls, ls, indexing='ij')
        points = np.stack([bb_x.flatten(), bb_y.flatten(), bb_z.flatten(), np.ones(8)])
        combs = np.array(list(combinations(points.T.tolist(), 2)))
        connected = (combs[:, 0, :] == combs[:, 1, :]).sum(1) == 3
        line_pairs = combs[connected]

        # 8x2x4
        line_pairs[:, 0, :] = (T @ line_pairs[:, 0, :].T).T
        line_pairs[:, 1, :] = (T @ line_pairs[:, 1, :].T).T

        first_points = point_rel_to_camera(line_pairs[:, 0, :-1], camera_pose = data["base_to_camera"].numpy().squeeze())
        second_points = point_rel_to_camera(line_pairs[:, 1, :-1], camera_pose = data["base_to_camera"].numpy().squeeze())

        points_proj = np.stack([first_points, second_points], axis = 1)

        for i in range(12):
            self.bb_lines[i].set_xdata(points_proj[i, :, 0])
            self.bb_lines[i].set_ydata(points_proj[i, :, 1])
            self.bb_lines[i].set_alpha(line_alpha)

        self.image_plot.set_data(data['image_with_gun'].squeeze().cpu().numpy().transpose(1, 2, 0))