from typing import Any
import torch
from src.models import BaseModel, ModelRegistry
from numpy import unravel_index, stack
import numpy as np

class RangeRescaler(torch.nn.Module):

    def __init__(self, in_range = [0., 1.], out_range = [0., 1.]) -> None:
        super().__init__()
        self.in_min = in_range[0]
        self.in_max = in_range[1]
        self.in_gap = self.in_max - self.in_min
        self.out_min = out_range[0]
        self.out_max = out_range[1]
        self.out_gap = self.out_max - self.out_min
        self.gap_ratio = self.out_gap / self.in_gap


    def __call__(self, x) -> Any:
        return (x - self.in_min) * self.gap_ratio + self.out_min
        
class FullyConvPredictorMixin:

    def __init__(self, *args, **kwargs) -> None:
        self.led_inference = kwargs.pop('led_inference')
        super().__init__(*args, **kwargs)

        if self.led_inference == 'gt':
            self.predict_leds = self._predict_led_gt_pos
        elif self.led_inference == 'pred':
            self.predict_leds = self._predict_led_pred_pos
        elif self.led_inference == 'hybrid':
            self.predict_leds = self._predict_led_hybrid
        elif self.led_inference == 'amax':
            self.predict_leds = self._predict_led_amax
        elif self.led_inference == "mean":
            self.predict_leds = self._predict_led_mean
        else:
            raise NotImplementedError("Invalid led inference mode")
        self.downscaler = torch.nn.AvgPool2d(8)


    def predict_pos_from_outs(self, image, outs, to_numpy= True):
        outs = outs[:, :1, ...]
        if not to_numpy:
            return outs

        out_map_shape = outs.shape[-2:]
        # outs = outs.view(outs.shape[0], -1)
        # max_idx = outs.argmax(1).cpu()
        # indexes = unravel_index(max_idx, out_map_shape)
        # #               x               y
        # indexes = stack([indexes[1], indexes[0]]).T.astype('float32')
        # indexes /= np.array([out_map_shape[1], out_map_shape[0]])
        # indexes *= np.array([image.shape[-1], image.shape[-2]])

        # y_scale_f = image.shape[0] / out_map_shape[0]
        # x_scale_f = image.shape[1] / out_map_shape[1]
        # indexes += np.array([x_scale_f, y_scale_f]) / 2
        # return indexes.astype(np.int32)

        outs = outs[:, :1, ...].detach()
        maxs = outs.flatten(-2).max(-1).values
        thr = maxs * .99
        outs = (outs > thr[..., None, None]) * outs
        ii, jj = torch.meshgrid(torch.arange(outs.shape[-2]), torch.arange(outs.shape[-1]), indexing='ij')
        coords = torch.stack([torch.reshape(ii, (-1,)), torch.reshape(jj, (-1,))], axis = -1)
        reshaped_maps = torch.reshape(outs, [-1, outs.shape[-2] * outs.shape[-1], 1])
        total_mass = torch.sum(reshaped_maps, axis = 1)
        centre_of_mass = torch.sum(reshaped_maps * coords, axis = 1) / total_mass

        indexes = stack([centre_of_mass[:, 1], centre_of_mass[:, 0]]).T.astype('float32')
        indexes /= np.array([out_map_shape[1], out_map_shape[0]])
        indexes *= np.array([image.shape[-1], image.shape[-2]])

        return indexes.astype(np.int32)

    def predict_pos(self, image):
        outs = self(image)
        return self.predict_pos_from_outs(image, outs)
    
    def predict_dist_from_outs(self, outs, to_numpy= True, pos_norm = None):
        if pos_norm is None:
            pos_map = outs[:, 0, ...].detach()
            pos_map_norm = pos_map / torch.sum(pos_map, axis = (-1, -2), keepdim=True)
        else:
            pos_map_norm = pos_norm.detach().squeeze()

        dist_map = outs[:, 1, ...]
        dist_scalars = (dist_map * pos_map_norm).sum(axis = (-1, -2))
        if not to_numpy:
            return dist_scalars
        else:
            return dist_scalars.detach().cpu().numpy()
    
    def predict_dist(self, image):
        outs = self(image)
        return self.predict_dist_from_outs(outs)
    
    def predict_orientation_from_outs(self, outs, to_numpy = True, pos_norm = None):
        if pos_norm is None:
            pos_map = outs[:, :1, ...].detach()
            pos_map_norm = (pos_map / torch.sum(pos_map, axis = (-1, -2), keepdim=True)).squeeze()
        else:
            pos_map_norm = pos_norm.detach().squeeze()

        if len(pos_map_norm.shape) < 4:
            pos_map_norm = pos_map_norm[..., None, :, :]
            
        maps =outs[:, 2:4, ...]
        scalars = (maps * pos_map_norm).sum(axis = (-1, -2))
        cos_scalars = scalars[:, 0, ...]
        sin_scalars = scalars[:, 1, ...]
        if not to_numpy:
            return cos_scalars, sin_scalars
        else:
            thetas = torch.atan2(sin_scalars, cos_scalars).detach().cpu().numpy()
            return thetas, cos_scalars.detach().cpu().numpy(), sin_scalars.detach().cpu().numpy()

    def _predict_led_pred_pos(self, outs, batch, to_numpy= True, pos_norm=None):
        led_maps = outs[..., 4:, :, :]
        if pos_norm is None:
            pos_map = outs[..., :1, :, :]
            pos_norm = (pos_map / torch.sum(pos_map, axis = (-1, -2), keepdim = True)).detach()
        masked_maps = pos_norm * led_maps
        if not to_numpy:
            return masked_maps.sum(axis = [-1, -2])
        else:
            return masked_maps.sum(axis = [-1, -2]).detach().cpu().numpy()
        
    def _predict_led_gt_pos(self, outs, batch, to_numpy= True, pos_norm = None):
        led_maps = outs[:, 4:, ...]
        pos_map = self.downscaler(batch["pos_map"].to(outs.device))[..., None, :, :]
        pos_map_norm = pos_map / (torch.sum(pos_map, axis = (-1, -2), keepdim=True) + self.epsilon)
        masked_maps = pos_map_norm * led_maps

        if not to_numpy:
            return masked_maps.sum(axis = [-1, -2]) * .9999
        else:
            return masked_maps.sum(axis = [-1, -2]).detach().cpu().numpy() * .9999
        
    def _predict_led_hybrid(self, outs, batch, to_numpy= True, pos_norm=None):
        led_maps = outs[:, 4:, ...]
        size = outs.shape[-2:]
        size = size[0] * size[1]
        pos_preds = outs[:, :1, ...]
        pos_map = torch.ones_like(pos_preds, device=outs.device) / size
        vis_flag = batch["robot_visible"]
        pos_map[vis_flag, ...] = pos_preds[vis_flag, ...]
        pos_map_norm = pos_map / torch.sum(pos_map, axis = (-1, -2), keepdim=True)
        masked_maps = pos_map_norm * led_maps
        if not to_numpy:
            return masked_maps.sum(axis = [-1, -2])
        else:
            return masked_maps.sum(axis = [-1, -2]).detach().cpu().numpy()

    def _predict_led_amax(self, outs, batch, to_numpy= True, pos_norm = None):
        led_maps = outs[:, 4:, ...]
        preds = torch.amax(led_maps, dim = (-1, -2))
        if not to_numpy:
            return preds
        else:
            return preds.detach().cpu().numpy()
        
    def _predict_led_mean(self, outs, batch, to_numpy= True, pos_norm = None):
        led_maps = outs[:, 4:, ...]
        # led_maps = torch.pow(led_maps * 2 - 1, 3)
        # preds = (torch.mean(led_maps, dim = (-1, -2)) + 1) / 2
        preds = torch.mean(led_maps, dim = (-1, -2))
        if not to_numpy:
            return preds
        else:
            return preds.detach().cpu().numpy()

    def predict_orientation(self, image):
        outs = self(image)
        return self.predict_orientation_from_outs(outs)
    

@ModelRegistry("model_s")
class Model_s(FullyConvPredictorMixin, BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.core_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.Conv2d(6, 8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, kernel_size=5, padding=6, stride=1, dilation = 3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=5, padding=4, stride=1, dilation=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1),

        )

        if self.task == 'pose_and_led':
            self.robot_pose_and_led_layer = torch.nn.Sequential(
                torch.nn.Conv2d(32, 10, kernel_size=1, padding=0, stride=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(10),
                torch.nn.Conv2d(10, 10, kernel_size=1, padding=0, stride=1),
            )
            self.forward = self.pose_and_leds_forward
            self.loss = self._robot_pose_and_leds_loss

            self.layers = torch.nn.Sequential(
                self.core_layers,
                self.robot_pose_and_led_layer
            )
        elif self.task == 'pretext':
            self.robot_pose_and_led_layer = torch.nn.Sequential(
                torch.nn.Conv2d(32, 10, kernel_size=1, padding=0, stride=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(10),
                torch.nn.Conv2d(10, 10, kernel_size=1, padding=0, stride=1),
            )
            self.forward = self.pose_and_leds_forward
            self.loss = self._robot_pretext_loss

            self.layers = torch.nn.Sequential(
                self.core_layers,
                self.robot_pose_and_led_layer
            )
        elif self.task == 'tuning' or self.task == 'downstream':
            self.robot_pose_and_led_layer = torch.nn.Sequential(
                torch.nn.Conv2d(32, 10, kernel_size=1, padding=0, stride=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(10),
                torch.nn.Conv2d(10, 10, kernel_size=1, padding=0, stride=1),
            )
            self.forward = self.pose_and_leds_forward
            self.loss = self._robot_pose_loss

            self.layers = torch.nn.Sequential(
                self.core_layers,
                self.robot_pose_and_led_layer
            )

        self.MAX_DIST_M = 5.
        self.downscaler = torch.nn.AvgPool2d(8)
            
    def _robot_pose_and_leds_loss(self, batch, model_out):
        supervised_label = batch["supervised_flag"].to(model_out.device)

        proj_loss, proj_map_norm = self._robot_projection_loss(batch, model_out, return_norm=True, detach_norm=False)
        dist_loss = self._robot_distance_loss(batch, model_out, proj_map_norm.detach())
        
        orientation_loss = self._robot_orientation_loss(batch, model_out, proj_map_norm.detach())
        led_loss, led_losses = self._robot_led_loss(batch, model_out, proj_map_norm)

        unsupervised_label = ~supervised_label
        
        led_loss = (led_loss.mean(-1) * (3/2)) * unsupervised_label

        proj_loss_norm = proj_loss * supervised_label
        dist_loss_norm = (dist_loss / self.MAX_DIST_M ** 2) * supervised_label
        ori_loss_norm = (orientation_loss / 4) * supervised_label
        
        return proj_loss_norm, dist_loss_norm, ori_loss_norm,\
            led_loss, led_losses       
    
    def _robot_pretext_loss(self, batch, model_out):
        proj_map_norm = model_out[..., :1, :, :]
        proj_map_norm = proj_map_norm / (torch.sum(proj_map_norm, dim = (-1, -2), keepdim=True) + self.epsilon)

        led_loss, led_losses = self._robot_led_loss(batch, model_out, proj_map_norm)

        
        led_loss = (led_loss.mean(-1) * (3))
        proj_loss_norm = torch.zeros_like(led_loss, requires_grad=True)
        dist_loss_norm = torch.zeros_like(led_loss, requires_grad=True)
        ori_loss_norm = torch.zeros_like(led_loss, requires_grad=True)

        return proj_loss_norm, dist_loss_norm, ori_loss_norm,\
            led_loss, led_losses       

    def _robot_pose_loss(self, batch, model_out):

        proj_loss, proj_map_norm = self._robot_projection_loss(batch, model_out, return_norm=True)
        dist_loss = self._robot_distance_loss(batch, model_out, proj_map_norm)
        orientation_loss = self._robot_orientation_loss(batch, model_out, proj_map_norm)

        
        led_loss = torch.zeros(model_out.shape[0], 1, requires_grad=True)
        led_losses = torch.zeros(6)

        proj_loss_norm = proj_loss
        dist_loss_norm = (dist_loss / self.MAX_DIST_M ** 2)
        ori_loss_norm = (orientation_loss / 4)
        
        return proj_loss_norm, dist_loss_norm, ori_loss_norm,\
            led_loss, led_losses       
    

    def _robot_projection_loss(self, batch, model_out : torch.Tensor, return_norm = False, detach_norm = True):
        proj_pred = self.predict_pos_from_outs(
            image = batch["image"].to(model_out.device),
            outs=model_out,
            to_numpy=False).float()
        downscaled_gt_proj = self.downscaler(batch['pos_map'][:, None, ...].to(model_out.device))
        proj_pred_norm = proj_pred / (proj_pred + self.epsilon).sum(axis=(-1, -2), keepdims=True)
        loss = 1 - (proj_pred_norm * downscaled_gt_proj).sum(axis = (-1, -2, -3))

        if return_norm:
            if detach_norm:
                return loss, proj_pred_norm.detach()
            else:
                return loss, proj_pred_norm
        else:
            return loss
        
    def _robot_distance_loss(self, batch, model_out : torch.Tensor, proj_map_norm : torch.Tensor):
        dist_pred = self.predict_dist_from_outs(model_out,
                                                to_numpy=False,
                                                pos_norm=proj_map_norm).float()
        dist_gt = batch["distance_rel"].to(model_out.device)
        error = torch.nn.functional.mse_loss(dist_pred, dist_gt, reduction='none')
        return error
    
    def _robot_orientation_loss(self, batch, model_out, proj_map_norm):
        theta = batch["pose_rel"][:, -1].to(model_out.device)
        cos_pred, sin_pred = self.predict_orientation_from_outs(
            outs=model_out,
            to_numpy=False,
            pos_norm=proj_map_norm
        )
        theta_cos = torch.cos(theta)
        theta_sin = torch.sin(theta)
        cos_error = torch.nn.functional.mse_loss(cos_pred.float(), theta_cos, reduction='none')
        sin_error = torch.nn.functional.mse_loss(sin_pred.float(), theta_sin, reduction='none')

        return cos_error + sin_error

    def _robot_led_loss(self, batch, model_out, proj_map_norm):
        led_trues = batch["led_mask"].to(model_out.device) # BATCH_SIZE x 6
        led_preds = self.predict_leds(
            model_out,
            batch,
            to_numpy=False,
            pos_norm=proj_map_norm
        ).float()
        losses = torch.zeros_like(led_trues, device=model_out.device, dtype=torch.float32)
        for i in range(led_preds.shape[1]):
            losses[:, i] = torch.nn.functional.binary_cross_entropy(
                    led_preds[:, i], led_trues[:, i].float(), reduction='none')
        losses[:, 1] = 0.
        losses[:, 2] = 0.
        return losses, losses.detach().mean(0)        

    def pose_and_leds_forward(self, x):
        out = self.layers(x)
        out = torch.cat(
            [
                torch.nn.functional.sigmoid(out[:, :2, ...]), # pos and dist
                torch.nn.functional.tanh(out[:, 2:4, ...]), # orientation
                torch.nn.functional.sigmoid(out[:, 4:, ...]), # leds
                
            ],
            axis = 1)
        out[:, 1, ...] = out[:, 1, ...] * self.MAX_DIST_M
        return out

    def to(self, *args, **kwargs):
        res = super().to(*args, **kwargs)
        self.downscaler = self.downscaler.to(*args, **kwargs)
        return res

@ModelRegistry("model_s_wide")
class Model_s_wide(Model_s):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.core_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 12, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(),
            torch.nn.Conv2d(12, 24, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(24, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=5, padding=6, stride=1, dilation = 3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=5, padding=4, stride=1, dilation=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1),

        )
        self.robot_pose_and_led_layer = torch.nn.Sequential(
                torch.nn.Conv2d(64, 10, kernel_size=1, padding=0, stride=1),
                torch.nn.BatchNorm2d(10),
                torch.nn.ReLU(),
                torch.nn.Conv2d(10, 10, kernel_size=1, padding=0, stride=1),
                torch.nn.BatchNorm2d(10)
        )
        self.forward = self.pose_and_leds_forward
        # self.loss = self._robot_pose_and_leds_loss

        self.layers = torch.nn.Sequential(
            self.core_layers,
            self.robot_pose_and_led_layer
        )
    
@ModelRegistry("model_s_small_rf")
class Model_s_wide(Model_s):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.core_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 12, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(),
            torch.nn.Conv2d(12, 24, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(24, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=5, padding=6, stride=1, dilation = 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=5, padding=4, stride=1, dilation= 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1),

        )
        self.robot_pose_and_led_layer = torch.nn.Sequential(
                torch.nn.Conv2d(64, 10, kernel_size=1, padding=0, stride=1),
                torch.nn.BatchNorm2d(10),
                torch.nn.ReLU(),
                torch.nn.Conv2d(10, 10, kernel_size=1, padding=0, stride=1),
                torch.nn.BatchNorm2d(10)
        )
        self.forward = self.pose_and_leds_forward
        # self.loss = self._robot_pose_and_leds_loss

        self.layers = torch.nn.Sequential(
            self.core_layers,
            self.robot_pose_and_led_layer
        )