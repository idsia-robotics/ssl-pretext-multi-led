from typing import Any, Tuple
import torch
from src.models import BaseModel, ModelRegistry
from torch.nn.functional import binary_cross_entropy as bce
from torchvision.transforms.functional import resize, InterpolationMode
from numpy import unravel_index, stack
import numpy as np
import torch.nn.functional as F



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
        else:
            raise NotImplementedError("Invalid led inference mode")
        self.downscaler = torch.nn.AvgPool2d(8)


    def predict_pos_from_outs(self, image, outs, to_numpy= True):
        outs = outs[:, :1, ...]
        if not to_numpy:
            return outs

        out_map_shape = outs.shape[-2:]
        outs = outs.view(outs.shape[0], -1)
        max_idx = outs.argmax(1).cpu()
        indexes = unravel_index(max_idx, out_map_shape)
        #               x               y
        indexes = stack([indexes[1], indexes[0]]).T.astype('float32')
        indexes /= np.array([out_map_shape[1], out_map_shape[0]])
        indexes *= np.array([image.shape[-1], image.shape[-2]])

        y_scale_f = image.shape[0] / out_map_shape[0]
        x_scale_f = image.shape[1] / out_map_shape[1]
        indexes += np.array([x_scale_f, y_scale_f]) / 2
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
            self.MAX_DIST_M = 5.
        self.downscaler = torch.nn.AvgPool2d(8)

        
    def _robot_pose_and_leds_loss(self, batch, model_out):
        supervised_label = batch["supervised_flag"].to(model_out.device)

        proj_loss, proj_map_norm = self._robot_projection_loss(batch, model_out, return_norm=True)
        dist_loss = self._robot_distance_loss(batch, model_out, proj_map_norm)
        
        orientation_loss = self._robot_orientation_loss(batch, model_out, proj_map_norm)
        led_loss, led_losses = self._robot_led_loss(batch, model_out, proj_map_norm)

        unsupervised_label = ~supervised_label
        
        led_loss = (led_loss.mean(-1) * (3/2)) * unsupervised_label

        proj_loss_norm = proj_loss * supervised_label
        dist_loss_norm = (dist_loss / self.MAX_DIST_M ** 2) * supervised_label
        ori_loss_norm = (orientation_loss / 4) * supervised_label
        
        return proj_loss_norm, dist_loss_norm, ori_loss_norm,\
            led_loss, led_losses       
    
    def _robot_projection_loss(self, batch, model_out : torch.Tensor, return_norm = False):
        proj_pred = self.predict_pos_from_outs(
            image = batch["image"].to(model_out.device),
            outs=model_out,
            to_numpy=False).float()
        downscaled_gt_proj = self.downscaler(batch['pos_map'][:, None, ...].to(model_out.device))
        # downscaled_gt_proj = downscaled_gt_proj / torch.sum(downscaled_gt_proj, dim = (-1, -2), keepdim=True)
        # downscaled_gt_proj_norm = downscaled_gt_proj / (downscaled_gt_proj.sum(axis = (-1, -2), keepdims = True) + self.epsilon)
        proj_pred_norm = proj_pred / (proj_pred + self.epsilon).sum(axis=(-1, -2), keepdims=True)
        # loss = torch.nn.functional.mse_loss(proj_pred.float(), downscaled_gt_proj.float(), reduction='none').mean(axis = (-1, -2, -3))
        loss = 1 - (proj_pred_norm * downscaled_gt_proj).sum(axis = (-1, -2, -3))
        # loss = torch.nn.functional.mse_loss(
        #     proj_pred,
        #     downscaled_gt_proj.float(),
        #     reduction='none'
        # ).mean(dim = (-1, -2))
        if return_norm:
            return loss, proj_pred_norm.detach()
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
        self.loss = self._robot_pose_and_leds_loss

        self.layers = torch.nn.Sequential(
            self.core_layers,
            self.robot_pose_and_led_layer
        )
    

@ModelRegistry("model_s_opt")
class Model_s_optimized(Model_s):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.Conv2d(6, 8, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, kernel_size=5, padding=6, stride=1, dilation = 3, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=5, padding=4, stride=1, dilation=2, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1),

        )

        if self.task == 'presence':
            self.robot_presence_layer = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.LazyLinear(256),
                torch.nn.LazyLinear(1),
                torch.nn.Sigmoid()
            )
            self.layers = torch.nn.Sequential(
                self.core_layers,
                self.robot_presence_layer
            )
            self.loss = self.__robot_presence_loss
        if self.task == 'position':
            self.robot_position_layer = torch.nn.Sequential(
                torch.nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1),
                torch.nn.Conv2d(3, 1, kernel_size=1, padding=0, stride=1),
                torch.nn.Sigmoid()
            )

            self.layers = torch.nn.Sequential(
                self.core_layers,
                self.robot_position_layer
            )
            self.loss = self._robot_position_loss
        elif self.task == 'pose':
            self.robot_pose_layer = torch.nn.Sequential(
                torch.nn.Conv2d(32, 4, kernel_size=1, padding=0, stride=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(4),
                torch.nn.Conv2d(4, 4, kernel_size=1, padding=0, stride=1),
                # torch.nn.Sigmoid()
            )

            self.forward = self.__position_and_orientation_forward
            self.loss = self.__robot_pose_loss
            self.layers = torch.nn.Sequential(
                self.core_layers,
                self.robot_pose_layer
            )
            self.MAX_DIST_M = 5.
        elif self.task == 'pose_and_led':
            self.robot_pose_and_led_layer = torch.nn.Sequential(
                torch.nn.Conv2d(32, 10, kernel_size=1, padding=0, stride=1, bias=False),
                torch.nn.BatchNorm2d(10),
                torch.nn.ReLU(),
                torch.nn.Conv2d(10, 10, kernel_size=1, padding=0, stride=1),
                # torch.nn.Sigmoid()
            )
            self.forward = self.pose_and_leds_forward
            self.loss = self._robot_pose_and_leds_loss

            self.layers = torch.nn.Sequential(
                self.core_layers,
                self.robot_pose_and_led_layer
            )
            self.MAX_DIST_M = 5.

class ConvBlock(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias = False),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias = False),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias = False),
            torch.nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layers(x)
        

@ModelRegistry("deep_model_s")
class Deep_Model_s(Model_s):
    def __init__(self, *args, **kwargs):
        super(Deep_Model_s, self).__init__(*args, **kwargs)

        self.core_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=7, padding=3, stride=2, bias = False),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.Conv2d(6, 8, kernel_size=5, padding=2, stride=1, bias = False),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=5, padding=2, stride=2, bias = False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=1, bias = False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2, bias = False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1, bias = False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1, bias = False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 10, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1),

        )

        if self.task == 'pose_and_led':
            self.deep_robot_pose_and_led_layer = [ConvBlock(10, 1) for _ in range(10)]
            self.forward = lambda x: Deep_Model_s.forward(self, x)

            self.layers = torch.nn.Sequential(
                self.core_layers,
            )
            self.MAX_DIST_M = 5.
        
    def forward(self, x):
        core_out = self.core_layers(x)
        result = torch.cat(
            [b(core_out) for b in self.deep_robot_pose_and_led_layer],
            dim = 1
        )
        out = torch.cat(
            [
                torch.nn.functional.sigmoid(result[:, :2, ...]), # pos and dist
                torch.nn.functional.tanh(result[:, 2:4, ...]), # orientation
                torch.nn.functional.sigmoid(result[:, 4:, ...]), # leds
                
            ],
            axis = 1)
        return out
    
    def to(self, *args, **kwargs):
        res = super().to(*args, **kwargs)
        self.deep_robot_pose_and_led_layer = [
            b.to(*args, **kwargs) for b in self.deep_robot_pose_and_led_layer
        ]
        return res
