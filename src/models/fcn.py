from typing import Any, Tuple
import torch
from src.models import BaseModel, ModelRegistry
from torch.nn.functional import binary_cross_entropy as bce
from torch.nn.functional import mse_loss
from torchvision.transforms.functional import resize
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
        

    
@ModelRegistry("model_s")
class Model_s(BaseModel):
    def __init__(self, *args, **kwargs):
        super(Model_s, self).__init__(*args, **kwargs)

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
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, dilation = 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=0, stride=1, dilation=2),
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
            self.loss = self.__robot_position_loss
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
                torch.nn.Conv2d(32, 10, kernel_size=1, padding=0, stride=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(10),
                torch.nn.Conv2d(10, 10, kernel_size=1, padding=0, stride=1),
                # torch.nn.Sigmoid()
            )
            self.forward = self.__pose_and_leds_forward
            self.loss = self.__robot_pose_and_leds_loss

            self.layers = torch.nn.Sequential(
                self.core_layers,
                self.robot_pose_and_led_layer
            )
            self.MAX_DIST_M = 5.
        
    
    def __robot_presence_loss(self, batch, y_pred):
        y_pred_probs = torch.zeros((y_pred.shape[0], 2))
        y_pred_probs[:, 1] = y_pred[:, 0]
        y_pred_probs[:, 0] = 1 - y_pred[:, 0]

        y_true_classes = torch.nn.functional.one_hot(batch['robot_visible'].long(), 2).float()
        return bce(y_pred_probs, y_true_classes)

    def __robot_position_loss(self, batch, model_out : torch.tensor, eps = 1e-6):
        pos_true = batch['pos_map'][:, None, ...].to(model_out.device)
        pos_true = resize(pos_true, model_out.shape[-2:], antialias=False).float()

        pos_pred_sum = torch.sum(model_out + eps, axis = [-1, -2], keepdim=True)
        pos_pred_norm = model_out / pos_pred_sum
        self.__pose_pred_norm_cache = pos_pred_norm.detach()
        loss = 1 - (pos_pred_norm * pos_true).sum(axis = (-1, -2))
        return loss
    
    def __robot_distance_loss(self, batch, model_out):
        dist_out = model_out[:, 1:2, ...]
        dist_gt = batch["distance_rel"].to(dist_out.device)
        pos_out_norm = self.__pose_pred_norm_cache.detach()
        error = (dist_gt[:, None, None, None] - dist_out) ** 2
        return (error * pos_out_norm).sum(axis = [-1, -2])

    def __robot_orientation_loss(self, batch, model_out):
        theta = batch["pose_rel"][:, -1].to(model_out.device)
        theta_cos = torch.cos(theta)
        theta_sin = torch.sin(theta)
        pos_out_norm = self.__pose_pred_norm_cache.detach()

        model_out_cos = model_out[:, 2:3, ...]
        model_out_sin = model_out[:, 3:4, ...]
        cos_error = (theta_cos[:, None, None, None] - model_out_cos) ** 2
        sin_error = (theta_sin[:, None, None, None] - model_out_sin) ** 2
        return ((cos_error + sin_error) * pos_out_norm).sum(axis = [-1, -2])
    

    def __robot_pose_loss(self, batch, model_out : Tuple[torch.tensor, torch.tensor]):
        proj_loss = self.__robot_position_loss(batch, model_out[:, :1, ...])
        dist_loss = self.__robot_distance_loss(batch, model_out)
        orientation_loss = self.__robot_orientation_loss(batch, model_out)

        # Rescale all 3 losses to [0, 1]
        # proj loss is already in [0, 1]

        # Distance loss will never be exactly at the [0, 1] range because i'm not sure what the max
        # distance between the camera and the robot will be in the dataset.
        proj_loss_norm = proj_loss
        dist_loss_norm = dist_loss / self.MAX_DIST_M
        ori_loss_norm = orientation_loss / 4
        return .5 * proj_loss_norm + .3 * dist_loss_norm + .2 * ori_loss_norm, \
                    proj_loss.detach().mean(), dist_loss.detach().mean(), orientation_loss.detach().mean()



    def __led_status_loss(self, batch, model_out):
        led_outs = model_out[:, 4:, ...]
        pos_trues = batch["pos_map"].to(led_outs.device)
        pos_trues = resize(pos_trues, model_out.shape[-2:], antialias=False).float()
        pos_trues = pos_trues / pos_trues.sum(axis = [-1, -2])[..., None, None]

        led_trues = batch["led_mask"].to(led_outs.device) # BATCH_SIZE x 6

        masked_led_outs = led_outs * pos_trues[:, None, ...]
        led_preds = masked_led_outs.sum(axis=[-1, -2])
        losses = [0] * 6
        for i in range(led_preds.shape[1]):
            led_mask = batch["led_visibility_mask"].to(model_out.device)[:, i]
            losses[i] = torch.nn.functional.binary_cross_entropy(
                led_preds[:, i], led_trues[:, i].float(), reduction='none'
            ) * led_mask
            losses[i] = losses[i].sum() / led_mask.sum()
        return sum(losses) / 6, losses

    
    def __robot_pose_and_leds_loss(self, batch, model_out, epoch, weights):
        proj_loss = self.__robot_position_loss(batch, model_out[:, :1, ...])
        dist_loss = self.__robot_distance_loss(batch, model_out)
        orientation_loss = self.__robot_orientation_loss(batch, model_out)
        led_loss, led_losses = self.__led_status_loss(batch, model_out)

        proj_loss_norm = proj_loss
        dist_loss_norm = dist_loss / self.MAX_DIST_M
        ori_loss_norm = orientation_loss / 2

        loss = weights['pos'] * proj_loss_norm + weights['dist'] * dist_loss_norm + weights['ori'] * ori_loss_norm + \
        led_loss * weights['led']
        
        return loss, proj_loss.detach().mean(), dist_loss.detach().mean(), orientation_loss.detach().mean(),\
            led_loss, led_losses
        # if epoch == -1:
        #     return .8 * pose_loss + .0 * led_loss, led_loss, proj_loss, dist_loss, ori_loss,\
        #     led_losses
        # else:
        #     return .8 * pose_loss + .2 * led_loss, led_loss, proj_loss, dist_loss, ori_loss,\
        #     led_losses



    def __position_and_orientation_forward(self, x):
        out = self.layers(x)
        out = torch.cat(
            [
                torch.nn.functional.sigmoid(out[:, :2, ...]),
                #torch.nn.functional.tanh(out[:, 2:, ...]),
                out[:, 2:, ...]
            ],
            axis = 1)
        out = out * torch.tensor([1., self.MAX_DIST_M, 1., 1.])[None, :, None, None].to(out.device)
        return out
    
    def __pose_and_leds_forward(self, x):
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

    

    def predict_pos_from_out(self, image, outs):
        outs = outs[:, :1, ...]
        out_map_shape = outs.shape[-2:]
        outs = outs.view(outs.shape[0], -1)
        max_idx = outs.argmax(1).cpu()
        indexes = unravel_index(max_idx, out_map_shape)
        #               x               y
        indexes = stack([indexes[1], indexes[0]]).T.astype('float32')
        indexes /= np.array([out_map_shape[1], out_map_shape[0]])
        indexes *= np.array([image.shape[-1], image.shape[-2]])
        return indexes.astype(np.int32)
    
    def predict_pos(self, image):
        outs = self(image)
        return self.predict_pos_from_out(image, outs)
    
    def predict_dist_from_outs(self, outs):
        pos_map = outs[:, 0, ...]
        dist_map = outs[:, 1, ...]
        pos_map_norm = pos_map / torch.sum(pos_map, axis = (-1, -2), keepdim=True)
        dist_scalars = (dist_map * pos_map_norm).sum(axis = (-1, -2))
        return dist_scalars.detach().cpu().numpy()
    
    def predict_dist(self, image):
        outs = self(image)
        return self.predict_dist_from_outs(outs)
    
    def predict_orientation_from_outs(self, outs, return_cos_sin = False):
        pos_map = outs[:, 0, ...]
        cos_map =outs[:, 2, ...]
        sin_map =outs[:, 3, ...]

        pos_map_norm = pos_map / torch.sum(pos_map, axis = (-1, -2), keepdim=True)
        cos_scalars = (cos_map * pos_map_norm).sum(axis = (-1, -2))
        sin_scalars = (sin_map * pos_map_norm).sum(axis = (-1, -2))
        thetas = torch.atan2(sin_scalars, cos_scalars).detach().cpu().numpy()
        if not return_cos_sin:
            return thetas
        else:
            return thetas, cos_scalars.detach().cpu().numpy(), sin_scalars.detach().cpu().numpy()

    def predict_leds_from_outs(self, outs):
        pos_map = outs[:, :1, ...]
        led_maps = outs[:, 4:, ...]
        pos_map_norm = pos_map / torch.sum(pos_map, axis = (-1, -2), keepdim=True)
        masked_maps = pos_map_norm * led_maps
        # breakpoint()
        return masked_maps.sum(axis = [-1, -2]).detach().cpu().numpy()
        # return masked_maps.max(-1)[0].detach().cpu().numpy()


    def predict_orientation(self, image):
        outs = self(image)
        return self.predict_orientation_from_outs(outs)
    
    def predict_leds(self, image):
        outs = self(image)
        return self.predict_leds_from_outs(outs)
    





