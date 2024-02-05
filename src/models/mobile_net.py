from torch import Tensor
from torchvision.models import MobileNetV2
import torch
from src.models import ModelRegistry, BaseModel
from src.models.fcn import FullyConvPredictorMixin
import torch.nn as nn



def dwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            #depthwise
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU6(inplace=True),
        )
    )

def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

def conv1x1_raw(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
        )
    )

def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend([
            #dw
            dwise_conv(hidden_dim, stride=stride),
            #pw
            conv1x1(hidden_dim, ch_out)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)

@ModelRegistry("mobile_net")
class MobileNetV2(FullyConvPredictorMixin, BaseModel):
    def __init__(self, **kwargs):

        super(MobileNetV2, self).__init__(**kwargs)

        self.configs=[
            # t, c, n, s
            [1, 32, 1, 2],
            [6, 64, 2, 2],
            [6, 128, 2, 1],
        ]

        self.stem_conv = conv3x3(3, 32, stride=2)

        layers = []
        input_channel = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c

        self.layers = nn.Sequential(*layers)

        self.last_conv = conv1x1_raw(input_channel, 10)

        # self.last_layer = torch.nn.Conv2d(1280, 10, kernel_size=1, padding=0, stride=1)
        # self.last_layer = torch.nn.Conv2d(1280, 10, kernel_size=3, padding=3, stride=1, dilation=3)

        self.MAX_DIST_M = 5.
#        self.avg_pool2d = nn.AvgPool2d(4)
#        self.upscaler = nn.ConvTranspose2d(10, 10, 2, stride=2)
        self.loss = self._robot_pose_and_leds_loss

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        return self.pose_and_leds_non_lin(x)

    def pose_and_leds_non_lin(self, x):
        out = torch.cat(
            [
                torch.nn.functional.sigmoid(x[:, :2, ...]), # pos and dist
                torch.nn.functional.tanh(x[:, 2:4, ...]), # orientation
                torch.nn.functional.sigmoid(x[:, 4:, ...]), # leds
                
            ],
            axis = 1)
        out[:, 1, ...] = out[:, 1, ...] * self.MAX_DIST_M
        return out


    def _robot_pose_and_leds_loss(self, batch, model_out):
        supervised_label = batch["supervised_flag"].to(model_out.device)

        proj_loss, proj_map_norm = self._robot_projection_loss(batch, model_out, return_norm=True)
        dist_loss = self._robot_distance_loss(batch, model_out, proj_map_norm)
        
        orientation_loss = self._robot_orientation_loss(batch, model_out, proj_map_norm)
        led_loss, led_losses = self._robot_led_loss(batch, model_out, proj_map_norm)

        unsupervised_label = ~supervised_label
        
        led_loss = led_loss.mean(-1) * unsupervised_label

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
        downscaled_gt_proj_norm = downscaled_gt_proj / (downscaled_gt_proj.sum(axis = (-1, -2), keepdims = True) + 1e-6)
        proj_pred_norm = proj_pred / (proj_pred + self.epsilon).sum(axis=(-1, -2), keepdims=True)
        loss = 1 - (proj_pred_norm * downscaled_gt_proj_norm).sum(axis = (-1, -2))
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
        return losses, losses.detach().mean(0)        
