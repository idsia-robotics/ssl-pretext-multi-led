from torch import Tensor
from torchvision.models import MobileNetV2
import torch
from src.models import ModelRegistry, BaseModel
from src.models.fcn import FullyConvPredictorMixin, FullyConvLossesMixin
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
class MobileNetV2(BaseModel, FullyConvPredictorMixin, FullyConvLossesMixin):
    def __init__(self, task):
        super(MobileNetV2, self).__init__(task)

        self.configs=[
            # t, c, n, s
            [1, 32, 1, 1],
            [6, 64, 2, 2],
            [6, 128, 3, 2],
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

        self.last_conv = conv1x1(input_channel, 10)

        # self.last_layer = torch.nn.Conv2d(1280, 10, kernel_size=1, padding=0, stride=1)
        # self.last_layer = torch.nn.Conv2d(1280, 10, kernel_size=3, padding=3, stride=1, dilation=3)

        self.MAX_DIST_M = 5.
        self.loss = self._robot_pose_and_leds_loss
        self.avg_pool2d = nn.AvgPool2d(2)
        # self.upscaler = nn.ConvTranspose2d(10, 10, 4)
        # self.upscaler = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x = self.avg_pool2d(x)
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        # x = self.last_layer(x)
        # x = self.upscaler(x)
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
