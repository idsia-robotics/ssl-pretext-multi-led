from torch import Tensor
from torchvision.models import MobileNetV2
import torch
from src.models import ModelRegistry, BaseModel
from src.models.fcn import FullyConvPredictorMixin, FullyConvLossesMixin


@ModelRegistry("mobile_net")
class NN_ci(BaseModel, MobileNetV2, FullyConvPredictorMixin, FullyConvLossesMixin):

    def __init__(self, *args, **kwargs):
        settings = [
            # t, c, n, s
            [1, 24, 1, 1],
            [6, 52, 2, 2],
            [6, 86, 3, 2],
            # [6, 128, 2, 2],
            # [6, 256, 1, 1],
        ]

        BaseModel.__init__(self, *args, **kwargs)

        MobileNetV2.__init__(
            self,
            1,
            round_nearest=8,
            inverted_residual_setting=settings
        )
        
        self.new_features = torch.nn.Sequential(*(list(self.features.children())[:-1]))
        self.new_features = torch.nn.Sequential(self.new_features,
                                                list(self.features.children())[-1][0],
                                                list(self.features.children())[-1][1])
        self.features = self.new_features
        self.last_layer = torch.nn.Conv2d(1280, 10, kernel_size=1, padding=0, stride=1)
        # self.last_layer = torch.nn.Conv2d(1280, 10, kernel_size=3, padding=3, stride=1, dilation=3)

        self.MAX_DIST_M = 5.
        self.loss = self._robot_pose_and_leds_loss




    def _forward_impl(self, x: Tensor) -> Tensor:
        feat = self.features(x)
        out = self.last_layer(feat)
        return self.pose_and_leds_non_lin(out)

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

    def forward(self, x):
        return self._forward_impl(x)