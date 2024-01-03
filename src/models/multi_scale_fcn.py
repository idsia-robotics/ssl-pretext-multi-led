import torch
from src.models import ModelRegistry
from src.models.fcn import Model_s

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels = 64) -> None:
        super().__init__()
        assert in_channels // 8 > 0 and in_channels % 8 == 0

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            torch.nn.BatchNorm2d(in_channels // 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=1),
            torch.nn.BatchNorm2d(in_channels // 8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            torch.nn.BatchNorm2d(1),
            
        )
    def forward(self, x):
        return self.layers(x)
    
@ModelRegistry("multiscale_model_s")
class MS_Model_s(Model_s):
    
    def __init__(self, *args, **kwargs) -> None:
        super(MS_Model_s, self).__init__(*args, **kwargs)
        self.downsample_layer = torch.nn.AvgPool2d(2)
        self.upsample_layer = torch.nn.Upsample((45, 80), mode='bilinear')

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
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=2, stride=1, dilation = 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=2, stride=1, dilation=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

        )

        self.robot_pose_and_led_layer = torch.nn.ModuleList(
        #     ConvBlock(64) for _ in range(10)
        # ])
        # (
           [     torch.nn.Conv2d(64, 10, kernel_size=1, padding=0, stride=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(10),
                torch.nn.Conv2d(10, 10, kernel_size=1, padding=0, stride=1),
                # torch.nn.Sigmoid()
            ])
        self.forward = self.forward_fn

    def forward_fn(self, x):
        downsampled = self.downsample_layer(x)
        out = self.core_layers(x)
        downsampled_out = self.core_layers(downsampled)
        downsampled_out = self.upsample_layer(downsampled_out)
        concat = torch.cat([out, downsampled_out], dim = 1)
        out = torch.cat([l(concat) for l in self.robot_pose_and_led_layer], dim = 1)
        out = torch.cat(
            [
                torch.nn.functional.sigmoid(out[:, :2, ...]), # pos and dist
                torch.nn.functional.tanh(out[:, 2:4, ...]), # orientation
                torch.nn.functional.sigmoid(out[:, 4:, ...]), # leds
                
            ],
            axis = 1)
        out[:, 1, ...] = out[:, 1, ...] * self.MAX_DIST_M
        return out

    

