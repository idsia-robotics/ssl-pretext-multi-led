import torch
from src.models import ModelRegistry
from src.models.fcn import Model_s

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

            # Let's go deeper
            torch.nn.Conv2d(32, 32, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

        )

        self.robot_pose_and_led_layer = torch.nn.Sequential(
                torch.nn.Conv2d(64, 10, kernel_size=1, padding=0, stride=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(10),
                torch.nn.Conv2d(10, 10, kernel_size=1, padding=0, stride=1),
                # torch.nn.Sigmoid()
            )
        self.forward = self.forward_fn

    def forward_fn(self, x):
        downsampled = self.downsample_layer(x)
        out = self.core_layers(x)
        downsampled_out = self.core_layers(downsampled)
        downsampled_out = self.upsample_layer(downsampled_out)
        concat = torch.cat([out, downsampled_out], dim = 1)
        out = self.robot_pose_and_led_layer(concat)
        out = torch.cat(
            [
                torch.nn.functional.sigmoid(out[:, :2, ...]), # pos and dist
                torch.nn.functional.tanh(out[:, 2:4, ...]), # orientation
                torch.nn.functional.sigmoid(out[:, 4:, ...]), # leds
                
            ],
            axis = 1)
        out[:, 1, ...] = out[:, 1, ...] * self.MAX_DIST_M
        return out

    


