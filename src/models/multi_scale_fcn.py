from typing import Any
import numpy as np
import scipy
import torch
from src.models import ModelRegistry
from src.models.fcn import Model_s
from torch.nn import functional as F

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
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1),
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
        # out = torch.cat([l(concat) for l in self.robot_pose_and_led_layer], dim = 1)
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

    
class GaussianLayer(torch.nn.Module):
    def __init__(self, kernel_size = 3):
        super(GaussianLayer, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(kernel_size // 2), 
            torch.nn.Conv2d(3, 3, kernel_size=kernel_size, stride=1, padding=0, bias=None, groups=3)
        )

        self.kernel_size = kernel_size
        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((self.kernel_size,self.kernel_size))
        n[self.kernel_size // 2,self.kernel_size // 2] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=self.kernel_size // 5)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))

class LaplacianPyramidBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = (3,3), padding = 1, stride = 1,
                 activation_block = torch.nn.ReLU, blur_kernel_size = 3) -> None:
        
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
        )

        self.act_block = activation_block() if activation_block else lambda x: x
        
        if stride == 1:
            self.forward = self._no_skip_impl
        else:
            self.kernel = GaussianLayer(blur_kernel_size)
            self.forward = self._skip_impl

        self.downscaler = torch.nn.AvgPool2d(stride, stride)

    def _no_skip_impl(self, x, image):
        features = self.layers(x)
        if image is None:
            image = x
        return self.act_block(features), self.downscaler(image).detach()

    def _skip_impl(self, x, image):
        breakpoint()
        features = self.layers(x)
        if image is None:
            image = x
        downscaled = self.downscaler(self.kernel(image)).detach()
        return self.act_block(features + downscaled), downscaled

    # def make_gaussian_kernel(self, kernel_size, sigma):
    #     ts = torch.linspace(-kernel_size // 2, kernel_size // 2 + 1, kernel_size)
    #     gauss = torch.exp((-(ts / sigma)**2 / 2))
    #     kernel = gauss / gauss.sum()
    #     return kernel
    
    # def fast_gaussian_blur(self, img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    #     trailing_dims = img.shape[:-3]
    #     kernel_size = kernel.shape[0]

    #     padding = (left, right, top, bottom)
    #     padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
    #     img = F.pad(img, padding, mode="constant", value=0)

    #     # Separable 2d conv
    #     breakpoint()
    #     kernel = kernel.view(*trailing_dims, 1, kernel_size, 1)
    #     img = F.conv1d(img, kernel)
    #     kernel = kernel.view(*trailing_dims, 1, 1, kernel_size)
    #     img = F.conv1d(img, kernel)

    #     return img

    def __call__(self, *args) -> Any:
        if len(args) > 1:
            return self.forward(args[0], args[1])
        else:
            return self.forward(args[0], args[0])


    
# @ModelRegistry("laplacian_ms")
class Laplacian_ms(Model_s):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.layers = torch.nn.Sequential(
            LaplacianPyramidBlock(in_channels=3, out_channels=8, kernel_size=7, padding=3, stride=1, blur_kernel_size=7),
            LaplacianPyramidBlock(in_channels=16, out_channels=16, kernel_size=7, padding=3, stride=2, blur_kernel_size=7),
            LaplacianPyramidBlock(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=1, blur_kernel_size=5),
            LaplacianPyramidBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2, blur_kernel_size=3),
            LaplacianPyramidBlock(in_channels=32, out_channels=64, kernel_size=3, padding=0, stride=2, blur_kernel_size=3),
            LaplacianPyramidBlock(in_channels=64, out_channels=10, kernel_size=1, padding=0, stride=1, blur_kernel_size=1),
        )


