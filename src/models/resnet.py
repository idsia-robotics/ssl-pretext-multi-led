import numpy as np
from numpy import unravel_index, stack

import torch
from torchvision.models import MobileNetV2
import torch
from src.models import ModelRegistry, BaseModel
from src.models.fcn import FullyConvPredictorMixin
import torch.nn as nn
from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, RawScoresOutputTarget
from src.models import ModelRegistry, BaseModel

@ModelRegistry("cam")
class ResnetCAMWrapper(BaseModel):
    

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop('led_inference')
        # kwargs.pop('task')

        super().__init__(*args, **kwargs)
        self.configs=[
            # t, c, n, s
            [1, 32, 1, 2],
            [6, 64, 2, 2],
            [6, 128, 2, 1],
        ]
        self.model = MobileNetV2(num_classes=6, inverted_residual_setting=self.configs)
        target_layers = [self.model.features[-1][0]]
        self.cam = AblationCAM(self.model, target_layers=target_layers)

    
    def forward(self, x):
        return torch.nn.functional.sigmoid(self.model(x))
    
    def loss(self, batch, model_out):
        led_preds = model_out
        led_labels = batch['led_mask'].to(model_out.device)
        losses = torch.zeros_like(led_labels, device = model_out.device, dtype=torch.float32)
        for i in range(led_labels.shape[1]):
            losses[:, i] = torch.nn.functional.binary_cross_entropy(
                led_preds[:, i], led_labels[:, i].float(), reduction='none'
            )
        # We only care about 4 leds
        losses[:, 1] = 0.
        losses[:, 2] = 0.
        return losses.mean() * 1.5, losses.detach().mean(0)
    
    def predict_leds(self, x):
        out = self(x)
        return out
    
    def predict_pos(self, images):
        led_ids = [0, 3, 4, 5]
        coords = np.zeros((images.shape[0], 4, 2))


        for image_idx in range(images.shape[0]):
            x = images[image_idx, ...][None, ...]
            for i, l in enumerate(led_ids):
                maps = self.cam(input_tensor=x, targets=[ClassifierOutputTarget(l)])
                out_map_shape = maps.shape[-2:]
                maps = maps.reshape((*maps.shape[:-2], -1))
                max_idx = maps.argmax(1)
                indexes = unravel_index(max_idx, out_map_shape)
                #               x               y
                indexes = stack([indexes[1], indexes[0]]).T.astype('float32')
                indexes /= np.array([out_map_shape[1], out_map_shape[0]])
                indexes *= np.array([x.shape[-1], x.shape[-2]])

                y_scale_f = x.shape[0] / out_map_shape[0]
                x_scale_f = x.shape[1] / out_map_shape[1]
                indexes += np.array([x_scale_f, y_scale_f]) / 2

                coords[:, i, :] = indexes
        return coords.mean(1)

    def optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)            





    


