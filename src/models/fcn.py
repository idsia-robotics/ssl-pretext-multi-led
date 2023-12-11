import torch
from src.models import BaseModel, ModelRegistry
from torch.nn.functional import binary_cross_entropy as bce
from torch.nn.functional import mse_loss
from torchvision.transforms.functional import resize
from numpy import unravel_index, stack
import numpy as np

@ModelRegistry("model_s")
class Model_s(BaseModel):
    def __init__(self, *args, **kwargs):
        super(Model_s, self).__init__(*args, **kwargs)

        self.core_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.Conv2d(6, 12, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(4),
            torch.nn.Conv2d(12, 24, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(24, 48, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(48, 48, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 3, kernel_size=1, padding=0, stride=1),
        )

        self.robot_presence_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(256),
            torch.nn.LazyLinear(1),
            torch.nn.Sigmoid()
        )

        self.robot_position_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 1, kernel_size=1, padding=0, stride=1),
            torch.nn.Sigmoid()
        )


        if self.task == 'presence':
            self.layers = torch.nn.Sequential(
                self.core_layers,
                self.robot_presence_layer
            )
            self.loss = self.__robot_presence_loss
        if self.task == 'position':
            self.layers = torch.nn.Sequential(
                self.core_layers,
                self.robot_position_layer
            )
            
            self.loss = self.__robot_position_loss
        
    
    def __robot_presence_loss(self, batch, y_pred):
        y_pred_probs = torch.zeros((y_pred.shape[0], 2))
        y_pred_probs[:, 1] = y_pred[:, 0]
        y_pred_probs[:, 0] = 1 - y_pred[:, 0]

        y_true_classes = torch.nn.functional.one_hot(batch['robot_visible'].long(), 2).float()
        return bce(y_pred_probs, y_true_classes)

    def __robot_position_loss(self, batch, model_out):
        pos_true = batch['pos_map'][:, None, ...]
        pos_true = resize(pos_true, model_out.shape[-2:], antialias=False).float()
        return mse_loss(model_out, pos_true)
    
    def predict_pos(self, image):
        outs = self(image)
        outs = torch.zeros_like(outs)
        map_shape = outs.shape[-2:]
        outs[..., 20, 20] = 1
        outs = outs.view(outs.shape[0], -1)
        max_idx = outs.argmax(1).cpu()
        indexes = unravel_index(max_idx, map_shape)
        #               x               y
        indexes = stack([indexes[1], indexes[0]]).T.astype('float32')
        indexes /= np.array([map_shape[1], map_shape[0]])
        indexes *= np.array([image.shape[-1], image.shape[-2]])
        return indexes.astype(np.int32)
    





