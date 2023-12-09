import torch
from src.models import BaseModel, ModelRegistry
from torch.nn.functional import binary_cross_entropy_with_logits as bce

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

        if self.task == 'presence':
            self.layers = torch.nn.Sequential(
                self.core_layers,
                self.robot_presence_layer
            )
            self.loss = self.__robot_presence_loss
        
    
    def __robot_presence_loss(self, y_true, y_pred):
        return bce(y_pred.squeeze(), y_true.squeeze())





