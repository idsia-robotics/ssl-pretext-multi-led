from typing import Any
import torch

class BaseModel(torch.nn.Module):

    def __init__(self, task, checkpoint_file = None) -> None:
        super().__init__()
        self.task = task
        self.checkpoint_file = checkpoint_file
        self.layers = None
    
    def loss(self, y_true, y_pred):
        raise NotImplementedError()
    
    def forward(self, x):
        return self.layers(x)
    
    def optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)


model_registry = {}


def get_model(model_name) -> BaseModel:
    cls = model_registry.get(model_name)
    if not cls:
        raise ValueError(f"Model with name {model_name} not found")
    else:
        return cls


def ModelRegistry(name):

    def wrapper(cls):
        """
        Decorator
        """
        if name not in model_registry:
            model_registry[name] = cls
        else:
            raise ValueError(f"Model with name {name} already exists in registry: {model_registry[name]}")
        return cls
    return wrapper


import src.models.fcn
