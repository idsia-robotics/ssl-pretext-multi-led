from pathlib import Path
from typing import Any
import torch
from mlflow import get_experiment_by_name, search_runs, log_artifact, get_run
from mlflow.artifacts import download_artifacts

class BaseModel(torch.nn.Module):

    def __init__(self, task , checkpoint_file = None, **kwargs) -> None:
        super().__init__()
        self.task = task
        self.checkpoint_file = checkpoint_file
        self.layers = None
        self.epsilon = torch.tensor([1e-10])
        self.MAX_DIST_M = 5.
    
    def loss(self, y_true, y_pred):
        raise NotImplementedError()
    
    def forward(self, x):
        return self.layers(x)
    
    def optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def save_checkpoint(self, path, **kwargs):
        torch.save({
            'model_state_dict' : self.state_dict(),
            'model_name' : self.model_name,
            **kwargs
        },
        path)

    def log_checkpoint(self, checkpoint_id, **kwargs):
        path = Path(f"/tmp/checkpoint_{checkpoint_id}.tar")
        self.save_checkpoint(str(path), **kwargs)
        log_artifact(path)
        path.unlink()

    def load_from_checkpoint(self, data):
        self.load_state_dict(data["model_state_dict"])
        return data
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    def to(self, *args, **kwargs):
        self.epsilon = self.epsilon.to(*args, **kwargs)
        return super().to(*args, **kwargs)



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
            cls.model_name = name
        else:
            raise ValueError(f"Model with name {name} already exists in registry: {model_registry[name]}")
        return cls
    return wrapper


import src.models.fcn
import src.models.multi_scale_fcn
import src.models.mobile_net
import src.models.resnet
import src.models.clip

def load_model_mlflow(mlflow_run_name, experiment_id, checkpoint_idx, model_kwargs, return_run_id = False,
                      return_run_params = False):
    result = []
    runs = search_runs([experiment_id], filter_string=f"params.run_name = '{mlflow_run_name}'")
    if len(runs) == 1:
        run = runs.iloc[0]
        run_id = run['run_id']
        # artifact_uri = f"runs:/{run_id}/checkpoint_{checkpoint_idx}.tar"
        checkpoint_path = download_artifacts(run_id=run_id,
                                             artifact_path=f"checkpoint_{checkpoint_idx}.tar",
                                             dst_path="/tmp/.checkpoints")
            
        checkpoint_path = Path(checkpoint_path)
        if not torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)
        model = get_model(checkpoint["model_name"])(**model_kwargs)
        model.load_from_checkpoint(checkpoint)
        result.append(model)
        if return_run_id:
            result.append(run_id)
        if return_run_params:
            mlflow_run = get_run(run_id=run_id)
            result.append(mlflow_run.data.params)
        return result
    elif len(runs) == 0:
        raise ValueError(f"Could not find run with experiment id {experiment_id} and name {mlflow_run_name}")
    else:
        raise NotImplemented("Not handling case where multiple runs have the same name yet")


def load_model_raw(checkpoint_path, model_kwargs):
    if not torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    model = get_model(checkpoint["model_name"])(**model_kwargs)
    model.load_from_checkpoint(checkpoint)
    return model
