import os
import torch
from .interface import ExperimentLogger

class WandbLogger(ExperimentLogger):
    """Save experiment logs to wandb
    """
    def __init__(self, path, name, group, project, entity, tags):
        import wandb
        from wandb import AlertLevel
        self.wandb = wandb
        self.run = wandb.init(
                group=group,
                project=project,
                entity=entity,
                tags=tags
            )
        wandb.run.name = name

    def log_metric(self, name, value, step=None):
        if step is not None:
            self.wandb.log({
                name: value,
                'step': step})
        else:
            self.wandb.log({name: value})

    def log_text(self, name, text):
        pass

    def log_image(self, name, image):
        self.wandb.log({name: [self.wandb.Image(image)]})

    def log_parameter(self, dictionary):
        self.wandb.config.update(dictionary)

    def save_model(self, name, state_dict):
        None

    def finish(self):
        self.wandb.finish()