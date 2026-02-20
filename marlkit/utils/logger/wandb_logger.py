import wandb
from typing import Dict, Any 
from .base_logger import BaseLogger 

class WandBLogger(BaseLogger): 

    def __init__(self, project: str, run_name: str, config: Dict[str, Any]): 
        super().__init__(project, run_name, config) 
        wandb.init(
            project=project, 
            name=run_name, 
            config=config 
        )

    def log_scalar(self, tag: str, value: float, step: int): 
        wandb.log({tag: value}, step=step) 
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int): 
        wandb.log({f"{tag}/{k}": v for k, v in values.items()}, step=step)

    def log_histogram(self, tag: str, values, step: int): 
        wandb.log({tag: wandb.Histogram(values)}, step=step) 

    def log_config(self): 
        pass  # Already handled by wandb.init(config=...)

    def close(self): 
        wandb.finish() 

