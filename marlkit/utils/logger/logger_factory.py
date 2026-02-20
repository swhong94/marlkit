from typing import Dict, Any 

from .tensorboard_logger import TensorBoardLogger 
from .wandb_logger import WandBLogger 

def make_logger(logger_type: str, 
                project: str, 
                run_name: str, 
                config: Dict[str, Any]): 
    logger_type = logger_type.lower() 

    if logger_type == 'tensorboard': 
        return TensorBoardLogger(project, run_name, config) 
    elif logger_type == 'wandb': 
        return WandBLogger(project, run_name, config) 
    else: 
        raise ValueError(f"Unknown logger_type: {logger_type}")
    
    