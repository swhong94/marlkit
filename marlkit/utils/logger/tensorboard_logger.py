
from torch.utils.tensorboard import SummaryWriter

from typing import Dict, Any 
from .base_logger import BaseLogger 
import os 

class TensorBoardLogger(BaseLogger): 
    def __init__(self, project: str, run_name: str, config: Dict[str, Any]): 
        log_dir = os.path.join("runs", project, run_name)
        super().__init__(project=project, run_name=run_name, config=config)
        self.writer = SummaryWriter(log_dir=log_dir) 
        self.log_config() 

    def log_scalar(self, tag: str, value: float, step: int): 
        self.writer.add_scalar(tag, value, step) 
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int): 
        self.writer.add_scalars(tag, values, step) 

    def log_histogram(self, tag: str, values, step: int): 
        self.writer.add_histogram(tag, values, step) 

    def log_config(self): 
        for k, v in self.config.items(): 
            self.writer.add_text(f"config", f"{k}: {v}")

    def close(self): 
        self.writer.close() 


