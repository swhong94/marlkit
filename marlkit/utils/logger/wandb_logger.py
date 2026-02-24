import wandb
from typing import Dict, Any 
from .base_logger import BaseLogger 

class WandBLogger(BaseLogger):
    
    def __init__(self, 
                 project: str, 
                 run_name: str, 
                 config: Dict[str, Any]): 
        super().__init__(project, run_name, config) 
        wandb.init(
            project=project, 
            name=run_name, 
            config=config, 
        )
        self._step_buffer: Dict[str, Any] = {} 
        self._current_step: int | None = None 
    
    def _flush(self): 
        """Send accumulated metrics for the current step."""
        if self._step_buffer and self._current_step is not None: 
            wandb.log(self._step_buffer, step=self._current_step) 
            self._step_buffer = {} 
    
    def log_scalar(self, tag: str, value: float, step: int): 
        if self._current_step is not None and step != self._current_step:
            self._flush() 
        self._current_step = step 
        self._step_buffer[tag] = value 
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int): 
        if self._current_step is not None and step != self._current_step: 
            self._flush() 
        self._current_step = step 
        self._step_buffer.update({f"{tag}/{k}": v for k, v in values.items()}) 

    def log_histogram(self, tag: str, values, step: int): 
        if self._current_step is not None and step != self._current_step: 
            self._flush() 
        self._current_step = step 
        self._step_buffer[tag] = wandb.Histogram(values) 
    
    def log_config(self): 
        pass 

    def close(self): 
        self._flush() 
        wandb.finish() 
        

