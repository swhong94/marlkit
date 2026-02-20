from abc import ABC, abstractmethod 
from typing import Any, Dict, Tuple 


class BaseLogger(ABC): 
    """
    Abstract base logger for MARL experiments. 
    """
    def __init__(self, project: str, run_name: str, config: Dict[str, Any]):
        self.project = project 
        self.run_name = run_name 
        self.config = config 

    @abstractmethod 
    def log_scalar(self, tag: str, value: float, step: int): 
        pass 

    @abstractmethod
    def log_scalars(self, tag: str, values: Dict[str, float], step: int): 
        pass 

    @abstractmethod 
    def log_histogram(self, tag: str, values, step: int): 
        pass 

    @abstractmethod 
    def log_config(self): 
        pass 

    @abstractmethod
    def close(self): 
        pass 

