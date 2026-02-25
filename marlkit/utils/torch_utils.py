import random 
import numpy as np 
import torch 

def set_seed(seed: int=0) -> None: 
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 

def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float: 
    # 1 - Var[y_true - y_pred] / Var[y_true]
    var_y = torch.var(y_true) 
    if var_y.item() < 1e-12: 
        return float('nan') 
    return (1.0 - torch.var(y_true - y_pred) / var_y).item() 




