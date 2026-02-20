from dataclasses import dataclass 

@dataclass 
class SuperSuitConfig: 
    enabled: bool = True 
    pad_observations: bool = True 
    pad_action_space: bool = True 
    normalize_obs: bool = False 
    frame_stack: int = 1    # 1 disables frame stacking 


def apply_supersuit_wrappers(pz_env, cfg: SuperSuitConfig): 
    """ 
    Apply SuperSuit wrappers to a PettingZoo environment 
    
    Notes: 
        - pad_observations_v0, pad_action_space_v0 help parameter sharing when agents have heterogeneous shapes 
        - normalize_obs_v0 helps training with stability 
        - frame_stack_v1 increases obs dimension by K factor (good for POMDP even before Recurrent Networks)
    """
    if not cfg.enabled: 
        return pz_env 
    
    import supersuit as ss 

    if cfg.pad_observations: 
        pz_env = ss.pad_observations_v0(pz_env) 
    
    if cfg.pad_action_space:
        pz_env = ss.pad_action_space_v0(pz_env) 

    if cfg.normalize_obs: 
        pz_env = ss.normalize_obs_v0(pz_env) 
    
    if cfg.frame_stack is not None and int(cfg.frame_stack) > 1: 
        pz_env = ss.frame_stack_v1(pz_env, cfg.frame_stack) 

    return pz_env 



