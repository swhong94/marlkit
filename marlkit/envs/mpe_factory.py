
from __future__ import annotations 

from dataclasses import dataclass
from typing import Any, Dict, Optional 

@dataclass 
class MPEConfig: 
    # common 
    max_cycles: int = 25 
    continuous_actions: bool = False 

    # spread/adversary (uses N) 
    N: int = 3 
    local_ratio: float = 0.5 

    # tag 
    num_good: int = 1 
    num_adversaries: int = 3 
    num_obstacles: int = 2 

SUPPORTED = {
    "simple_spread_v3", 
    "simple_reference_v3", 
    "simple_crypto_v3", 
    "simple_world_comm_v3", 
    "simple_push_v3", 
    "simple_adversary_v3", 
    "simple_tag_v3", 
}

def _import_mpe_module(module_name: str): 
    # Pettingzoo legacy path first, then mpe2 fallback 
    try: 
        return __import__(f"pettingzoo.mpe.{module_name}", fromlist=[module_name])
    except Exception: 
        return __import__(f"mpe2.envs.{module_name}", fromlist=[module_name])
    

def make_mpe_env(module_name: str, cfg: Optional[MPEConfig]=None, **overrides): 
    """
    module_name: e.g., "simple_spread_v3" 
    Returns: PettingZoo parallel env 
    """
    if module_name not in SUPPORTED: 
        raise ValueError(f"Unsupported MPE module '{module_name}'. SUPPORTED: {sorted(SUPPORTED)}")

    cfg = cfg or MPEConfig() 
    for k, v in overrides.items(): 
        setattr(cfg, k, v) 
    
    mod = _import_mpe_module(module_name) 

    if module_name in {"simple_spread_v3", "simple_adversary_v3"}: 
        # Spread/Adversary uses N plus local_ratio (spread uses it; adversary ignores it) 
        kwargs : Dict[str, Any] = dict(
            N = int(cfg.N), 
            max_cycles = int(cfg.max_cycles), 
            continuous_actions = bool(cfg.continuous_actions), 
        )
        # Simple Spread uses local ratio; safe to pass only when spread 
        if module_name == "simple_spread_v3": 
            kwargs["local_ratio"] = float(cfg.local_ratio) 
        
        return mod.parallel_env(**kwargs) 
    
    if module_name == "simple_reference_v3": 
        # reference uses local ratio, not N 
        return mod.parallel_env(
            local_ratio = float(cfg.local_ratio), 
            max_cycles = int(cfg.max_cycles), 
            continuous_actions = bool(cfg.continuous_actions),
        )
    
    if module_name in {"simple_crypto_v3", "simple_world_comm_v3", "simple_push_v3"}: 
        # These environments accept max_cycles + continuous_actions 
        return mod.parallel_env(
            max_cycles = int(cfg.max_cycles), 
            continuous_actions = bool(cfg.continuous_actions),
        )
    
    if module_name == "simple_tag_v3": 
        # Tag uses num_good, num_adversaries, num_obstacles 
        return mod.parallel_env(
            num_good = int(cfg.num_good), 
            num_adversaries = int(cfg.num_adversaries), 
            num_obstacles = int(cfg.num_obstacles), 
            max_cycles = int(cfg.max_cycles), 
            continuous_actions = bool(cfg.continuous_actions),
        )
    
    raise ValueError(f"Unsupported env module: {module_name}")
    