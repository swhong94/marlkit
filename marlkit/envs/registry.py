

"""
marlkit/envs/registry.py 

Centralized environment registry

Every env is registered as a factory function with the signature: 

    factory(num_agents, seed, max_cycles, **kwargs) -> (env, obs_dim, action_dim, num_agents_actual) 

The returned `env` must implement: 
    reset(seed) -> (obs_np, critic_obs_np) 
    step(actions) -> (obs_np, critic_obs_np, reward, done, info) 
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple 

ENV_REGISTRY: Dict[str, Callable[..., Tuple]] = {} 

def register_env(name: str, factory_fn: Callable[..., Tuple]) -> None: 
    """Register an env factory under *name*."""
    if name in ENV_REGISTRY: 
        raise ValueError(f"Environment '{name}' is already registered.") 
    ENV_REGISTRY[name] = factory_fn 

def make_env(name: str, **kwargs) -> Tuple: 
    """
    Look up *name* in the registry and call its factory. 
    Returns (env, obs_dim, action_dim, num_agents) 
    """
    if name not in ENV_REGISTRY: 
        raise ValueError(
            f"Unknown env '{name}'. Registered: {sorted(ENV_REGISTRY.keys())}"
        )
    return ENV_REGISTRY[name](**kwargs)

# ------------------------------------------------------------------------
# Convenience: wrap any PettingZoo parallel_env callable 
# ------------------------------------------------------------------------ 

def register_pettingzoo_env(
    name: str, 
    pz_env_fn: Callable[..., Any], 
    team_reward: bool = True, 
)-> None: 
    """
    Register a PettingZoo parallel env so users don't have to manually 
    wrap it with PettingZooParallelAdapter. 
    
    pz_env_fn: a callable that returns a PettingZoo parallel env, 
               e.g., `lambda **kw: my_module.parallel_env(**kw)
    """

    from marlkit.envs.pettingzoo_adapter import PettingZooParallelAdapter, PZAdapterConfig
    
    def _factory(num_agents: int, seed: int, max_cycles: int, **kw): 
        pz_env = pz_env_fn(**kw) 
        adapter_cfg = PZAdapterConfig(team_reward=team_reward) 
        env = PettingZooParallelAdapter(pz_env, cfg=adapter_cfg) 
        env.reset(seed=seed) 
        return env, env.obs_dim, env.action_dim, env.num_agents 

    register_env(name, _factory) 

# ------------------------------------------------------------------------
# Built-in registrations 
# ------------------------------------------------------------------------
def _register_toy_simple_spread(): 
    def _factory(num_agents: int=5, seed: int=0, max_cycles: int=25, **_kw): 
        from marlkit.envs.simple_spread import SimpleSpreadParallelEnv, SimpleSpreadEnvConfig
        env = SimpleSpreadParallelEnv(SimpleSpreadEnvConfig(num_agents=num_agents, episode_len=max_cycles)) 
        env.reset(seed=seed) 
        return env, env.obs_dim, env.action_dim, env.num_agents
    register_env("toy_simple_spread", _factory) 

def _register_mpe_envs(): 
    from marlkit.envs.pettingzoo_adapter import PettingZooParallelAdapter, PZAdapterConfig
    from marlkit.envs.mpe_factory import make_mpe_env, MPEConfig
    from marlkit.envs.wrappers import SuperSuitConfig, apply_supersuit_wrappers

    _MAPPING = {
        "mpe_simple_spread":    "simple_spread_v3", 
        "mpe_simple_reference": "simple_reference_v3", 
        "mpe_simple_crypto":    "simple_crypto_v3", 
        "mpe_simple_world_comm":"simple_world_comm_v3",
        "mpe_simple_push":      "simple_push_v3", 
        "mpe_simple_adversary": "simple_adversary_v3", 
        "mpe_simple_tag":       "simple_tag_v3", 
    }

    for toolkit_name, module_name in _MAPPING.items(): 
        def _factory(num_agents: int=3, 
                     seed: int=0, 
                     max_cycles: int=25, 
                     continuous_actions: bool=False, 
                     local_ratio: float=0.5, 
                     num_good: int=1, 
                     num_adversaries: int=3, 
                     num_obstacles: int=2, 
                     ss_cfg: "Optional[SuperSuitConfig]"=None, 
                     _module=module_name, 
                     **_kw):
            mpe_cfg = MPEConfig(
                max_cycles=max_cycles, 
                continuous_actions=continuous_actions, 
                N=num_agents, 
                local_ratio=local_ratio,
                num_good=num_good, 
                num_adversaries=num_adversaries,
                num_obstacles=num_obstacles
            )
            pz_env = make_mpe_env(module_name=_module, cfg=mpe_cfg) 

            if ss_cfg is not None and ss_cfg.enabled: 
                pz_env = apply_supersuit_wrappers(pz_env, ss_cfg) 
            
            env = PettingZooParallelAdapter(pz_env, cfg=PZAdapterConfig(team_reward=True))
            env.reset(seed=seed) 
            return env, env.obs_dim, env.action_dim, env.num_agents 
        
        register_env(toolkit_name, _factory) 

# ------------------------------------------------------------------------
# Heterogeneous env (custom, no PettingZoo dependency)
# ------------------------------------------------------------------------
def _register_simple_hetero():
    def _factory(
        num_agents: int = 4,   # total (scouts + workers); split evenly by default
        seed: int = 0,
        max_cycles: int = 25,
        num_scouts: "Optional[int]" = None,
        num_workers: "Optional[int]" = None,
        **_kw,
    ):
        from marlkit.envs.simple_hetero import SimpleHeteroEnv, SimpleHeteroConfig

        # If explicit scout/worker counts given, use them; else split evenly
        if num_scouts is None and num_workers is None:
            num_scouts = num_agents // 2
            num_workers = num_agents - num_scouts
        elif num_scouts is None:
            num_scouts = max(1, num_agents - num_workers)
        elif num_workers is None:
            num_workers = max(1, num_agents - num_scouts)

        cfg = SimpleHeteroConfig(
            num_scouts=num_scouts,
            num_workers=num_workers,
            episode_len=max_cycles,
        )
        env = SimpleHeteroEnv(cfg)
        env.reset(seed=seed)
        return env, env.obs_dim, env.action_dim, env.num_agents

    register_env("simple_hetero", _factory)


def _register_simple_hetero_pz():
    """Register the PZ-compatible version (requires SuperSuit for padding)."""
    from marlkit.envs.wrappers import SuperSuitConfig, apply_supersuit_wrappers
    
    def _factory(
        num_agents: int = 4,
        seed: int = 0,
        max_cycles: int = 25,
        num_scouts: "Optional[int]" = None,
        num_workers: "Optional[int]" = None,
        ss_cfg: "Optional[SuperSuitConfig]" = None,
        **_kw,
    ):
        from marlkit.envs.simple_hetero import SimpleHeteroPZEnv, SimpleHeteroConfig
        from marlkit.envs.pettingzoo_adapter import PettingZooParallelAdapter, PZAdapterConfig
        from marlkit.envs.wrappers import SuperSuitConfig, apply_supersuit_wrappers

        if num_scouts is None and num_workers is None:
            num_scouts = num_agents // 2
            num_workers = num_agents - num_scouts
        elif num_scouts is None:
            num_scouts = max(1, num_agents - num_workers)
        elif num_workers is None:
            num_workers = max(1, num_agents - num_scouts)

        cfg = SimpleHeteroConfig(
            num_scouts=num_scouts,
            num_workers=num_workers,
            episode_len=max_cycles,
        )
        pz_env = SimpleHeteroPZEnv(cfg)

        # SuperSuit padding makes the heterogeneous spaces uniform
        # so the PettingZooParallelAdapter (which assumes homogeneous) works
        if ss_cfg is None:
            ss_cfg = SuperSuitConfig(
                enabled=True,
                pad_observations=True,
                pad_action_space=True,
            )
        if ss_cfg.enabled:
            pz_env = apply_supersuit_wrappers(pz_env, ss_cfg)

        env = PettingZooParallelAdapter(pz_env, cfg=PZAdapterConfig(team_reward=True))
        env.reset(seed=seed)
        return env, env.obs_dim, env.action_dim, env.num_agents

    register_env("simple_hetero_pz", _factory)


# ----------------------------------------------------------------
# Cooperative homogeneous foraging env 
# ---------------------------------------------------------------- 
def _register_simple_foraging(): 
    def _factory(num_agents: int = 3, seed: int = 0, max_cycles: int=50, **_kw): 
        from marlkit.envs.simple_foraging import SimpleForagingConfig, SimpleForagingEnv
        cfg = SimpleForagingConfig(num_agents=num_agents, episode_len=max_cycles) 
        env = SimpleForagingEnv(cfg) 
        env.reset(seed=seed) 
        return env, env.obs_dim, env.action_dim, env.num_agents 
    register_env("simple_foraging", _factory) 


def _register_simple_foraging_pz(): 
    def _factory(num_agents: int=3, seed: int=0, max_cycles: int=50, **_kw): 
        from marlkit.envs.simple_foraging import SimpleForagingPZEnv, SimpleForagingConfig 
        from marlkit.envs.pettingzoo_adapter import PettingZooParallelAdapter, PZAdapterConfig
        pz_env = SimpleForagingPZEnv(SimpleForagingConfig(num_agents=num_agents, episode_len=max_cycles))
        env = PettingZooParallelAdapter(pz_env, cfg=PZAdapterConfig(team_reward=True)) 
        env.reset(seed=seed) 
        return env, env.obs_dim, env.action_dim, env.num_agents 
    register_env("simple_foraging_pz", _factory) 



# Run registrations on import 
_register_toy_simple_spread() 
_register_mpe_envs() 
_register_simple_hetero()
_register_simple_hetero_pz()
_register_simple_foraging() 
_register_simple_foraging_pz() 
