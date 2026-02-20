from __future__ import annotations 

from dataclasses import dataclass 
from typing import Any, Dict, List, Tuple, Optional

import numpy as np 


@dataclass 
class PZAdapterConfig: 
    """v0 assumptions: parallel env, fixed agents, homogeneous obs/action shapes"""
    team_reward: bool = True        # If True, use sum rewards into scalar team reward 
    strict_agents: bool = True      # If True, error on changes of agent set 
    use_action_mask: bool = True    # If True, infos contain action masks, optionally enforced 



class PettingZooParallelAdapter: 
    """ 
    Wrap a PettingZoo environment so that it matches our environment settings. 
        reset() -> (obs_np, critic_obs_np)
        step(action_np) -> (obs_np, critic_obs_np, reward(float), done(bool), info(dict)) 
    
    We keep: 
        obs_np_shape: (N, obs_dim) 
        actions_np_shape: (N, ) 
        critic_obs_np: concat(obs_np) shape: (N * obs_dim, )
    """
    def __init__(self, pz_parallel_env, cfg: PZAdapterConfig = PZAdapterConfig()): 
        self.env = pz_parallel_env 
        self.cfg = cfg 

        # agent ordering: fixed one at reset 
        self.agents: List[str] = [] 
        self.num_agents: int = 0 
        self.obs_dim: int = 0 
        self.action_dim: int = 0 

        # Validate discrete action spaces (v0) 
        # PettinZoo spaces are gymansium spaces 
        # We'll set these after first reset after when agents are known 

    def reset(self, seed: "Optional[int]" = None) -> Tuple[np.ndarray, np.ndarray]: 
        obs_dict, info_dict = self.env.reset(seed=seed) 


        # Fix ordering 
        if not self.agents: 
            self.agents = list(obs_dict.keys()) 
            self.num_agents = len(self.agents) 


            # infer obs_dim from first agent 
            first_obs = obs_dict[self.agents[0]] 
            first_obs = np.asarray(first_obs, dtype=np.float32).reshape(-1)  
            self.obs_dim = int(first_obs.shape[0])

            # infer action_dim from action space
            # v0 Assumes discrete action
            a0 = self.env.action_space(self.agents[0]) 
            if not hasattr(a0, 'n'): 
                raise ValueError("v0 only supports discrete actions spaces, got: {a0}")
            self.action_dim = int(a0.n) 

        else:
            if self.cfg.strict_agents: 
                if set(obs_dict.keys()) != set(self.agents): 
                    raise ValueError("Agent set changed across agents; v0 assumes fixed agents")
        
        obs_np = self._obs_dict_to_array(obs_dict) 
        critic_obs = self._make_critic_obs(obs_np) 

        return obs_np, critic_obs 
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]: 
        actions_dict = self._actions_array_to_dict(actions) 

        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self.env.step(actions_dict) 

        if self.cfg.strict_agents and set(obs_dict.keys()) != set(self.agents): 
            raise ValueError("Agent set changed during episode; v0 assumes fixed agents") 
        
        obs_np = self._obs_dict_to_array(obs_dict) 
        critic_obs = self._make_critic_obs(obs_np) 

        reward_vec = np.array([reward_dict[a] for a in self.agents], dtype=np.float32)
        # rewards 
        if self.cfg.team_reward: 
            reward = float(reward_vec.sum()) 
        else:
            # If you later add per-agent rewards, return vector here 
            reward = float(reward_vec.mean())
        
        terminated_all = bool(all(term_dict[a] for a in self.agents)) 
        truncated_all = bool(all(trunc_dict[a] for a in self.agents)) 

        done = bool(terminated_all or truncated_all) 

        info = {
            "reward_dict": reward_dict, 
            "reward_vec": reward_vec,
            "terminated": term_dict, 
            "truncated": trunc_dict, 
            "terminated_all": terminated_all, 
            "truncated_all": truncated_all, 
            "info_dict": info_dict,
        }

        return obs_np, critic_obs, reward, done, info 
    
    def _obs_dict_to_array(self, obs_dict: Dict[str, Any]) -> np.ndarray: 
        obs_list = [] 
        for a in self.agents: 
            o = np.asarray(obs_dict[a], dtype=np.float32).reshape(-1) 
            if o.shape[0] != self.obs_dim: 
                raise ValueError(f"Obs dim mismatch for {a}, {o.shape[0]} vs {self.obs_dim}")
            obs_list.append(o) 
        return np.stack(obs_list, axis=0) # (N, obs_dim) 
    
    def _actions_array_to_dict(self, actions: np.ndarray) -> Dict[str, int]: 
        actions_np = np.asarray(actions) 
        if actions_np.shape != (self.num_agents, ): 
            raise ValueError(f"Actions shape must be {(self.num_agents, )}, got {actions_np.shape}")
        
        actions_dict = {} 
        for i, a in enumerate(self.agents): 
            act = int(actions_np[i]) 
            # v0 Discrete action check 
            if act < 0 or act >= self.action_dim: 
                raise ValueError(f"Invalid action {act} for agent {a}, action_dim={self.action_dim}")
            actions_dict[a] = act 
        return actions_dict
    
    def _make_critic_obs(self, obs_np: np.ndarray) -> np.ndarray: 
        return obs_np.reshape(-1).astype(np.float32) 
    

