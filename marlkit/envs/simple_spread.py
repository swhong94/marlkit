import numpy as np 
from dataclasses import dataclass 
from typing import Optional

@dataclass
class SimpleSpreadEnvConfig:
    num_agents: int = 5 
    episode_len: int = 25 
    collision_penalty: float = 1.0 
    coverage_reward: float = 1.0 


class SimpleSpreadParallelEnv: 
    """ 
    Parallel, discrete-action, shared team-reward environment 

    Observation per agent: a small vector that includes 
        - normalized timestep 
        - agent_id / num_agents 
    (We keep it minimal, MAPPO still learn coordination due to shared reward) 
    """
    def __init__(self, config: SimpleSpreadEnvConfig): 
        self.config = config 
        self.num_agents = config.num_agents 
        self.obs_dim = 2 # [normalized timestep, agent_id / num_agents] 
        self.action_dim = config.num_agents 
        self.t = 0 

    def reset(self, seed: "Optional[int]" = None) -> np.ndarray:

        if seed is not None:
            np.random.seed(seed)
        self.t = 0
        obs = self.get_obs()
        critic_obs = self.get_critic_obs(obs)
        return obs, critic_obs
    
    def step(self, actions: np.ndarray):
        """ 
        actions: shape (N, ) int targets 
        returns: obs, critic_obs, reward, done, info 
        """
        assert actions.shape == (self.num_agents, )
        self.t += 1 

        # Coverage = number of unique targets chosen 
        unique = len(set(actions.tolist()))
        collisions = self.num_agents - unique 

        reward = unique * self.config.coverage_reward - collisions * self.config.collision_penalty 

        done = self.t >= self.config.episode_len 

        obs = self.get_obs() 
        critic_obs = self.get_critic_obs(obs) 

        info = {"unique": unique, "collisions": collisions} 

        return obs, critic_obs, float(reward), done, info 

    def get_obs(self): 
        # obs = [t / T, i / N] for agent i 
        t_norm = self.t / max(1, self.config.episode_len) 
        obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32) 
        for i in range(self.num_agents): 
            obs[i, 0] = t_norm 
            obs[i, 1] = i / max(1, self.num_agents -1)
        return obs 
        
    def get_critic_obs(self, obs: np.ndarray) -> np.ndarray: 
        # Concat all agents observations -> (N * obs_dim, ) 
        return obs.reshape(-1).astype(np.float32) 


