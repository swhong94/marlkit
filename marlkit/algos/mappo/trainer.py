# MAPPO Trainer (Thin subclass) 
import numpy as np 
import torch 

from marlkit.algos.ppo.base import BasePPOTrainer 

class MAPPOTrainer(BasePPOTrainer): 
    @torch.no_grad() 
    def critic_values_from_step(self, obs_np, critic_obs_np):
        # Centralized critic V(critic_obs) -> repeated across agents 
        x = torch.tensor(critic_obs_np[None, :], device=self.device) 
        v = self.critic(x).item() 
        return np.full((self.env.num_agents, ), v, dtype=np.float32) 
    
    @torch.no_grad() 
    def critic_last_values(self, obs_np, critic_obs_np):
        x = torch.tensor(critic_obs_np[None, :], device=self.device) 
        v = self.critic(x).item() 
        return np.full((self.env.num_agents, ), v, dtype=np.float32)
    
    def critic_forward_minibatch(self, b_obs, b_ids, b_critic_obs): 
        return self.critic(b_critic_obs) 
    
    