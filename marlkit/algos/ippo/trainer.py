# IPPO Trainer (thin subclass) 

import numpy as np
import torch 

from marlkit.algos.ppo.base import BasePPOTrainer


class IPPOTrainer(BasePPOTrainer): 
    @torch.no_grad() 
    def critic_values_from_step(self, obs_np: np.ndarray, critic_obs_np: np.ndarray): 
        # Decentralized critic (V(o^i, id)) for each agent 
        obs_t = torch.tensor(obs_np, device=self.device) 
        ids = torch.arange(self.env.num_agents, device=self.device, dtype=torch.long) 
        v = self.critic(obs_t, ids)
        return v.detach().cpu().numpy().astype(np.float32) 

    @torch.no_grad() 
    def critic_last_values(self, obs_np: np.ndarray, critic_obs_np: np.ndarray):
        obs_t = torch.tensor(obs_np, device=self.device) 
        ids = torch.arange(self.env.num_agents, device=self.device, dtype=torch.long) 
        v = self.critic(obs_t, ids) 
        return v.detach().cpu().numpy().astype(np.float32) 

    def critic_forward_minibatch(self, b_obs, b_ids, b_critic_obs):
        # IPPO critic uses obs + id  
        return self.critic(b_obs, b_ids) 
    