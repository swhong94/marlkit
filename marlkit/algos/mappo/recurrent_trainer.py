# Recurrent MAPPO Trainer (thin subclass) 

import numpy as np 
import torch 

from marlkit.algos.ppo.recurrent_base import RecurrentBasePPOTrainer 

class RecurrentMAPPOTrainer(RecurrentBasePPOTrainer): 
    """  
    MAPPO with recurrent networks. 
    Centralized critic: single hidden state (batch=1), value repeated across agents. 
    """

    @property 
    def critic_batch(self) -> int: 
        return 1 
    
    @torch.no_grad() 
    def critic_values_from_step(self, obs_np, critic_obs_np, critic_h): 
        x = torch.tensor(critic_obs_np, device=self.device).reshape(1, 1, -1) 
        v, new_h = self.critic(x, critic_h) 
        val = v.item() 
        return np.full((self.env.num_agents, ), val, dtype=np.float32), new_h 
    
    @torch.no_grad() 
    def critic_last_values(self, obs_np, critic_obs_np, critic_h): 
        x = torch.tensor(critic_obs_np, device=self.device).reshape(1, 1, -1) 
        v, new_h = self.critic(x, critic_h) 
        val = v.item() 
        return np.full((self.env.num_agents, ), val, dtype=np.float32), new_h 
    
    def critic_forward_chunk(self, b_obs, b_critic_obs, b_ids, b_critic_h, b_critic_c, b_done_mask):
        hidden = self._build_hidden(b_critic_h, b_critic_c)
        v, _ = self.critic(b_critic_obs, hidden, b_done_mask)
        return v 
    