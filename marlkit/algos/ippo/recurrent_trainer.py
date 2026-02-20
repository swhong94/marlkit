# Recurrent IPPO Trainer (thin subclass) 

import numpy as np
import torch

from marlkit.algos.ppo.recurrent_base import RecurrentBasePPOTrainer

class RecurrentIPPOTrainer(RecurrentBasePPOTrainer): 
    """ 
    IPPO with recurrent networks. 
    Decentralized critic: per-agent hidden states (batch=N) 
    """

    @property
    def critic_batch(self) -> int: 
        return self.cfg.num_agents 
    
    @torch.no_grad()
    def critic_values_from_step(self, obs_np, critic_obs_np, critic_h): 
        obs_t = torch.tensor(obs_np, device=self.device).unsqueeze(0) 
        ids = torch.arange(self.env.num_agents, device=self.device, dtype=torch.long) 
        v, new_h = self.critic(obs_t, ids, critic_h) 
        vals = v.squeeze(0).cpu().numpy().astype(np.float32) 
        return vals, new_h
    
    @torch.no_grad() 
    def critic_last_values(self, obs_np, critic_obs_np, critic_h):
        obs_t = torch.tensor(obs_np, device=self.device).unsqueeze(0) 
        ids = torch.arange(self.env.num_agents, device=self.device, dtype=torch.long) 
        v, new_h = self.critic(obs_t, ids, critic_h) 
        vals = v.squeeze(0).cpu().numpy().astype(np.float32) 
        return vals, new_h
    
    def critic_forward_chunk(self, b_obs, b_critic_obs, b_ids, b_critic_h, b_critic_c, b_done_mask):
        hidden = self._build_hidden(b_critic_h, b_critic_c) 
        v, _ = self.critic(b_obs, b_ids, hidden, b_done_mask) 
        return v
    