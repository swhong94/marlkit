import numpy as np 
import torch 

class MultiAgentRolloutBuffer: 
    """
    Stores rollout for PPO-style updates.
    Shapes:
        obs:        (T, N, obs_dim)
        critic_obs: (T, C)   [optional; used by MAPPO]
        actions:    (T, N)
        logp:       (T, N)
        rewards:    (T, N)   [we can fill shared reward by repeating]
        dones:      (T,)     [shared done]
        values:     (T, N)   [MAPPO repeats centralized V across agents]
        adv/ret:    (T, N)
    """
    def __init__(self, 
                 T: int, 
                 N: int, 
                 obs_dim: int, 
                 critic_obs_dim: int, 
                 device: str):
        self.T, self.N = T, N 
        self.obs_dim = obs_dim 
        self.critic_obs_dim = critic_obs_dim 
        self.device = device 
        self.reset() 

    def reset(self, ): 
        T, N = self.T, self.N 
        self.obs = np.zeros((T, N, self.obs_dim), dtype=np.float32) 
        self.critic_obs = np.zeros((T, self.critic_obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, N), dtype=np.int64) 
        self.logp = np.zeros((T, N), dtype=np.float32) 
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, ), dtype=np.float32)
        self.terminated = np.zeros((T, ), dtype=np.float32) 
        self.truncated = np.zeros((T, ), dtype=np.float32)  
        self.values = np.zeros((T, N), dtype=np.float32) 
        self.truncation_values = np.zeros((T, N), dtype=np.float32)

        self.advantages = np.zeros((T, N), dtype=np.float32) 
        self.returns = np.zeros((T, N), dtype=np.float32)
        self.ptr = 0 

    def add(self, obs, critic_obs, actions, logp, rewards, done, values, terminated: bool, truncated: bool, truncation_values=None):
        t = self.ptr 
        self.obs[t] = obs 
        self.critic_obs[t] = critic_obs 
        self.actions[t] = actions 
        self.logp[t] = logp 
        self.rewards[t] = rewards 
        self.dones[t] = float(done) 
        self.terminated[t] = float(terminated)
        self.truncated[t] = float(truncated) 
        self.values[t] = values 
        if truncation_values is not None:
            self.truncation_values[t] = truncation_values
        self.ptr += 1

    def compute_gae(self, last_values: np.ndarray, gamma: float, lam: float, bootstrap_on_truncation: bool = True): 
        """
        last_values: (N, ) np.ndarray, bootstrapped values at end 
        """
        T, N = self.T, self.N 
        adv = np.zeros((N, ), dtype=np.float32) 

        for t in reversed(range(T)): 
            term = self.terminated[t] 
            trunc = self.truncated[t] 
            done = term > 0.5 or trunc > 0.5

            next_values = last_values if t == T - 1 else self.values[t + 1] 

            # On truncation, bootstrap with V(s') from the truncated episode
            # (not V(first obs of next episode) which is what self.values[t+1] holds)
            if trunc > 0.5 and bootstrap_on_truncation:
                next_values = self.truncation_values[t]
                nonterminal_value = 1.0
            elif term > 0.5:
                nonterminal_value = 0.0 
            else:
                nonterminal_value = 1.0 

            # Never propagate advantage across episode boundaries
            nonterminal_adv = 0.0 if done else 1.0

            delta = self.rewards[t] + gamma * next_values * nonterminal_value - self.values[t]
            adv = delta + gamma * lam * nonterminal_adv * adv 
            self.advantages[t] = adv 
        
        self.returns = self.advantages + self.values 

        # Normalize advantage globally (over all T * N) 
        mean = self.advantages.mean() 
        std = self.advantages.std() 
        self.advantages = ((self.advantages - mean) / (std + 1e-12)).astype(np.float32)

    def get_flat(self): 
        """ 
        Returns flattened tensors for PPO:
            obs_flat:       (T * N, obs_dim) tensor 
            agent_ids:      (T * N, ) tensor
            actions_flat:   (T * N, ) tensor
            old_logp_flat:  (T * N, ) tensor
            adv_flat:       (T * N, ) tensor
            ret_flat:       (T * N, ) tensor
        Also returns critic_obs_rep: (T * N, critic_obs_dim) tensor for MAPPO
        """
        T, N = self.T, self.N 

        obs_flat = torch.tensor(self.obs.reshape(T * N, self.obs_dim), device=self.device) 
        agent_ids = torch.arange(N, device=self.device, dtype=torch.long).repeat(T)
        actions_flat = torch.tensor(self.actions.reshape(T * N), device=self.device) 
        old_logp_flat = torch.tensor(self.logp.reshape(T * N), device=self.device) 
        adv_flat = torch.tensor(self.advantages.reshape(T * N), device=self.device)
        ret_flat = torch.tensor(self.returns.reshape(T * N), device=self.device)

        # For MAPPO, critic obs 
        critic_obs = torch.tensor(self.critic_obs, device=self.device) # (T, critic_obs_dim) 
        critic_obs_rep = critic_obs.repeat_interleave(N, dim=0) # (T * N, critic_obs_dim) 

        return obs_flat, agent_ids, actions_flat, old_logp_flat, adv_flat, ret_flat, critic_obs_rep 
    
    



        