# marlkit/algos/ppo/recurrent_buffer.py 

import numpy as np
import torch 

class RecurrentMultiAgentRolloutBuffer: 
    """
    Rollout buffer for recurrent PPO. 

    Key difference from MultiAgentRolloutBuffer: 
        - Stores actor/critic hidden states at each step 
        - get_chunks() returns sequential chunks intead of flath shuffled samples 
    
    Shapes: 
        obs:        (T, N, obs_dim)
        critic_obs: (T, critic_obs_dim) 
        actions:    (T, N)
        logp:       (T, N) 
        rewards:    (T, N) 
        dones:      (T)
        values:     (T, N)
        actor_h:    (T, num_layers, N, hidden_dim)  - actor hidden state
        actor_c:    (T, num_layers, N, hidden_dim)  - actor cell state (LSTM only, zeros for GRU)
        critic_h:   (T, num_layers, critic_batch, hidden_dim)   - critic hidden 
        critic_c:   (T, num_layers, critic_batch, hidden_dim)   - critic cell 
            where critic_batch = 1 for MAPPO (centralized), N for IPPO (per-agent)
        adv/ret:    (T, N)
    """
    def __init__(self, 
                 T: int, 
                 N: int, 
                 obs_dim: int, 
                 critic_obs_dim: int, 
                 hidden_dim: int, 
                 num_layers: int, 
                 critic_batch: int, # 1 for MAPPO, N for IPPO
                 device: str): 
        self.T, self.N = T, N 
        self.obs_dim = obs_dim 
        self.critic_obs_dim = critic_obs_dim 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.critic_batch = critic_batch
        self.device = device
        self.reset() 

    def reset(self): 
        T, N = self.T, self.N 
        L, H = self.num_layers, self.hidden_dim 
        CB = self.critic_batch 

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

        # Hidden states - always allocate h and c (c unused for GRU) 
        self.actor_h = np.zeros((T, L, N, H), dtype=np.float32) 
        self.actor_c = np.zeros((T, L, N, H), dtype=np.float32) 
        self.critic_h = np.zeros((T, L, CB, H), dtype=np.float32) 
        self.critic_c = np.zeros((T, L, CB, H), dtype=np.float32) 

        self.advantages = np.zeros((T, N), dtype=np.float32) 
        self.returns = np.zeros((T, N), dtype=np.float32) 
        self.ptr = 0 

    def add(self, obs, critic_obs, 
            actions, logp, rewards, values, 
            done, terminated, truncated, truncation_values=None,
            actor_h=None, actor_c=None, 
            critic_h=None, critic_c=None):
        """  
        actor_h: (num_layers, N, hidden_dim) np array - hidden BEFORE this step 
        actor_c: same shape - cell state BEFORE this step (None for GRU) 
        critic_h: (num_layers, critic_batch, hidden_dim) - same convention 
        critic_c: same 
        """
        t = self.ptr 
        self.obs[t] = obs 
        self.critic_obs[t] = critic_obs 
        self.actions[t] = actions 
        self.logp[t] = logp 
        self.rewards[t] = rewards 
        self.values[t] = values 
        self.dones[t] = float(done)
        self.terminated[t] = float(terminated) 
        self.truncated[t] = float(truncated) 
        if truncation_values is not None: 
            self.truncation_values[t] = truncation_values
        if actor_h is not None: 
            self.actor_h[t] = actor_h 
        if actor_c is not None: 
            self.actor_c[t] = actor_c 
        if critic_h is not None: 
            self.critic_h[t] = critic_h 
        if critic_c is not None: 
            self.critic_c[t] = critic_c 
        self.ptr += 1 

    def compute_gae(self, last_values, gamma, lam, bootstrap_on_truncation=True): 
        """Same as MLP buffer"""
        T, N = self.T, self.N 
        adv = np.zeros((N, ), dtype=np.float32) 

        for t in reversed(range(T)): 
            term = self.terminated[t] 
            trunc = self.truncated[t] 
            done = term > 0.5 or trunc > 0.5 

            next_values = last_values if t == T - 1 else self.values[t + 1] 

            if trunc > 0.5 and bootstrap_on_truncation: 
                next_values = self.truncation_values[t] 
                nonterminal_value = 1.0 
            elif term > 0.5:
                nonterminal_value = 0.0 
            else:
                nonterminal_value = 1.0 
            
            nonterminal_adv = 0.0 if done else 1.0 

            delta = self.rewards[t] + gamma * next_values * nonterminal_value - self.values[t] 
            adv = delta + gamma * lam * nonterminal_adv * adv 
            self.advantages[t] = adv 

        self.returns = self.advantages + self.values 

        mean = self.advantages.mean() 
        std = self.advantages.std() 
        self.advantages = ((self.advantages - mean) / (std + 1e-12)).astype(np.float32)


    def get_chunks(self, chunk_len: int) -> dict: 
        """
        Split the rollout into fixed-length sequential chunks for recurrent training 
        
        Returns dict of tensors: 
            obs:            (num_seq, chunk_len, obs_dim)
            critic_obs:     (num_seq, chunk_len, critic_obs_dim) 
            actions:        (num_seq, chunk_len)
            old_logp:       (num_seq, chunk_len) 
            advantages:     (num_seq, chunk_len)  
            returns:        (num_seq, chunk_len)
            agent_ids:      (num_seq, )             - which agent each sequence belongs to  
            done_masks:     (num_seq, chunk_len)    - 1.0 at episode boundaries 
            init_actor_h:   (num_layers, num_seq, hidden_dim) 
            init_actor_c:   (num_layers, num_seq, hidden_dim)
            init_critic_h:  (num_layers, num_seq, hidden_dim)
            init_critic_c:  (num_layers, num_seq, hidden_dim)
            
        where num_seq = num_chunks * N (each agent in each chunk = one sequence)
        """
        T, N = self.T, self.N 
        assert T % chunk_len == 0, f"rollout_steps ({T}) must be divisible by chunk_len ({chunk_len})"
        num_chunks = T // chunk_len 

        dev = self.device 

        # ---- Reshape (T, N, ...) -> (num_chunks, chunk_len, N, ...) ---- 
        def chunk_and_merge(arr, extra_dims): 
            """  
            arr: (T, N, *extra_dims) or (T, N)
            Returns: (num_chunks * N, chunk_len, *extra_dims)
            """
            shape = (num_chunks, chunk_len, N) + extra_dims 
            x = arr.reshape(shape) 
            # Transpose to (C, N, L, ...) so each agent-chunk is contiguous 
            axes = [0, 2, 1] + list(range(3, len(shape))) 
            x = np.transpose(x, axes) 
            # Merge chunks and agents 
            merge_shape = (num_chunks * N, chunk_len) + extra_dims 
            return x.reshape(merge_shape) 

        obs = torch.tensor(chunk_and_merge(self.obs, (self.obs_dim,)), device=dev) 
        actions = torch.tensor(chunk_and_merge(self.actions, ()), device=dev) 
        old_logp = torch.tensor(chunk_and_merge(self.logp, ()), device=dev) 
        advantages = torch.tensor(chunk_and_merge(self.advantages, ()), device=dev) 
        returns = torch.tensor(chunk_and_merge(self.returns, ()), device=dev) 

        # critic_obs: (T, critic_obs_dim) - not per-agent, repeat for each agent 
        # (T, ) -> (C, L) -> repeat N -> (C * N, L, critic_obs_dim) 
        critic_obs_chunked = self.critic_obs.reshape(num_chunks, chunk_len, self.critic_obs_dim) 
        critic_obs_rep = np.repeat(critic_obs_chunked, N, axis=0) 
        critic_obs = torch.tensor(critic_obs_rep, device=dev) 

        # done_masks: (T, ) -> (C, L) -> repeat N -> (C * N, L) 
        dones_chunked = self.dones.reshape(num_chunks, chunk_len) 
        done_masks = torch.tensor(np.repeat(dones_chunked, N, axis=0), device=dev) 

        # agent_ids: for each sequence, which agent it is 
        # Pattern: chunk0: [0, 1, ..., N-1], chunk2: [0, 1, ..., N-1], ... 
        agent_ids = torch.arange(N, device=dev, dtype=torch.long).repeat(num_chunks) 

        # ---- Hidden states at chunk boundaries ---- 
        # actor_h: (T, L, N, H) -> extract at t=0, chunk_len, 2*chunk_len, ...
        # Result: (num_chunks, L, N, H) -> transpose to (num_chunks, N, L, H) 
        #       -> merge to (C * N, L, H) -> transpose to (L, C * N, H) 
        chunk_starts = np.arange(0, T, chunk_len) 

        def extract_init_hidden(h_arr): 
            """h_arr: (T, num_layers, batch, hidden_dim) -> (num_layers, num_seq, hidden_dim)
                where batch = N for actor, critic_batch for critic 
            """
            h_init = h_arr[chunk_starts] 
            B = h_init.shape[2] 
            h_init = np.transpose(h_init, (0, 2, 1, 3)) 
            h_init = h_init.reshape(num_chunks * B, self.num_layers, self.hidden_dim) 
            h_init = np.transpose(h_init, (1, 0, 2)) 
            return torch.tensor(h_init, device=dev) 
        
        init_actor_h = extract_init_hidden(self.actor_h) 
        init_actor_c = extract_init_hidden(self.actor_c) 

        # For critic: critic_batch might be 1 (MAPPO) or N (IPPO) 
        # If critic_batch=1, we need to repeat to match num_seq = C * N 
        init_critic_h = extract_init_hidden(self.critic_h) 
        init_critic_c = extract_init_hidden(self.critic_c) 
        if self.critic_batch == 1: 
            # (L, C * 1, H) -> repeat each entry N times -> (L, C * N, H) 
            init_critic_h = init_critic_h.repeat_interleave(N, dim=1) 
            init_critic_c = init_critic_c.repeat_interleave(N, dim=1) 

        return { 
            "obs": obs,                     # (num_seq, chunk_len, obs_dim) 
            "critic_obs": critic_obs,       # (num_seq, chunk_len, critic_obs_dim)
            "actions": actions,             # (num_seq, chunk_len) 
            "old_logp": old_logp,           # (num_seq, chunk_len) 
            "advantages": advantages,       # (num_seq, chunk_len) 
            "returns": returns,             # (num_seq, chunk_len) 
            "agent_ids": agent_ids,         # (num_seq, ) 
            "done_masks": done_masks,       # (num_seq, chunk_len) 
            "init_actor_h": init_actor_h,   # (L, num_seq, H) 
            "init_actor_c": init_actor_c,   # (L, num_seq, H) 
            "init_critic_h": init_critic_h, # (L, num_seq, H) 
            "init_critic_c": init_critic_c, # (L, num_seq, H) 
        }


