# marlkit/algos/ppo/recurrent_base.py 

import abc 
import numpy as np 
import torch 
import torch.nn as nn 
from torch.optim import Adam 

from marlkit.algos.ppo.recurrent_buffer import RecurrentMultiAgentRolloutBuffer
from marlkit.algos.ppo.recurrent_networks import Hidden
from marlkit.utils.torch_utils import explained_variance 



class RecurrentBasePPOTrainer(abc.ABC): 
    """  
    Recurrent PPO trainer for multi-agent settings. 
    
    Parallel structure to BasePPOTrainer but with hidden state management 
    and chunk-based BPTT training 
    """

    def __init__(self, env, actor, critic, cfg): 
        self.env = env 
        self.actor = actor 
        self.critic = critic 

        self.cfg = cfg 
        self.device = cfg.device 

        self.actor_opt = Adam(self.actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay) 
        self.critic_opt = Adam(self.critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay) 

        self.buffer = RecurrentMultiAgentRolloutBuffer(
            T=cfg.rollout_steps,
            N=cfg.num_agents, 
            obs_dim=cfg.obs_dim, 
            critic_obs_dim=cfg.obs_dim * cfg.num_agents, 
            hidden_dim=cfg.recurrent_hidden_dim,
            num_layers=cfg.recurrent_num_layers, 
            critic_batch=self.critic_batch, 
            device=cfg.device
        )

    def _setup_lr_schedule(self,): 
        """Create LR schedulers based on cfg.lr_schedule."""
        if self.cfg.lr_schedule == "linear": 
            total = getattr(self.cfg, '_lr_total_iters', 1) 
            lr_fn = lambda step: max(1.0 - step / total, 0.0) 
            self.actor_sched = torch.optim.lr_scheduler.LambdaLR(self.actor_opt, lr_fn) 
            self.critic_sched = torch.optim.lr_scheduler.LambdaLR(self.critic_opt, lr_fn) 
        else:
            self.actor_sched = None 
            self.critic_sched = None 

    def _step_lr_schedule(self, ): 
        """Step LR schedulers (call once per update)"""
        if self.actor_sched is not None: 
            self.actor_sched.step() 
            self.critic_sched.step() 

    def save_checkpoint(self, path, iteration): 
        """Save actor, critic, optimizers, and iteration to a .pth file."""
        state = {
            "iteration": iteration,
            "actor": self.actor.state_dict(), 
            "critic": self.critic.state_dict(), 
            "actor_opt": self.actor_opt.state_dict(), 
            "critic_opt": self.critic_opt.state_dict() 
        }
        if self.actor_sched is not None: 
            state["actor_sched"] = self.actor_sched.state_dict() 
            state["critic_sched"] = self.critic_sched.state_dict() 
        torch.save(state, path) 

    def load_checkpoint(self, path): 
        """Load a checkpoint. Returns the saved iteration number."""
        ckpt = torch.load(path, map_location=self.device) 
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"]) 
        self.actor_opt.load_state_dict(ckpt["actor_opt"]) 
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if self.actor_sched is not None and "actor_sched" in ckpt: 
            self.actor_sched.load_state_dict(ckpt["actor_sched"]) 
            self.critic_sched.load_state_dict(ckpt["critic_sched"])
        return ckpt.get("iteration") 

    # ---- Subclass must define critic_batch ---- 
    @property 
    @abc.abstractmethod 
    def critic_batch(self) -> int: 
        """1 for MAPPO (centralized), N for IPPO (per-agent)."""
    

    # ---- Subclass hooks ---- 
    @torch.no_grad()
    @abc.abstractmethod 
    def critic_values_from_step(
        self, obs_np, critic_obs_np, critic_h
    ) -> tuple: 
        """
        Return (values_np, new_critic_h) 
        values_np: (N, ) float32 numpy array 
        critic_h: hidden state (architecture_specific) 
        """

    @torch.no_grad() 
    @abc.abstractmethod
    def critic_last_values(
        self, obs_np, critic_obs_np, critic_h
    ) -> np.ndarray: 
        """Bootstrap values at rollotu end, shape (N, )."""

    @abc.abstractmethod
    def critic_forward_chunk(
        self, b_obs, b_critic_obs, b_ids, b_critic_h, b_critic_c, b_done_mask
    ) -> torch.Tensor: 
        """ 
        Process a minibatch of sequential chunks through the critic.
        b_obs:        (chunk_len, mb, obs_dim)       — per-agent obs (used by IPPO)
        b_critic_obs: (chunk_len, mb, critic_obs_dim) — joint obs (used by MAPPO)
        Returns: values (chunk_len, mb) 
        """

    def _h_to_np(self, h: Hidden): 
        """Convert hidden to numpy arrays (h_np, c_np). For GRU, c_np is zeros."""
        if isinstance(h, tuple): 
            # LSTM: (h, c) - each (num_layers, batch, hidden_dim) 
            return h[0].cpu().numpy(), h[1].cpu().numpy() 
        else: 
            # GRU: single tensor (num_layers, batch, hidden_dim)
            # Transformer: context buffer (context_len, batch, hidden_dim) — variable shape
            h_np = h.cpu().numpy()
            L = self.cfg.recurrent_num_layers
            H = self.cfg.recurrent_hidden_dim
            if h_np.shape[0] != L or (h_np.ndim == 3 and h_np.shape[2] != H):
                # Transformer context doesn't fit buffer slots — store zeros
                B = h_np.shape[1] if h_np.ndim >= 2 else 1
                z = np.zeros((L, B, H), dtype=np.float32)
                return z, z.copy()
            return h_np, np.zeros_like(h_np) 
        
    
    @torch.no_grad() 
    def collect_rollouts(self, seed=None): 
        self.buffer.reset() 
        obs, critic_obs = self.env.reset(seed=seed) 
        N = self.env.num_agents 

        # Initialize hidden states 
        actor_h = self.actor.init_hidden(N, self.device) 
        critic_h = self.critic.init_hidden(self.critic_batch, self.device) 

        for _ in range(self.cfg.rollout_steps): 
            # Snapshot hidden states BEFORE this step (for buffer storage)
            actor_h_np, actor_c_np = self._h_to_np(actor_h) 
            critic_h_np, critic_c_np = self._h_to_np(critic_h) 

            # Critic values for GAE 
            values, critic_h = self.critic_values_from_step(obs, critic_obs, critic_h) 

            # Actor forward: obs (1, N, obs_dim), agent_ids (N, ) 
            obs_t = torch.tensor(obs, device=self.device).unsqueeze(0) 
            agent_ids = torch.arange(N, device=self.device, dtype=torch.long) 
            dist, actor_h = self.actor(obs_t, agent_ids, actor_h) 

            # dist has logits shape (1, N, action_dim) - squeeze seq dim 
            actions = dist.sample() 
            logp = dist.log_prob(actions) 
            actions = actions.squeeze(0) 
            logp = logp.squeeze(0) 

            actions_np = actions.cpu().numpy().astype(np.int64) 
            logp_np = logp.cpu().numpy().astype(np.float32) 

            # Step environment 
            next_obs, next_critic_obs, reward, done, info = self.env.step(actions_np) 

            terminated = False 
            truncated = False 
            if isinstance(info, dict): 
                terminated = bool(info.get("terminated_all", False)) 
                truncated = bool(info.get("truncated_all", False)) 
            if done and not terminated and not truncated: 
                truncated = True  
            # Truncation bootstrap (using current critic hidden, before reset) 
            truncation_values = None 
            if truncated: 
                trunc_vals, _ = self.critic_values_from_step(
                    next_obs, next_critic_obs, critic_h
                )
                truncation_values = trunc_vals.astype(np.float32) 
            
            # Rewards 
            rewards_np = None 
            if self.cfg.use_per_agent_rewards and isinstance(info, dict): 
                rv = info.get("reward_vec", None) 
                if rv is not None: 
                    rewards_np = np.asarray(rv, dtype=np.float32) 
            if rewards_np is None: 
                rewards_np = np.full((N, ), float(reward), dtype=np.float32) 
            
            self.buffer.add(
                obs=obs, 
                critic_obs=critic_obs,
                actions=actions_np, 
                logp=logp_np, 
                rewards=rewards_np, 
                values=values.astype(np.float32), 
                done=done, 
                terminated=terminated,
                truncated=truncated, 
                truncation_values=truncation_values, 
                actor_h=actor_h_np, 
                actor_c=actor_c_np, 
                critic_h=critic_h_np, 
                critic_c=critic_c_np
            )

            obs, critic_obs = next_obs, next_critic_obs 

            # Reset hidden states on episode end 
            if done: 
                obs, critic_obs = self.env.reset() 
                actor_h = self.actor.init_hidden(N, self.device) 
                critic_h = self.critic.init_hidden(self.critic_batch, self.device) 
        
        last_vals, _ = self.critic_last_values(obs, critic_obs, critic_h) 
        self.buffer.compute_gae(last_vals, self.cfg.gamma, self.cfg.gae_lambda, bootstrap_on_truncation=True) 

    
    def update(self): 
        chunks = self.buffer.get_chunks(self.cfg.chunk_len) 
        num_seq = chunks["obs"].shape[0] 
        chunk_len = self.cfg.chunk_len 
        mb = min(self.cfg.minibatch_size, num_seq)  # clamp: minibatch_size is in sequence units

        # Diagnostic accumulators 
        diag = {
            "loss_actor": 0.0, 
            "loss_critic": 0.0, 
            "loss_total": 0.0, 
            "entropy": 0.0, 
            "approx_kl": 0.0, 
            "clip_frac": 0.0, 
            "ratio_mean": 0.0, 
            "ratio_std": 0.0, 
            "grad_norm_actor": 0.0, 
            "grad_norm_critic": 0.0, 
            "num_minibatches": 0, 
        }
        clip_eps = self.cfg.clip_eps 

        for _ in range(self.cfg.ppo_epochs): 
            perm = torch.randperm(num_seq, device=self.device) 

            for start in range(0, num_seq, mb): 
                idx = perm[start:start+mb] 
                cur_mb = idx.shape[0] 

                # Slice minibatch - all shapes: (cur_mb, chunk_len, ...) 
                b_obs = chunks["obs"][idx] 
                b_act = chunks["actions"][idx] 
                b_old_logp = chunks["old_logp"][idx] 
                b_adv = chunks["advantages"][idx] 
                b_ret = chunks["returns"][idx] 
                b_ids = chunks["agent_ids"][idx] 
                b_critic_obs = chunks["critic_obs"][idx] 
                b_done = chunks["done_masks"][idx] 

                # Initial hidden states: (num_layers, mb, hidden_dim) 
                b_actor_h = chunks["init_actor_h"][:, idx, :] 
                b_actor_c = chunks["init_actor_c"][:, idx, :] 
                b_critic_h = chunks["init_critic_h"][:, idx, :] 
                b_critic_c = chunks["init_critic_c"][:, idx, :] 

                # --- Transpose to (seq_len, batch, ...) for RNN --- 
                b_obs_seq = b_obs.transpose(0, 1) 
                b_critic_obs_seq = b_critic_obs.transpose(0, 1) 
                b_done_seq = b_done.transpose(0, 1) 

                # --- Actor forward through chunk --- 
                actor_hidden = self._build_hidden(b_actor_h, b_actor_c) 
                dist, _ = self.actor(b_obs_seq, b_ids, actor_hidden, b_done_seq) 

                b_act_seq = b_act.transpose(0, 1) 
                logp = dist.log_prob(b_act_seq) 
                entropy = dist.entropy().mean() 

                # Flatten chunk dim for PPO loss: (L, mb) -> (L * mb) 
                logp_flat = logp.reshape(-1) 
                old_logp_flat = b_old_logp.reshape(-1) 
                adv_flat = b_adv.reshape(-1) 
                ret_flat = b_ret.reshape(-1) 

                logp_diff = logp_flat - old_logp_flat 
                ratio = torch.exp(logp_diff) 

                surr1 = ratio * adv_flat 
                surr2 = torch.clamp(ratio, 1.0-clip_eps, 1.0+clip_eps) * adv_flat 
                actor_loss = -torch.min(surr1, surr2).mean() - self.cfg.ent_coef * entropy 

                # --- Critic forward through chunk --- 
                v = self.critic_forward_chunk(
                    b_obs_seq, b_critic_obs_seq, b_ids, b_critic_h, b_critic_c, b_done_seq
                )
                v_flat = v.reshape(-1) 
                critic_loss = ((v_flat - ret_flat) ** 2).mean() 

                loss = actor_loss + self.cfg.vf_coef * critic_loss 

                # ---- Diagnostics ---- 
                with torch.no_grad(): 
                    approx_kl = (old_logp_flat - logp_flat).mean() 
                    clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean() 
                    ratio_mean = ratio.mean() 
                    ratio_std = ratio.std(unbiased=False) 

                # ---- Update step ---- 
                self.actor_opt.zero_grad() 
                self.critic_opt.zero_grad() 
                loss.backward() 

                gn_a = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm) 
                gn_c = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm) 


                self.actor_opt.step() 
                self.critic_opt.step() 

                # Accumulate 
                diag["loss_actor"] += actor_loss.item() 
                diag["loss_critic"] += critic_loss.item() 
                diag["loss_total"] += loss.item() 
                diag["entropy"] += entropy.item() 
                diag["approx_kl"] += approx_kl.item() 
                diag["clip_frac"] += clip_frac.item() 
                diag["ratio_mean"] += ratio_mean.item() 
                diag["ratio_std"] += ratio_std.item() 
                diag["grad_norm_actor"] += float(gn_a) 
                diag["grad_norm_critic"] += float(gn_c) 
                diag["num_minibatches"] += 1 

        # Explained variance (full pass) 
        with torch.no_grad(): 
            all_v = [] 
            for start in range(0, num_seq, mb):
                idx = perm[start : start + mb] if start + mb <= num_seq else torch.arange(min(start, num_seq), num_seq, device=self.device) 
                b_obs = chunks["obs"][idx] 
                b_critic_obs = chunks["critic_obs"][idx] 
                b_ids = chunks["agent_ids"][idx] 
                b_done = chunks["done_masks"][idx] 
                b_actor_h = chunks["init_actor_h"][:, idx, :] 
                b_actor_c = chunks["init_actor_c"][:, idx, :] 
                b_critic_h = chunks["init_critic_h"][:, idx, :] 
                b_critic_c = chunks["init_critic_c"][:, idx, :] 
                b_obs_seq = b_obs.transpose(0, 1) 
                b_critic_obs_seq = b_critic_obs.transpose(0, 1) 
                b_done_seq = b_done.transpose(0, 1) 
                v = self.critic_forward_chunk(
                    b_obs_seq, b_critic_obs_seq, b_ids, b_critic_h, b_critic_c, b_done_seq
                ) 
                all_v.append(v.reshape(-1)) 
            v_all = torch.cat(all_v) 
            r_all = chunks["returns"].reshape(-1) 
            ev = explained_variance(v_all, r_all)


        m = max(1, diag["num_minibatches"]) 
        out = {k: v/m for k, v in diag.items() if k!= "num_minibatches"}
        out["explained_variance"] = ev 

        self._step_lr_schedule() 

        return out 
    
    def _build_hidden(self, h, c): 
        """  
        Build Hidden from h, c tensors. 
        If LSTM (c is non-zero), return (h, c). Else return h (GRU). 
        For Transformer, return empty context (variable-length context 
        cannot be stored in fixed-size buffer, so start fresh each chunk).
        """
        if self.cfg.recurrent_type == "lstm": 
            return (h, c) 
        elif self.cfg.recurrent_type == "transformer": 
            # Transformer expects (0, batch, hidden_dim) as empty context 
            # at chunk boundaries; the chunk itself provides sequence context.
            batch_size = h.shape[1] 
            hidden_dim = h.shape[2] 
            return torch.zeros(0, batch_size, hidden_dim, device=h.device) 
        else: 
            return h 
        
