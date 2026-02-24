# The shared PPO engine 

import abc 
import numpy as np
import torch 
import torch.nn as nn 
from torch.optim import Adam 

from marlkit.algos.ppo.buffer import MultiAgentRolloutBuffer 
from marlkit.utils.torch_utils import explained_variance 


class BasePPOTrainer(abc.ABC): 
    """Shared PPO trainer for multi-agent settings. 
        - Shared actor network across agents 
        - Two sub-classes implemented: MAPPO and IPPO 
    """
    def __init__(self, env, actor, critic, cfg): 
        self.env = env 
        self.actor = actor 
        self.critic = critic 

        self.cfg = cfg 
        self.device = cfg.device 

        self.actor_opt = Adam(self.actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
        self.critic_opt = Adam(self.critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay) 

        self.buffer = MultiAgentRolloutBuffer(
            T=cfg.rollout_steps, 
            N=cfg.num_agents, 
            obs_dim=cfg.obs_dim, 
            critic_obs_dim=cfg.obs_dim * cfg.num_agents, 
            device=cfg.device 
        )


    # ---- Subclass hooks ---- 
    @torch.no_grad() 
    @abc.abstractmethod 
    def critic_values_from_step(self, obs_np: np.ndarray, critic_obs_np: np.ndarray) -> np.ndarray: 
        """  
        Return values for each agent, shape (N, ) 
        MAPPO: centralized V repeated across N agents 
        IPPO: per_agent V(o^i, id) 
        """
    
    @torch.no_grad() 
    @abc.abstractmethod
    def critic_last_values(self, obs_np: np.ndarray, critic_obs_np: np.ndarray) -> np.ndarray: 
        """  
        Bootstrap values at rollout end, shape (N, ) 
        """

    @abc.abstractmethod 
    def critic_forward_minibatch(self, b_obs, b_ids, b_critic_obs): 
        """  
        Return critic predictions V for the PPO critic loss on the minibatch 
        Must return shape (B, ) 
        """
    

    # ---- rollout ---- 
    @torch.no_grad() 
    def collect_rollouts(self, seed=None): 
        self.buffer.reset() 
        obs, critic_obs = self.env.reset(seed=seed)

        for _ in range(self.cfg.rollout_steps): 
            # critic values (per-agent) for GAE
            values = self.critic_values_from_step(obs, critic_obs) # (N, ) 

            # Sample actions from actor (shared actor) 
            obs_t = torch.tensor(obs, device=self.device) 
            agent_ids = torch.arange(self.env.num_agents, device=self.device, dtype=torch.long)
            dist = self.actor.dist(obs_t, agent_ids) 

            actions = dist.sample() # (N, ) 
            logp = dist.log_prob(actions) # (N, ) 

            actions_np = actions.cpu().numpy().astype(np.int64) 
            logp_np = logp.cpu().numpy().astype(np.float32)

            # Step environment using action_np 
            next_obs, next_critic_obs, reward, done, info = self.env.step(actions_np)

            terminated = False 
            truncated = False 
            if isinstance(info, dict): 
                terminated = bool(info.get("terminated_all", False))
                truncated = bool(info.get("truncated_all", False))
            # If the env doesn't provide terminated/truncated flags, 
            # treat done as truncation (bootstrap with V(s')) so GAE 
            # doesn't silently ignore episode boundaries 
            if done and not terminated and not truncated: 
                truncated = True  

            # Compute V(next_obs) from the truncated episode BEFORE reset
            truncation_values = None 
            if truncated: 
                truncation_values = self.critic_values_from_step(next_obs, next_critic_obs).astype(np.float32)

            # Shared rewards -> repeat for each agent
            rewards_np = None 

            if self.cfg.use_per_agent_rewards and isinstance(info, dict): 
                rv = info.get("reward_vec", None) 
                if rv is not None: 
                    rewards_np = np.asarray(rv, dtype=np.float32) 
                    if rewards_np.shape != (self.env.num_agents, ): 
                        raise ValueError(f"Reward vector shape mismatch, {rewards_np.shape}, expected {(self.env.num_agents, )}")
            if rewards_np is None:  
                rewards_np = np.full((self.env.num_agents, ), float(reward), dtype=np.float32) 

            self.buffer.add(
                obs=obs, 
                critic_obs=critic_obs, 
                actions=actions_np, 
                logp=logp_np, 
                rewards=rewards_np, 
                done=done, 
                values=values.astype(np.float32), 
                terminated=terminated, 
                truncated=truncated,
                truncation_values=truncation_values 
            )

            obs, critic_obs = next_obs, next_critic_obs 

            if done: 
                obs, critic_obs = self.env.reset() 

        last_vals = self.critic_last_values(obs, critic_obs) 
        self.buffer.compute_gae(last_vals, self.cfg.gamma, self.cfg.gae_lambda, bootstrap_on_truncation=True) 
    
    def update(self, ): 
        obs_flat, agent_ids, actions_flat, old_logp_flat, adv_flat, ret_flat, critic_obs_rep = self.buffer.get_flat() 
        total = obs_flat.shape[0] 
        mb = self.cfg.minibatch_size 
        assert mb <= total, "minibatch_size must be <= T * N"

        # --- Diagnostic Accumulators --- 
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
            "num_minibatches": 0
        }

        clip_eps = self.cfg.clip_eps 

        for _ in range(self.cfg.ppo_epochs): 
            perm = torch.randperm(total, device=self.device) 
            for start in range(0, total, mb): 
                batch = perm[start:start+mb]
                
                b_obs = obs_flat[batch] 
                b_ids = agent_ids[batch] 
                b_act = actions_flat[batch] 
                b_old_logp = old_logp_flat[batch] 
                b_adv = adv_flat[batch] 
                b_ret = ret_flat[batch] 

                # ---- Actor loss ---- 
                dist = self.actor.dist(b_obs, b_ids) 
                logp = dist.log_prob(b_act) 
                entropy = dist.entropy().mean() 

                logp_diff = logp - b_old_logp 
                ratio = torch.exp(logp_diff) 

                # PPO clipped objective 
                surr1 = ratio * b_adv 
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * b_adv 
                actor_loss = -torch.min(surr1, surr2).mean() - self.cfg.ent_coef * entropy 

                # ---- Critic Loss (via Subclass) ---- 
                b_critic_obs = critic_obs_rep[batch] 

                v = self.critic_forward_minibatch(
                    b_obs, b_ids, b_critic_obs
                )
                critic_loss = ((v - b_ret)**2).mean()

                loss = actor_loss + self.cfg.vf_coef * critic_loss 

                # ---- Diagnostics ---- 
                with torch.no_grad(): 
                    # A Common Approx for KL for PPO: 
                    approx_kl = (b_old_logp - logp).mean()

                    # A fraction of samples that are clipped 
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

                # ---- Accumulate Diagnostics ---- 
                diag['loss_actor'] += float(actor_loss.item()) 
                diag['loss_critic'] += float(critic_loss.item()) 
                diag['loss_total'] += float(loss.item()) 
                diag['entropy'] += float(entropy.item()) 
                diag['approx_kl'] += float(approx_kl.item()) 
                diag['clip_frac'] += float(clip_frac.item()) 
                diag['ratio_mean'] += float(ratio_mean.item()) 
                diag['ratio_std'] += float(ratio_std.item())
                diag['grad_norm_actor'] += float(gn_a)
                diag['grad_norm_critic'] += float(gn_c)
                diag['num_minibatches'] += 1


        # Diagnostics (Explained Variance) 
        with torch.no_grad(): 
            v_all = self.critic_forward_minibatch(
                obs_flat, agent_ids, critic_obs_rep
            )
            ev = explained_variance(v_all, ret_flat) 
        
        # Average diagnostics over number of minibatches 
        m = max(1, diag['num_minibatches']) 
        out = {k: v / m for k, v in diag.items() if k != 'num_minibatches'} 
        out['explained_variance'] = ev 

        return out





