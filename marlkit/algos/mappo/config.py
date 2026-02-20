from dataclasses import dataclass 

@dataclass 
class MAPPOConfig: 
    # Rollout 
    num_agents: int = 5 
    obs_dim: int = 2
    action_dim: int = 5 
    rollout_steps: int = 256 
    gamma: float = 0.99 
    gae_lambda: float = 0.95 

    # PPO 
    clip_eps: float = 0.2 
    ent_coef: float = 0.01 
    vf_coef: float = 0.5 
    max_grad_norm: float = 0.5 
    ppo_epochs: int = 4 
    minibatch_size: int = 256   # across time and agents 

    # Optimization
    actor_lr: float = 3e-4 
    critic_lr: float = 3e-4 
    weight_decay: float = 0.0 

    # NETS 
    hidden_dim: int = 128 
    id_embed_dim: int = 16 

    # RECURRENT (only used when use_recurrent=True) 
    use_recurrent: bool = False 
    recurrent_type: str = "gru" # one of "gru", "lstm", or "transformer" 
    recurrent_hidden_dim: int = 64 
    recurrent_num_layers: int = 1 
    chunk_len: int = 16             # BPTT chunk length 
    transformer_nhead: int = 4       # Only for transformer 
    transformer_context_len: int = 20 # Only for transformer

    # RUNTIME 
    device: str = 'cpu' 
    seed: int = 0 

    # LOGGING 
    log_every: int = 10 
    logger_type: str = "none"           # "none", "wandb", or "tensorboard"
    wandb_project: str = "marlkit"      # WandB project name

    # Reward Handling 
    use_per_agent_rewards: bool = False # True for PettingZoo envs with per-agent rewards 
    
     