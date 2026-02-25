import argparse 
import torch 
import numpy as np

from marlkit.algos.ppo.config import PPOConfig
from marlkit.utils.torch_utils import set_seed 

from marlkit.algos.ppo.networks import SharedActor, CentralCritic, SharedCritic 
from marlkit.algos.ppo.recurrent_networks import (
    RecurrentSharedActor, RecurrentCentralizedCritic,
    RecurrentSharedCritic, build_backbone)
from marlkit.algos.mappo.recurrent_trainer import RecurrentMAPPOTrainer
from marlkit.algos.ippo.recurrent_trainer import RecurrentIPPOTrainer
from marlkit.algos.mappo.trainer import MAPPOTrainer 
from marlkit.algos.ippo.trainer import IPPOTrainer 

# Supersuit wrappers 
from marlkit.envs.wrappers import SuperSuitConfig, apply_supersuit_wrappers

# Logger 
from marlkit.utils.logger.logger_factory import make_logger
import os 


@torch.no_grad() 
def evaluate_policy(env, 
                    actor, 
                    device: str, 
                    num_agents: int,
                    episodes: int = 5, 
                    seed: int = 0, 
                    deterministic: bool = False, 
                    use_per_agent_rewards: bool = False,
                    recurrent=False): 
    """ Evaluation loop 
    
    - deterministic=False: sample from policy (not greedy) 
    - deterministic=True: greedy argmax 
    - supports per-agent reward vectors (info['reward_vec']) if available 
    """
    ep_returns = [] 
    ep_lengths = [] 

    # If per-agent rewards exist, track them too 
    ep_returns_vec = [] # List of (N, ) arrays 

    for ep in range(episodes): 
        obs, critic_obs = env.reset(seed=seed + ep) 
        done = False 
        ep_ret = 0.0 
        ep_len = 0

        ret_vec = np.zeros((num_agents, ), dtype=np.float32) 

        # Initialize hidden for recurrent actor 
        if recurrent: 
            actor_h = actor.init_hidden(num_agents, device) 

        while not done: 
            obs_t = torch.tensor(obs, device=device) 
            ids = torch.arange(num_agents, device=device, dtype=torch.long) 

            if recurrent: 
                obs_seq = obs_t.unsqueeze(0)            # (1, N, obs_dim)
                dist, actor_h = actor(obs_seq, ids, actor_h) 
            else:
                dist = actor.dist(obs_t, ids) 

            if deterministic: 
                if hasattr(dist, 'probs'): 
                    act = dist.probs.argmax(dim=-1)     # Categorical
                else:
                    act = dist.mean                     # Normal / Continuous 
            else:
                act = dist.sample() 

            if recurrent: 
                act = act.squeeze(0) 
            
            act_np = act.cpu().numpy()

            obs, critic_obs, reward, done, info = env.step(act_np) 

            ep_ret += reward 
            ep_len += 1 

            if use_per_agent_rewards and isinstance(info, dict): 
                rv = info.get("reward_vec", None) 
                if rv is not None: 
                    rv = np.asarray(rv, dtype=np.float32) 
                    if rv.shape == (num_agents, ): 
                        ret_vec += rv
        
        ep_returns.append(ep_ret) 
        ep_lengths.append(ep_len) 
        ep_returns_vec.append(ret_vec) 
    
    out = {
        "eval/return_mean": float(np.mean(ep_returns)), 
        "eval/return_std": float(np.std(ep_returns)), 
        "eval/len_mean": float(np.mean(ep_lengths)), 
        "eval/len_std": float(np.std(ep_lengths))
    }
    if use_per_agent_rewards:
        mat = np.stack(ep_returns_vec, axis=0) # (episodes, N) 
        out.update({
            "eval/per_agent_return_mean": mat.mean(axis=0), 
            "eval/per_agent_return_std": mat.std(axis=0)
        })
    return out 



def build_env(env_name: str, 
              num_agents: int, 
              seed: int, 
              max_cycles: int, 
              ss_cfg: "SuperSuitConfig | None" = None, 
              **env_kwargs): 
    """
    Returns (env, obs_dim, action_dim, num_agents_actual) 
    
    env follows our toolkit env interface 
    
        reset(seed) -> (obs_np, critic_obs_np) 
        step(action_np) -> (obs_np, critic_obs_np, reward, done, info) 
    """
    from marlkit.envs.registry import make_env 

    return make_env(
        env_name, 
        num_agents=num_agents, 
        seed=seed, 
        max_cycles=max_cycles, 
        ss_cfg=ss_cfg, 
        **env_kwargs, 
    )


def build_trainer(algo: str, env, cfg: PPOConfig): 
    if cfg.use_recurrent: 
        return _build_recurrent_trainer(algo, env, cfg) 
    return _build_mlp_trainer(algo, env, cfg) 


def _build_mlp_trainer(algo, env, cfg): 
    actor = SharedActor(
        obs_dim=cfg.obs_dim, 
        num_agents=cfg.num_agents, 
        action_dim=cfg.action_dim, 
        hidden_dim=cfg.hidden_dim, 
        id_embed_dim=cfg.id_embed_dim
    ).to(cfg.device) 

    if algo == 'mappo': 
        critic = CentralCritic(
            critic_obs_dim=cfg.num_agents * cfg.obs_dim, 
            hidden_dim=cfg.hidden_dim 
        ).to(cfg.device) 
        trainer = MAPPOTrainer(env=env, actor=actor, critic=critic, cfg=cfg) 
    elif algo == 'ippo': 
        critic = SharedCritic(
            obs_dim=cfg.obs_dim, 
            num_agents=cfg.num_agents, 
            hidden_dim=cfg.hidden_dim, 
            id_embed_dim=cfg.id_embed_dim
        ).to(cfg.device)
        trainer = IPPOTrainer(env=env, actor=actor, critic=critic, cfg=cfg) 
    else:
        raise ValueError(f"algo must be 'mappo' or 'ippo', got {algo}") 
    
    return trainer, actor 

def _build_recurrent_trainer(algo, env, cfg): 
    # Build actor backbone 
    actor_backbone = build_backbone(
        recurrent_type=cfg.recurrent_type, 
        input_dim=cfg.obs_dim + cfg.id_embed_dim, 
        hidden_dim=cfg.recurrent_hidden_dim,
        num_layers=cfg.recurrent_num_layers,
        nhead=cfg.transformer_nhead, 
        context_len=cfg.transformer_context_len 
    )
    actor = RecurrentSharedActor(
        obs_dim=cfg.obs_dim, 
        num_agents=cfg.num_agents, 
        action_dim=cfg.action_dim,
        hidden_dim=cfg.recurrent_hidden_dim,
        id_embed_dim=cfg.id_embed_dim,
        backbone=actor_backbone
    ).to(cfg.device) 

    if algo == 'mappo': 
        critic_backbone = build_backbone(
            recurrent_type=cfg.recurrent_type, 
            input_dim=cfg.num_agents * cfg.obs_dim, 
            hidden_dim=cfg.recurrent_hidden_dim,
            num_layers=cfg.recurrent_num_layers,
            nhead=cfg.transformer_nhead,
            context_len=cfg.transformer_context_len,
        )
        critic = RecurrentCentralizedCritic(
            critic_obs_dim=cfg.num_agents * cfg.obs_dim, 
            hidden_dim=cfg.recurrent_hidden_dim, 
            backbone=critic_backbone 
        ).to(cfg.device) 
        trainer = RecurrentMAPPOTrainer(env=env, actor=actor, critic=critic, cfg=cfg) 
    elif algo == 'ippo':
        critic_backbone = build_backbone(
            recurrent_type=cfg.recurrent_type,
            input_dim=cfg.obs_dim + cfg.id_embed_dim, 
            hidden_dim=cfg.recurrent_hidden_dim,
            num_layers=cfg.recurrent_num_layers,
            nhead=cfg.transformer_nhead,
            context_len=cfg.transformer_context_len,
        )
        critic = RecurrentSharedCritic(
            obs_dim=cfg.obs_dim, 
            num_agents=cfg.num_agents, 
            hidden_dim=cfg.recurrent_hidden_dim,
            id_embed_dim=cfg.id_embed_dim,
            backbone=critic_backbone
        ).to(cfg.device)
        trainer = RecurrentIPPOTrainer(env=env, actor=actor, critic=critic, cfg=cfg) 
    else: 
        raise ValueError(f"algo must be 'mappo' or 'ippo', got {algo}")
    
    return trainer, actor 


def parse_args(): 
    p = argparse.ArgumentParser() 

    p.add_argument('--algo', choices=['mappo', 'ippo'], default='mappo', help='Training algorithm')
    p.add_argument('--env', default='mpe_simple_spread', help='Environment name') 
    p.add_argument('--num-agents', type=int, default=3, help='Number of participating agents') 
    p.add_argument('--seed', type=int, default=0) 
    p.add_argument('--iter', type=int, default=500) 
    p.add_argument('--max-cycles', type=int, default=25, help='Max episode length') 
    p.add_argument('--log-every', type=int, default=10, help='Logging frequency (in iterations)') 
    p.add_argument('--supersuit', action='store_true', help='Enable Supersuit wrappers (PettingZoo envs only)')
    p.add_argument('--pad-obs', action='store_true', help='Supersuit: pad observations')
    p.add_argument('--pad-act', action='store_true', help='Supersuit: pad actions') 
    p.add_argument('--norm-obs', action='store_true', help='Supersuit: normalize observations') 
    p.add_argument('--frame-stack', type=int, default=1, help='Supersuit: framestack K (K > 1)')

    # MPE-specific options 
    p.add_argument('--local-ratio', type=float, default=0.5, help='MPE: local ratio (simple_spread)')
    p.add_argument('--num-good', type=int, default=1, help="MPE: number of good agents (simple_tag)") 
    p.add_argument('--num-adversaries', type=int, default=3, help="MPE: number of adversary agents (simple_tag)")
    p.add_argument('--num-obstacles', type=int, default=2, help="MPE: number of obstacles (simple_tag)")

    # Custom PettingZoo environment 
    p.add_argument('--pz-env', type=str, default=None, 
                   help="Import path for a custom PettingZoo parallel env," 
                        "e.g., 'mypackag.my_env:parallel_env'. "
                        "Overrides --env with the registered name.")

    # Recurrent options 
    p.add_argument("--recurrent", action='store_true', help="Use recurrent networks") 
    p.add_argument("--recurrent-type", choices=['gru', 'lstm', 'transformer'], default='gru') 
    p.add_argument("--recurrent-hidden-dim", type=int, default=64)
    p.add_argument("--recurrent-num-layers", type=int, default=1) 
    p.add_argument("--chunk-len", type=int, default=16)

    # Logger options 
    p.add_argument("--logger", choices=["none", "wandb", "tensorboard"], default="none", help="Logger backend")
    p.add_argument("--wandb-project", type=str, default="marlkit", help="WandB project name")
    p.add_argument("--run-name", type=str, default=None, help="Run name for logger (auto-generated if omitted)")

    # Checkpoint Resume options 
    p.add_argument("--resume", type=str, default=None, 
                   help="Path to checkpoint to resume from .pt file to resume from")

    return p.parse_args() 


def main(): 
    args = parse_args() 

    ss_cfg = None 
    if args.supersuit: 
        ss_cfg = SuperSuitConfig(
            enabled=True, 
            pad_observations=args.pad_obs, 
            pad_action_space=args.pad_act, 
            normalize_obs=args.norm_obs,
            frame_stack=args.frame_stack
        )

    cfg = PPOConfig() 
    cfg.seed = args.seed 
    cfg.log_every = args.log_every 
    cfg.resume_from = args.resume 

    # Recurrent options 
    cfg.use_recurrent = args.recurrent 
    cfg.recurrent_type = args.recurrent_type 
    cfg.recurrent_hidden_dim = args.recurrent_hidden_dim 
    cfg.recurrent_num_layers = args.recurrent_num_layers 
    cfg.chunk_len = args.chunk_len 

    if cfg.use_recurrent: 
        assert cfg.rollout_steps % cfg.chunk_len == 0, \
            f"rollout_steps ({cfg.rollout_steps}) must be divisible by chunk_len ({cfg.chunk_len})"

    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    set_seed(cfg.seed) 

    # If user passed --pz-env, dynamically import and register it 
    if args.pz_env is not None: 
        import importlib 
        from marlkit.envs.registry import register_pettingzoo_env

        # Parse "module.path:callable_name" 
        if ":" not in args.pz_env: 
            raise ValueError(
                f"--pz-env must be 'module.path:callable', got '{args.pz_env}'"
            )
        mod_path, fn_name = args.pz_env.rsplit(':', 1) 
        mod = importlib.import_module(mod_path) 
        pz_env_fn = getattr(mod, fn_name) 

        # Register under the --env name (default to the callable name) 
        env_name = args.env if args.env != 'mpe_simple_spread' else fn_name 
        args.env = env_name 
        register_pettingzoo_env(env_name, pz_env_fn) 


    env, obs_dim, action_dim, n_actual = build_env(
        env_name=args.env, 
        num_agents=args.num_agents, 
        seed=cfg.seed, 
        max_cycles=args.max_cycles, 
        ss_cfg=ss_cfg, 
        # MPE-speicific (ignored by non-MPE envs via **_kw
        local_ratio=args.local_ratio, 
        num_good=args.num_good, 
        num_adversaries=args.num_adversaries, 
        num_obstacles=args.num_obstacles
    )

    if args.env.startswith('mpe_'): 
        cfg.use_per_agent_rewards = True

    # Sync cfg with actual env specs 
    cfg.num_agents = n_actual 
    cfg.obs_dim = obs_dim 
    cfg.action_dim = action_dim 

    # --- Logger setup --- 
    cfg.logger_type = args.logger 
    cfg.wandb_project = args.wandb_project 
    logger = None 
    if cfg.logger_type != "none": 
        from dataclasses import asdict 
        run_name = args.run_name or f"{args.algo}_{args.env}_s{cfg.seed}"
        logger = make_logger(
            logger_type=cfg.logger_type, 
            project=cfg.wandb_project, 
            run_name=run_name, 
            config=asdict(cfg)
        )

    trainer, actor = build_trainer(algo=args.algo, env=env, cfg=cfg)
    
    start_iter = trainer.load_checkpoint(cfg.resume_from) if cfg.resume_from else 0

    for it in range(start_iter + 1, args.iter + 1): 
        trainer.collect_rollouts(seed=cfg.seed + it) 
        stats = trainer.update() 
        if it % cfg.checkpoint_every == 0:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)  
            trainer.save_checkpoint(cfg.checkpoint_dir, it) 

        if it % cfg.log_every == 0: 

            # Eval episodes 
            eval_stats = evaluate_policy(
                env = env, 
                actor = actor, 
                device = cfg.device, 
                num_agents = cfg.num_agents, 
                episodes = cfg.eval_episodes, 
                seed = cfg.seed + 10_000 + it * 100, 
                deterministic=False, 
                use_per_agent_rewards=getattr(cfg, 'use_per_agent_rewards', False),
                recurrent=cfg.use_recurrent 
            )

            # Optional deterministic (argmax) evaluation too 
            eval_stats_det = evaluate_policy(
                env=env, 
                actor=actor, 
                device=cfg.device, 
                num_agents=cfg.num_agents, 
                episodes=cfg.eval_episodes_det, 
                seed=cfg.seed + 20_000 + it * 100, 
                deterministic=True, 
                use_per_agent_rewards=getattr(cfg, 'use_per_agent_rewards', False),
                recurrent=cfg.use_recurrent
            ) 

        
            if logger is not None: 
                # Training diagnostics 
                for k, v in stats.items(): 
                    logger.log_scalar(f"train/{k}", v, step=it)

                # Eval (stochastic) 
                for k, v in eval_stats.items(): 
                    if isinstance(v, (int, float)):
                        logger.log_scalar(k, v, step=it)

                # Eval (deterministic / greedy) 
                for k, v in eval_stats_det.items(): 
                    if isinstance(v, (int, float)):
                        logger.log_scalar(f"eval_greedy/{k.split('/')[-1]}", v, step=it)

            msg = (
                f"[{args.env} | {args.algo} | Iter {it}] "
                f"eval_return={eval_stats['eval/return_mean']:.2f} ± {eval_stats['eval/return_std']:.2f}    |"
                f"eval_len={eval_stats['eval/len_mean']:.1f}    |"
                f"greedy return={eval_stats_det['eval/return_mean']:.2f} ± {eval_stats_det['eval/return_std']:.2f}  |"
                f"EV={stats['explained_variance']:.3f}"
            )
            print(msg) 
            print(
                f"EV={stats['explained_variance']:.3f} "
                f"KL={stats['approx_kl']:.4f} "
                f"clip={stats['clip_frac']:.2f} "
                f"H={stats['entropy']:.2f} "
                f"Ratio={stats['ratio_mean']:.3f}±{stats['ratio_std']:.3f} "
                f"Vloss={stats['loss_critic']:.3f} "
                f"Aloss={stats['loss_actor']:.3f}"
                )


            # If per-agent is enabled
            if getattr(cfg, 'use_per_agent_rewards', False) and 'eval/per_agent_return_mean' in eval_stats: 
                means = eval_stats['eval/per_agent_return_mean']
                print("  Per-agent return mean (First 5): ", np.array2string(means[:5], precision=2))
                    

    if logger is not None: 
        logger.close()


if __name__ == '__main__': 
    main() 

    


