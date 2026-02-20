import torch 

from marlkit.algos.mappo.config import MAPPOConfig
from marlkit.envs.simple_spread import SimpleSpreadParallelEnv, SimpleSpreadEnvConfig
from marlkit.envs.pettingzoo_adapter import PettingZooParallelAdapter, PZAdapterConfig
from marlkit.envs.mpe_factory import make_mpe_simple_spread
from marlkit.utils.torch_utils import set_seed 

from marlkit.algos.ppo.networks import SharedActor, CentralCritic, SharedCritic
from marlkit.algos.mappo.trainer import MAPPOTrainer
from marlkit.algos.ippo.trainer import IPPOTrainer

def main(algo='mappo', pz=False): 
    cfg = MAPPOConfig()
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    set_seed(cfg.seed) 

    if not pz: 
        env = SimpleSpreadParallelEnv(SimpleSpreadEnvConfig(num_agents=cfg.num_agents))
    else: 
        pz_env = make_mpe_simple_spread(num_agents=cfg.num_agents, max_cycles=25, continuous_actions=False) 
        env = PettingZooParallelAdapter(pz_parallel_env=pz_env, cfg=PZAdapterConfig(team_reward=True))

        # Important: reset once to set obs/action dims when using PZAdapter
        _ = env.reset(seed=cfg.seed) 
        cfg.obs_dim = env.obs_dim 
        cfg.action_dim = env.action_dim 

    actor = SharedActor(
        obs_dim=cfg.obs_dim, 
        num_agents=cfg.num_agents, 
        action_dim=cfg.action_dim, 
        hidden_dim=cfg.hidden_dim, 
        id_embed_dim=cfg.id_embed_dim
    ).to(cfg.device)

    if algo=='mappo':
        critic = CentralCritic(
            critic_obs_dim = cfg.num_agents * cfg.obs_dim, 
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
    
    running = 0.0 
    for it in range(1, 501): # run 500 iterations 
        trainer.collect_rollouts(seed=cfg.seed + it) 
        stats = trainer.update() 

        # quick greedy eval 
        obs, critic_obs = env.reset(seed=cfg.seed + 999 + it) 
        done, ep_ret = False, 0.0 
        while not done:  
            obs_t = torch.tensor(obs, device=cfg.device) 
            ids = torch.arange(cfg.num_agents, device=cfg.device, dtype=torch.long) 
            dist = actor.dist(obs_t, ids) 
            act = dist.probs.argmax(dim=-1).cpu().numpy() 
            obs, critic_obs, reward, done, info = env.step(act) 

            ep_ret += reward 
        
        running = 0.9 * running + 0.1 * ep_ret 
        if it % cfg.log_every == 0: 
            print(
                f"[it={it:04d}] eval_ep_reward={ep_ret:7.2f}    "
                f"running_return={running:7.2f}  "
                f"EV={stats['explained_variance']:.3f}"
            )


if __name__ == "__main__": 
    main(algo='mappo', pz=True) 
