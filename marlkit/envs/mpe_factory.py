def make_mpe_simple_spread(num_agents: int=3, 
                           max_cycles: int=25, 
                           continuous_actions: bool=False,): 
    # PettingZoo legacy path: 
    try: 
        from pettingzoo.mpe import simple_spread_v3 
        env = simple_spread_v3.parallel_env(
            N=num_agents, 
            max_cycles=max_cycles, 
            continuous_actions=continuous_actions,    
        )
        return env 
    except Exception:
        # New path 
        from mpe2 import simple_spread_v3 
        env = simple_spread_v3.parallel_env(
            N=num_agents, 
            max_cycles=max_cycles, 
            continuous_actions=continuous_actions,
        )
        return env 


