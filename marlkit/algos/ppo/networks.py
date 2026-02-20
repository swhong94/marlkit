import torch 
import torch.nn as nn 

from torch.distributions import Categorical


class MLP(nn.Module): 
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,): 
        super().__init__() 
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Tanh(), 
            nn.Linear(hidden_dim, output_dim) 
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.net(x) 
    

class SharedActor(nn.Module): 
    """ 
    Parameter sharing with agent_id embedding 
    """
    def __init__(self, 
                 obs_dim: int, 
                 num_agents: int, 
                 action_dim: int, 
                 hidden_dim: int, 
                 id_embed_dim: int): 
        super().__init__() 
        self.id_embed = nn.Embedding(num_agents, id_embed_dim) 
        self.mlp = MLP(obs_dim + id_embed_dim, hidden_dim, action_dim) 

    def dist(self,
             obs: torch.Tensor, 
             agent_ids: torch.Tensor) -> Categorical: 
        embed = self.id_embed(agent_ids) 
        x = torch.cat([obs, embed], dim=-1) 
        logits = self.mlp(x) 
        return Categorical(logits=logits) 


class SharedCritic(nn.Module): 
    """ 
    Decentralized critic with shared parameters, parameter sharing with agent_id embedding
    """
    def __init__(self, 
                 obs_dim: int, 
                 num_agents: int, 
                 hidden_dim: int, 
                 id_embed_dim: int): 
        super().__init__() 
        self.id_embed = nn.Embedding(num_agents, id_embed_dim) 
        self.mlp = MLP(obs_dim + id_embed_dim, hidden_dim, 1)

    def forward(self,
                obs: torch.Tensor, 
                agent_ids: torch.Tensor) -> torch.Tensor: 
        embed = self.id_embed(agent_ids) 
        x = torch.cat([obs, embed], dim=-1) 
        return self.mlp(x).squeeze(-1) 
    

class CentralCritic(nn.Module): 
    """
    Centralized critic: Critic(Joint obs) -> V 
    """
    def __init__(self, 
                 critic_obs_dim: int, 
                 hidden_dim: int): 
        super().__init__() 
        self.v = MLP(critic_obs_dim, hidden_dim, 1) 

    def forward(self, 
                critic_obs: torch.Tensor) -> torch.Tensor: 
        return self.v(critic_obs).squeeze(-1) 

