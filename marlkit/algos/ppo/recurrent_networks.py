# marlkit/algos/ppo/recurrent_networks.py 

import abc 
import math 
import torch 
import torch.nn as nn 

from torch.distributions import Categorical 
from typing import Tuple, Optional, Union

# Type alias: hidden can be a tensor (GRU) or tuple of tensors (LSTM) 
Hidden = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] 

class RecurrentBackbone(nn.Module, abc.ABC): 
    """Base class for recurrent feature extractors"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int): 
        super().__init__() 
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.num_layers = num_layers 

    @abc.abstractmethod 
    def forward(
        self, 
        x: torch.Tensor, 
        h: Hidden, 
        done_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Hidden]: 
        """Returns (output, new_hidden). output: (seq_len, batch, hidden_dim)"""

    @abc.abstractmethod 
    def init_hidden(self, batch_size: int, device: torch.device) -> Hidden: 
        """Return zero-initialized hidden state."""


class GRUBackbone(RecurrentBackbone): 
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int): 
        super().__init__(input_dim, hidden_dim, num_layers) 
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers) 
    
    def forward(self, x, hidden, done_mask=None): 
        # x: (seq_len, batch, input_dim) 
        # hidden: (num_layers, batch, hidden_dim)
        # done_mask: (seq_len, batch) - 1.0 where episode ENDED at this step 
        seq_len = x.size(0) 
        outputs = [] 

        for t in range(seq_len): 
            # Single-step GRU: input (1, batch, input_dim) 
            out, hidden = self.gru(x[t:t+1], hidden) 
            outputs.append(out) 

            # Zero hidden AFTER processing the step where the episode ended,
            # so the next step (first of new episode) starts with fresh hidden.
            if done_mask is not None:
                # done_mask[t]: (batch, ) -> (1, batch, 1) for broadcasting 
                mask = (1.0 - done_mask[t]).unsqueeze(0).unsqueeze(-1) 
                hidden = hidden * mask 
        
        return torch.cat(outputs, dim=0), hidden    # (seq_len, batch, hidden_dim) 
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor: 
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
    
class LSTMBackbone(RecurrentBackbone):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int): 
        super().__init__(input_dim, hidden_dim, num_layers) 
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers)
    
    def forward(self, x, hidden, done_mask=None): 
        seq_len = x.size(0) 
        outputs = [] 
        h, c = hidden 

        for t in range(seq_len): 
            out, (h, c) = self.lstm(x[t:t+1], (h, c)) 
            outputs.append(out) 

            # Zero hidden AFTER processing the step where the episode ended,
            # so the next step (first of new episode) starts with fresh hidden.
            if done_mask is not None: 
                mask = (1.0 - done_mask[t]).unsqueeze(0).unsqueeze(-1) 
                h = h * mask 
                c = c * mask 
        return torch.cat(outputs, dim=0), (h, c) 

    def init_hidden(self, batch_size, device): 
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device) 
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device) 
        return (h, c) 


class TransformerBackbone(RecurrentBackbone): 
    """  
    Transformer encoder with causal mask.
    
    During training:  chunk is the sequence, no hidden state needed 
    During rollout:   sliding window of past projected features serves as "context"
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 num_layers: int = 1, 
                 nhead: int = 4, 
                 context_len: int = 20): 
        super().__init__(input_dim, hidden_dim, num_layers)
        self.proj = nn.Linear(input_dim, hidden_dim) 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4, 
            batch_first=False, # we use (seq_len, batch, dim) 
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.context_len = context_len 
    
    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor: 
        """Upper-triangular mask: position i can attend to <= i."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'), 
            diagonal=1
        )
    
    def forward(self, x, hidden, done_mask=None): 
        # x: (seq_len, batch, input_dim) 
        # hidden: (context_len, batch, hidden_dim) - past projected features (rollout only) 
        # done_mask: (seq_len, batch) 
        x_proj = self.proj(x) 

        # Concatenate context (past features) with current input 
        if hidden is not None and hidden.size(0) > 0: 
            full_seq = torch.cat([hidden, x_proj], dim=0) 
        else:
            full_seq = x_proj 
        
        causal = self._causal_mask(full_seq.size(0), x.device) 
        out = self.encoder(full_seq, mask=causal) 

        # Only return the output for the current input steps 
        out = out[-x.size(0):]

        new_context = full_seq[-self.context_len:].detach() 

        return out, new_context 
    
    def init_hidden(self, batch_size, device): 
        return torch.zeros(0, batch_size, self.hidden_dim, device=device) 


def build_backbone(recurrent_type: str, 
                   input_dim: int, 
                   hidden_dim: int, 
                   num_layers: int=1, 
                   **kwargs) -> RecurrentBackbone: 
    if recurrent_type == "gru": 
        return GRUBackbone(input_dim, hidden_dim, num_layers) 
    elif recurrent_type == "lstm": 
        return LSTMBackbone(input_dim, hidden_dim, num_layers) 
    elif recurrent_type == "transformer": 
        nhead = kwargs.get("nhead", 4) 
        context_len = kwargs.get("context_len", 20) 
        return TransformerBackbone(input_dim, hidden_dim, num_layers, nhead, context_len)
    else: 
        raise ValueError(f"Unsupported recurrent type: {recurrent_type}")


class RecurrentSharedActor(nn.Module): 
    """Parameter-shared actor with recurrent backbone. 
    
    obs + id_embed -> backbone(GRU/STM/Transformer) -> linear -> action logits
    """
    def __init__(self, 
                 obs_dim: int, 
                 num_agents: int, 
                 action_dim: int, 
                 hidden_dim: int, 
                 id_embed_dim: int, 
                 backbone: RecurrentBackbone): 
        super().__init__() 
        self.id_embed = nn.Embedding(num_agents, id_embed_dim) 
        self.backbone = backbone 
        self.head = nn.Linear(hidden_dim, action_dim) 

    def forward(self, 
                obs: torch.Tensor, 
                agent_ids: torch.Tensor, 
                hidden: Hidden, 
                done_mask: Optional[torch.Tensor] = None) -> Tuple[Categorical, Hidden]: 
        seq_len = obs.size(0) 
        # Expland agent_ids to match seq_len: (batch, ) -> (seq_len, batch, id_embed_dim) 
        embed = self.id_embed(agent_ids).unsqueeze(0).expand(seq_len, -1, -1) 
        x = torch.cat([obs, embed], dim=-1) # (seq_len, batch, obs_dim + id_embed_dim)

        features, new_hidden = self.backbone(x, hidden, done_mask) 
        logits = self.head(features) 
        return Categorical(logits=logits), new_hidden 

    def init_hidden(self, batch_size: int, device: torch.device) -> Hidden: 
        return self.backbone.init_hidden(batch_size, device) 


class RecurrentSharedCritic(nn.Module): 
    """Decentralized recurrent critic with parameter sharing. (for IPPO) 
    
    obs + id_embed -> backbone -> linear -> value """
    def __init__(self, obs_dim: int, num_agents: int, hidden_dim: int, 
                 id_embed_dim: int, backbone: RecurrentBackbone): 
        super().__init__() 
        self.id_embed = nn.Embedding(num_agents, id_embed_dim) 
        self.backbone = backbone 
        self.head = nn.Linear(hidden_dim, 1) 

    def forward(self, obs, agent_ids, hidden, done_mask=None): 
        seq_len = obs.size(0) 
        embed = self.id_embed(agent_ids).unsqueeze(0).expand(seq_len, -1, -1) 
        x = torch.cat([obs, embed], dim=-1) 
        features, new_hidden = self.backbone(x, hidden, done_mask) 
        values = self.head(features).squeeze(-1) 
        return values, new_hidden 
    
    def init_hidden(self, batch_size, device): 
        return self.backbone.init_hidden(batch_size, device) 



class RecurrentCentralizedCritic(nn.Module): 
    """Centralized recurrent critic (for MAPPO) 
    
    joint_obs -> backbone -> linear -> value 
    """
    def __init__(self, critic_obs_dim: int, hidden_dim: int, 
                 backbone: RecurrentBackbone): 
        super().__init__() 
        self.backbone = backbone 
        self.head = nn.Linear(hidden_dim, 1)
    
    def forward(self, critic_obs, hidden, done_mask=None): 
        # critic_obs: (seq_len, batch, critic_obs_dim) [batch=1 during rollout]
        features, new_hidden = self.backbone(critic_obs, hidden, done_mask) 
        values = self.head(features).squeeze(-1) 
        return values, new_hidden 

    def init_hidden(self, batch_size, device): 
        return self.backbone.init_hidden(batch_size, device) 
