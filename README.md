# MARLKit — Multi-Agent Reinforcement Learning Toolkit

A minimal, from-scratch PyTorch toolkit for multi-agent reinforcement learning (MARL) on cooperative tasks. Implements **MAPPO** (Multi-Agent PPO with a centralized critic) and **IPPO** (Independent PPO with a decentralized critic), both using parameter sharing across agents. Designed to be readable, hackable, and self-contained.

---

## Key Ideas

| Concept | How it works here |
|---|---|
| **CTDE** (Centralize Training, Decentralized Execution) | MAPPO trains a single centralized critic $V(o_1 \oplus o_2 \oplus \ldots \oplus o_N)$ on the concatenated observations of all agents, but each agent acts using only its own observation. |
| **Parameter Sharing** | A single actor network is shared by all agents. Agent identity is injected via a learned embedding $e_i$ concatenated to each agent's observation: $\pi(a \mid o_i, e_i)$. |
| **IPPO (baseline)** | Same shared actor, but the critic is also decentralized: $V(o_i, e_i)$ — each agent has its own value estimate (with shared weights). |
| **PPO Clipped Objective** | Standard clipped surrogate with entropy bonus and value-function coefficient. |
| **GAE** | Generalized Advantage Estimation ($\lambda$-returns) for low-variance advantage targets. |

---

## Repository Structure

```
marlkit/
├── __init__.py
├── train.py                        # Main entry point (CLI with argparse)
├── train_mappo.py                  # Standalone MAPPO training script (no CLI)
│
├── marlkit/
│   ├── algos/
│   │   ├── ppo/                    # Shared PPO engine (used by MAPPO & IPPO)
│   │   │   ├── base.py            # BasePPOTrainer — rollout collection, PPO update loop
│   │   │   ├── buffer.py          # MultiAgentRolloutBuffer — stores (T, N, ...) rollouts, GAE
│   │   │   └── networks.py        # SharedActor, SharedCritic, CentralCritic, MLP
│   │   │
│   │   ├── mappo/                  # MAPPO-specific code
│   │   │   ├── config.py          # MAPPOConfig dataclass (all hyperparameters)
│   │   │   └── trainer.py         # MAPPOTrainer — thin subclass of BasePPOTrainer
│   │   │
│   │   └── ippo/                   # IPPO-specific code
│   │       └── trainer.py         # IPPOTrainer — thin subclass of BasePPOTrainer
│   │
│   ├── envs/
│   │   ├── simple_spread.py       # Toy SimpleSpread env (no external deps)
│   │   ├── pettingzoo_adapter.py  # Adapter wrapping PettingZoo parallel envs
│   │   └── mpe_factory.py         # Factory to create MPE simple_spread_v3
│   │
│   └── utils/
│       ├── torch_utils.py         # set_seed(), explained_variance()
│       └── logger/
│           ├── base_logger.py     # Abstract BaseLogger interface
│           ├── logger_factory.py  # make_logger("tensorboard"|"wandb", ...)
│           ├── tensorboard_logger.py
│           └── wandb_logger.py
```

---

## Detailed Module Descriptions

### 1. Algorithms — `marlkit/algos/`

#### 1.1 `ppo/base.py` — `BasePPOTrainer`

The shared PPO training engine inherited by both MAPPO and IPPO. It provides:

- **`collect_rollouts(seed)`** — Runs the policy in the environment for `rollout_steps` timesteps, stores transitions in a `MultiAgentRolloutBuffer`, then calls `compute_gae()` with a bootstrapped terminal value.
- **`update()`** — Flattens the buffer into $(T \times N)$ samples, runs `ppo_epochs` of minibatch SGD with:
  - Clipped policy ratio: $\min\left(r_t A_t,\; \text{clip}(r_t, 1\pm\epsilon)\, A_t\right)$  
  - Entropy bonus: $- c_{\text{ent}} \cdot H[\pi]$  
  - Value loss: $c_{\text{vf}} \cdot (V - R)^2$  
  - Gradient clipping via `max_grad_norm`.
- Three **abstract hooks** that subclasses implement:
  - `critic_values_from_step(obs, critic_obs)` — value estimates during rollout (shape `(N,)`).
  - `critic_last_values(obs, critic_obs)` — bootstrap value at rollout end.
  - `critic_forward_minibatch(b_obs, b_ids, b_critic_obs)` — critic forward pass for the PPO loss.

#### 1.2 `ppo/buffer.py` — `MultiAgentRolloutBuffer`

Stores per-timestep, per-agent data:

| Field | Shape | Description |
|---|---|---|
| `obs` | `(T, N, obs_dim)` | Per-agent observations |
| `critic_obs` | `(T, critic_obs_dim)` | Concatenated joint observation |
| `actions` | `(T, N)` | Discrete actions |
| `logp` | `(T, N)` | Log-probabilities under the old policy |
| `rewards` | `(T, N)` | Shared reward replicated per agent |
| `dones` | `(T,)` | Episode termination flag (shared) |
| `values` | `(T, N)` | Critic value predictions |
| `advantages` | `(T, N)` | GAE-computed, globally normalized |
| `returns` | `(T, N)` | `advantages + values` |

`compute_gae(last_values, gamma, lam)` implements the standard reverse-sweep GAE:

$$\delta_t = r_t + \gamma V_{t+1}(1 - d_t) - V_t$$
$$A_t = \delta_t + \gamma \lambda (1 - d_t) A_{t+1}$$

`get_flat()` reshapes everything to `(T*N, ...)` tensors for minibatch PPO and replicates `critic_obs` per agent via `repeat_interleave`.

#### 1.3 `ppo/networks.py` — Neural Network Architectures

| Class | Input | Output | Used by |
|---|---|---|---|
| `MLP` | arbitrary | arbitrary | Building block for all networks |
| `SharedActor` | `(obs, agent_id)` → concat `[obs, embed(id)]` | `Categorical` distribution over actions | MAPPO & IPPO |
| `CentralCritic` | concatenated joint obs `(N * obs_dim,)` | scalar value $V$ | MAPPO |
| `SharedCritic` | `(obs, agent_id)` → concat `[obs, embed(id)]` | scalar value $V$ | IPPO |

All MLPs are 2-hidden-layer with `Tanh` activation: `Linear → Tanh → Linear → Tanh → Linear`.

#### 1.4 `mappo/trainer.py` — `MAPPOTrainer(BasePPOTrainer)`

Implements the three abstract hooks for centralized training:

- `critic_values_from_step`: Feeds the joint observation into `CentralCritic`, returns the scalar value replicated across all $N$ agents.
- `critic_last_values`: Same as above for bootstrapping.
- `critic_forward_minibatch`: Forward pass on `b_critic_obs` (the joint observation batch).

#### 1.5 `ippo/trainer.py` — `IPPOTrainer(BasePPOTrainer)`

Implements the three abstract hooks for decentralized training:

- `critic_values_from_step`: Feeds each agent's observation + agent ID into `SharedCritic`, returns per-agent values.
- `critic_last_values`: Same as above for bootstrapping.
- `critic_forward_minibatch`: Forward pass on `(b_obs, b_ids)`.

#### 1.6 `mappo/config.py` — `MAPPOConfig`

A `@dataclass` holding all hyperparameters (used by both MAPPO and IPPO):

```python
@dataclass
class MAPPOConfig:
    num_agents: int = 5
    obs_dim: int = 2
    action_dim: int = 5
    rollout_steps: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dim: int = 128
    id_embed_dim: int = 16
    device: str = 'cpu'
    seed: int = 0
    log_every: int = 10
```

---

### 2. Environments — `marlkit/envs/`

All environments expose a uniform interface:

```python
reset(seed=None) -> (obs_np, critic_obs_np)
step(actions_np) -> (obs_np, critic_obs_np, reward: float, done: bool, info: dict)
```

Where:
- `obs_np`: shape `(N, obs_dim)` — per-agent observations (float32)
- `critic_obs_np`: shape `(N * obs_dim,)` — concatenation of all agent observations (float32)
- `actions_np`: shape `(N,)` — one discrete action per agent (int64)
- `reward`: scalar float (shared team reward)
- `done`: bool (shared termination signal)

#### 2.1 `simple_spread.py` — `SimpleSpreadParallelEnv`

A **dependency-free** toy cooperative environment:

- Each of `N` agents chooses a target slot in `{0, 1, ..., N-1}`.
- **Reward** = `(unique targets chosen) × coverage_reward − (collisions) × collision_penalty`.
- **Observation** per agent: `[normalized_timestep, agent_id / (N-1)]` — just 2 dims.
- Episodes last `episode_len` steps (default 25).

Configuration via `SimpleSpreadEnvConfig(num_agents, episode_len, collision_penalty, coverage_reward)`.

#### 2.2 `pettingzoo_adapter.py` — `PettingZooParallelAdapter`

Wraps any PettingZoo **parallel** environment to match the toolkit's array-based interface:

- Converts `Dict[agent, obs]` ↔ `np.ndarray` (stacked in a fixed agent order).
- Aggregates per-agent rewards into a scalar team reward (sum or mean, via `team_reward` config).
- Merges per-agent `terminated`/`truncated` dicts into a single `done` bool.
- Infers `obs_dim` and `action_dim` from the first reset (supports discrete actions only).
- Validates that the agent set does not change mid-episode (`strict_agents=True`).

Configuration via `PZAdapterConfig(team_reward, strict_agents, use_action_mask)`.

#### 2.3 `mpe_factory.py` — `make_mpe_simple_spread()`

Factory function that creates a PettingZoo MPE `simple_spread_v3` parallel environment. Tries `pettingzoo.mpe` first, falls back to the `mpe2` package.

```python
make_mpe_simple_spread(num_agents=3, max_cycles=25, continuous_actions=False)
```

---

### 3. Utilities — `marlkit/utils/`

#### 3.1 `torch_utils.py`

- **`set_seed(seed)`** — Seeds `random`, `numpy`, and `torch` (CPU + CUDA) for reproducibility.
- **`explained_variance(y_pred, y_true)`** — Returns $1 - \text{Var}[y - \hat{y}] / \text{Var}[y]$. A diagnostic metric for critic quality (1.0 = perfect, 0.0 = no better than mean).

#### 3.2 `logger/`

A pluggable logging abstraction:

| Class | Backend | Log directory |
|---|---|---|
| `TensorBoardLogger` | `torch.utils.tensorboard.SummaryWriter` | `runs/<project>/<run_name>/` |
| `WandBLogger` | `wandb` | Managed by W&B |

Both implement `BaseLogger` with: `log_scalar()`, `log_scalars()`, `log_histogram()`, `log_config()`, `close()`.

Create a logger via the factory:

```python
from marlkit.utils.logger.logger_factory import make_logger
logger = make_logger("wandb", project="marlkit", run_name="run-1", config={...})
```

---

## Environment Interface Contract

To add a new environment, implement a class that satisfies:

```python
class MyEnv:
    num_agents: int
    obs_dim: int
    action_dim: int

    def reset(self, seed=None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (obs, critic_obs).
        obs shape: (num_agents, obs_dim)
        critic_obs shape: (num_agents * obs_dim,)
        """

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool, dict]:
        """actions shape: (num_agents,) int64.
        Returns (obs, critic_obs, reward, done, info).
        reward: scalar float (shared team reward).
        done: bool (shared episode termination).
        """
```

Or wrap any PettingZoo parallel environment with `PettingZooParallelAdapter`.

---

## Installation

### Dependencies

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy

**Optional** (for MPE environments):

```bash
pip install pettingzoo[mpe]
# or
pip install mpe2
```

**Optional** (for logging):

```bash
pip install tensorboard
pip install wandb
```

### Setup

```bash
git clone <repo-url> marlkit
cd marlkit
pip install torch numpy
# Optional:
pip install pettingzoo[mpe] wandb tensorboard
```

---

## Usage

### CLI (recommended)

```bash
# MAPPO on PettingZoo MPE Simple Spread (3 agents, 500 iterations)
python train.py --algo mappo --env mpe_simple_spread --num-agents 3 --iter 500

# IPPO on the toy environment (5 agents)
python train.py --algo ippo --env toy_simple_spread --num-agents 5 --iter 500

# All CLI options:
python train.py --help
```

**CLI arguments:**

| Flag | Default | Description |
|---|---|---|
| `--algo` | `mappo` | Algorithm: `mappo` or `ippo` |
| `--env` | `mpe_simple_spread` | Environment: `toy_simple_spread` or `mpe_simple_spread` |
| `--num-agents` | `3` | Number of agents |
| `--seed` | `0` | Random seed |
| `--iter` | `500` | Number of training iterations |
| `--max-cycles` | `25` | Max episode length (MPE only) |
| `--log-every` | `10` | Print metrics every N iterations |

### Standalone scripts

```bash
# MAPPO with PettingZoo MPE (hardcoded settings)
python train_mappo.py

```

### Expected output

Every `log_every` iterations, the script prints:

```
[mpe_simple_spread] | mappo it=0010]   eval_ep_reward=  -45.23  running=  -42.10  EV=0.032
[mpe_simple_spread] | mappo it=0020]   eval_ep_reward=  -38.77  running=  -39.45  EV=0.241
...
```

- **`eval_ep_reward`** — Greedy evaluation episode return (argmax actions).
- **`running`** — Exponentially smoothed return (α=0.1).
- **`EV`** — Explained variance of the critic (closer to 1.0 is better).

---

## Training Loop Summary

Each iteration:

1. **Rollout**: Run the current shared policy for `rollout_steps` (256) timesteps, collecting `(obs, action, logp, reward, done, value)` into the buffer.
2. **GAE**: Compute advantages and returns using Generalized Advantage Estimation with `gamma=0.99`, `lambda=0.95`.
3. **PPO Update**: Flatten the buffer to `T * N` samples. For `ppo_epochs` (4) epochs, shuffle and iterate in minibatches of size `minibatch_size` (256), computing:
   - Clipped actor loss + entropy bonus
   - MSE critic loss
   - Combined gradient step with gradient clipping
4. **Eval**: Run one greedy episode (argmax of the policy) and log the return.

---

## Extending the Toolkit

### Adding a new algorithm

1. Create `marlkit/algos/<your_algo>/trainer.py`.
2. Subclass `BasePPOTrainer` and implement the three hooks: `critic_values_from_step`, `critic_last_values`, `critic_forward_minibatch`.
3. Add the appropriate critic network to `ppo/networks.py` (or import your own).
4. Register it in `train.py`'s `build_trainer()` function.

### Adding a new environment

1. Either implement the env interface directly (see contract above) or use `PettingZooParallelAdapter` for any PettingZoo parallel env.
2. Register it in `train.py`'s `build_env()` function.

---

## License

This project is for research and educational purposes.
