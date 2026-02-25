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
├── train.py                            # Main entry point (CLI with argparse)
│
├── marlkit/
│   ├── algos/
│   │   ├── ppo/                        # Shared PPO engine (used by MAPPO & IPPO)
│   │   │   ├── config.py              # PPOConfig dataclass (all hyperparameters)
│   │   │   ├── base.py               # BasePPOTrainer — MLP rollout collection, PPO update
│   │   │   ├── recurrent_base.py     # RecurrentBasePPOTrainer — hidden-state mgmt, chunk BPTT
│   │   │   ├── buffer.py             # MultiAgentRolloutBuffer — (T, N, ...) rollouts, GAE
│   │   │   ├── recurrent_buffer.py   # RecurrentMultiAgentRolloutBuffer — + hidden states, chunks
│   │   │   ├── networks.py           # SharedActor, SharedCritic, CentralCritic, MLP
│   │   │   └── recurrent_networks.py # GRU/LSTM/Transformer backbones, recurrent actor & critics
│   │   │
│   │   ├── mappo/                      # MAPPO-specific (centralized critic)
│   │   │   ├── trainer.py            # MAPPOTrainer (BasePPOTrainer subclass)
│   │   │   └── recurrent_trainer.py  # RecurrentMAPPOTrainer
│   │   │
│   │   └── ippo/                       # IPPO-specific (decentralized critic)
│   │       ├── trainer.py            # IPPOTrainer (BasePPOTrainer subclass)
│   │       └── recurrent_trainer.py  # RecurrentIPPOTrainer
│   │
│   ├── envs/
│   │   ├── registry.py               # Centralized env registry (register_env, make_env)
│   │   ├── simple_spread.py          # Toy SimpleSpread env (no external deps)
│   │   ├── simple_hetero.py          # Heterogeneous scout/worker cooperative env
│   │   ├── simple_foraging.py        # Grid-based cooperative foraging env
│   │   ├── pettingzoo_adapter.py     # Adapter wrapping PettingZoo parallel envs
│   │   ├── mpe_factory.py            # Factory for all MPE environments
│   │   └── wrappers.py               # SuperSuit wrappers (pad, normalize, frame-stack)
│   │
│   ├── examples/
│   │   └── custom_env_example.py     # How to register a custom env
│   │
│   └── utils/
│       ├── torch_utils.py            # set_seed(), explained_variance()
│       └── logger/
│           ├── base_logger.py        # Abstract BaseLogger interface
│           ├── logger_factory.py     # make_logger("tensorboard"|"wandb", ...)
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

#### 1.6 Recurrent variants

Both MAPPO and IPPO have **recurrent counterparts** that subclass `RecurrentBasePPOTrainer` instead of `BasePPOTrainer`:

- **`RecurrentMAPPOTrainer`** — centralized critic with a single hidden state (`critic_batch=1`), value replicated across agents.
- **`RecurrentIPPOTrainer`** — decentralized critic with per-agent hidden states (`critic_batch=N`).

These add a `critic_forward_chunk` hook for chunk-based BPTT training, and use `RecurrentMultiAgentRolloutBuffer` which stores actor/critic hidden states at each timestep and splits rollouts into sequential chunks via `get_chunks(chunk_len)`.

Recurrent network backbones (GRU, LSTM, Transformer) are built via `build_backbone()` in `recurrent_networks.py`. The Transformer variant uses a causal-masked encoder with a sliding context window as "hidden state".

#### 1.7 `ppo/config.py` — `PPOConfig`

A `@dataclass` holding all hyperparameters (used by both MAPPO and IPPO, MLP and recurrent):

| Section | Key fields |
|---|---|
| **Rollout** | `num_agents`, `obs_dim`, `action_dim`, `rollout_steps`, `gamma`, `gae_lambda` |
| **PPO** | `clip_eps`, `clip_value`, `clip_value_eps`, `target_kl`, `ent_coef`, `vf_coef`, `max_grad_norm`, `ppo_epochs`, `minibatch_size` |
| **Optimization** | `actor_lr`, `critic_lr`, `weight_decay`, `lr_schedule` (`"constant"` or `"linear"`) |
| **Networks** | `hidden_dim`, `id_embed_dim` |
| **Recurrent** | `use_recurrent`, `recurrent_type` (`"gru"`, `"lstm"`, `"transformer"`), `recurrent_hidden_dim`, `recurrent_num_layers`, `chunk_len`, `transformer_nhead`, `transformer_context_len` |
| **Runtime** | `device`, `seed` |
| **Logging** | `log_every`, `logger_type` (`"none"`, `"wandb"`, `"tensorboard"`), `wandb_project` |
| **Evaluation** | `eval_episodes`, `eval_episodes_det` |
| **Checkpointing** | `checkpoint_dir`, `checkpoint_every`, `resume_from` |

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

#### 2.1 `registry.py` — Environment Registry

All environments are registered via `register_env(name, factory_fn)` and created via `make_env(name, **kwargs)`. Available environments:

| Name | Source | Dependencies |
|---|---|---|
| `toy_simple_spread` | Built-in `SimpleSpreadParallelEnv` | None |
| `mpe_simple_spread` | PettingZoo `simple_spread_v3` | `pettingzoo[mpe]` or `mpe2` |
| `mpe_simple_reference` | PettingZoo `simple_reference_v3` | " |
| `mpe_simple_crypto` | PettingZoo `simple_crypto_v3` | " |
| `mpe_simple_world_comm` | PettingZoo `simple_world_comm_v3` | " |
| `mpe_simple_push` | PettingZoo `simple_push_v3` | " |
| `mpe_simple_adversary` | PettingZoo `simple_adversary_v3` | " |
| `mpe_simple_tag` | PettingZoo `simple_tag_v3` | " |
| `simple_hetero` | Built-in heterogeneous scout/worker env | None |
| `simple_hetero_pz` | PettingZoo wrapper of `simple_hetero` | None |
| `simple_foraging` | Built-in grid foraging env | None |
| `simple_foraging_pz` | PettingZoo wrapper of `simple_foraging` | None |

Custom PettingZoo envs can be registered in one call via `register_pettingzoo_env(name, pz_env_fn)`, or at the CLI with `--pz-env mypackage.my_env:parallel_env`.

#### 2.2 `simple_spread.py` — `SimpleSpreadParallelEnv`

A **dependency-free** toy cooperative environment:

- Each of `N` agents chooses a target slot in `{0, 1, ..., N-1}`.
- **Reward** = `(unique targets chosen) × coverage_reward − (collisions) × collision_penalty`.
- **Observation** per agent: `[normalized_timestep, agent_id / (N-1)]` — just 2 dims.
- Episodes last `episode_len` steps (default 25).

Configuration via `SimpleSpreadEnvConfig(num_agents, episode_len, collision_penalty, coverage_reward)`.

#### 2.3 `simple_hetero.py` — Heterogeneous Cooperative Env

Two agent types (scouts and workers) cooperate. Scouts observe a noisy hint about a hidden target and broadcast a signal; workers observe the previous mean scout signal and select a task. Reward = correct worker matches − mismatch penalties.

- `SimpleHeteroEnv` — direct toolkit env (pads obs/actions to max dims across agent types).
- `SimpleHeteroPZEnv` — PettingZoo parallel env with heterogeneous obs/action spaces.

Configuration via `SimpleHeteroConfig(num_scouts, num_workers, episode_len, ...)`.

#### 2.4 `simple_foraging.py` — Grid Foraging Env

`N` identical agents on a discrete grid cooperate to collect food items. obs\_dim=6, action\_dim=5 (stay, up, down, left, right). Team-shared reward with collect bonus, step penalty, and completion bonus.

- `SimpleForagingEnv` — direct toolkit env.
- `SimpleForagingPZEnv` — PettingZoo parallel env.

Configuration via `SimpleForagingConfig(num_agents, grid_size, num_food, episode_len, ...)`.

#### 2.5 `pettingzoo_adapter.py` — `PettingZooParallelAdapter`

Wraps any PettingZoo **parallel** environment to match the toolkit's array-based interface:

- Converts `Dict[agent, obs]` ↔ `np.ndarray` (stacked in a fixed agent order).
- Aggregates per-agent rewards into a scalar team reward (sum or mean, via `team_reward` config).
- Merges per-agent `terminated`/`truncated` dicts into a single `done` bool.
- Infers `obs_dim` and `action_dim` from the first reset (supports discrete actions only).
- Extracts per-agent action masks from `info_dict` when `use_action_mask=True`.
- Validates that the agent set does not change mid-episode (`strict_agents=True`).

Configuration via `PZAdapterConfig(team_reward, strict_agents, use_action_mask)`.

#### 2.6 `mpe_factory.py` — MPE Environment Factory

`make_mpe_env(module_name, cfg)` creates any supported PettingZoo MPE environment. Dispatches env-specific kwargs (e.g., `N` and `local_ratio` for spread, `num_good`/`num_adversaries`/`num_obstacles` for tag). Tries `pettingzoo.mpe` first, falls back to `mpe2`.

Configuration via `MPEConfig(max_cycles, continuous_actions, N, local_ratio, num_good, num_adversaries, num_obstacles)`.

#### 2.7 `wrappers.py` — SuperSuit Wrappers

`apply_supersuit_wrappers(pz_env, cfg)` applies optional PettingZoo preprocessing:

- Observation padding (`pad_observations_v0`)
- Action space padding (`pad_action_space_v0`)
- Observation normalization (`normalize_obs_v0`)
- Frame stacking (`frame_stack_v1`)

Configuration via `SuperSuitConfig(enabled, pad_observations, pad_action_space, normalize_obs, frame_stack)`.

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

# Recurrent MAPPO with GRU
python train.py --algo mappo --env mpe_simple_spread --recurrent --recurrent-type gru

# With WandB logging and linear LR decay
python train.py --algo mappo --env mpe_simple_spread --logger wandb --lr-schedule linear

# Resume from checkpoint
python train.py --algo mappo --env mpe_simple_spread --resume checkpoints/mappo_SS_MPE/ckpt_100.pt

# Custom PettingZoo env
python train.py --pz-env mypackage.my_env:parallel_env --env my_env_name

# All CLI options:
python train.py --help
```

**CLI arguments:**

| Flag | Default | Description |
|---|---|---|
| **Core** | | |
| `--algo` | `mappo` | Algorithm: `mappo` or `ippo` |
| `--env` | `mpe_simple_spread` | Environment name (see registry table above) |
| `--num-agents` | `3` | Number of agents |
| `--seed` | `0` | Random seed |
| `--iter` | `500` | Number of training iterations |
| `--max-cycles` | `25` | Max episode length |
| `--log-every` | `10` | Print & eval frequency (in iterations) |
| **Recurrent** | | |
| `--recurrent` | `False` | Use recurrent networks |
| `--recurrent-type` | `gru` | `gru`, `lstm`, or `transformer` |
| `--recurrent-hidden-dim` | `64` | Recurrent hidden size |
| `--recurrent-num-layers` | `1` | Number of recurrent layers |
| `--chunk-len` | `16` | BPTT chunk length |
| **Logging** | | |
| `--logger` | `none` | `none`, `wandb`, or `tensorboard` |
| `--wandb-project` | `marlkit` | WandB project name |
| `--run-name` | auto | Run name (auto: `{algo}_{env}_s{seed}`) |
| **Checkpointing** | | |
| `--resume` | `None` | Path to `.pt` checkpoint to resume from |
| **Optimization** | | |
| `--lr-schedule` | `constant` | `constant` or `linear` (decay to 0) |
| **MPE-specific** | | |
| `--local-ratio` | `0.5` | Local reward ratio (simple\_spread) |
| `--num-good` | `1` | Good agents (simple\_tag) |
| `--num-adversaries` | `3` | Adversaries (simple\_tag) |
| `--num-obstacles` | `2` | Obstacles (simple\_tag) |
| **SuperSuit** | | |
| `--supersuit` | `False` | Enable SuperSuit wrappers |
| `--pad-obs` | `False` | Pad observations |
| `--pad-act` | `False` | Pad action space |
| `--norm-obs` | `False` | Normalize observations |
| `--frame-stack` | `1` | Frame stack depth (1 = disabled) |
| **Custom env** | | |
| `--pz-env` | `None` | Import path for a custom PZ parallel env (e.g. `mypackage.my_env:parallel_env`) |

### Expected output

Every `log_every` iterations, the script prints:

```
[mpe_simple_spread | mappo | Iter 10] eval_return=-45.23 ± 3.41    |eval_len=25.0    |greedy return=-42.10 ± 2.87  |EV=0.032
EV=0.032 KL=0.0051 clip=0.08 H=1.59 Ratio=1.002±0.041 Vloss=12.341 Aloss=-0.003
```

- **`eval_return`** — Stochastic evaluation return (sampled actions), mean ± std.
- **`greedy return`** — Deterministic evaluation return (argmax actions).
- **`EV`** — Explained variance of the critic (1.0 = perfect, 0.0 = no better than mean).
- **`KL`** — Approximate KL divergence between old and new policy.
- **`clip`** — Fraction of samples clipped by the PPO objective.
- **`H`** — Policy entropy.
- **`Vloss / Aloss`** — Critic and actor losses.

---

## Training Loop Summary

Each iteration:

1. **Rollout**: Run the current shared policy for `rollout_steps` (256) timesteps, collecting `(obs, action, logp, reward, terminated, truncated, value)` into the buffer. For recurrent models, hidden states are stored and reset on episode boundaries.
2. **GAE**: Compute advantages and returns using Generalized Advantage Estimation with `gamma=0.99`, `lambda=0.95`. Supports truncation bootstrapping.
3. **PPO Update**: For MLP models, flatten the buffer to `T × N` samples and iterate in shuffled minibatches. For recurrent models, split into sequential chunks of length `chunk_len` for BPTT. Both use:
   - Clipped actor loss + entropy bonus
   - MSE critic loss (optionally clipped)
   - Separate actor/critic optimizers with gradient clipping
   - Optional KL early stopping (`target_kl`)
4. **Eval**: Every `log_every` iterations, run stochastic and deterministic (greedy) evaluation episodes (`eval_episodes` / `eval_episodes_det`).
5. **Checkpoint**: Every `checkpoint_every` iterations, save model to `checkpoints/{algo}_{env_abbrev}/ckpt_{iter}.pt`.

---

## Extending the Toolkit

### Adding a new algorithm (PPO-based)

1. Create `marlkit/algos/<your_algo>/trainer.py`.
2. Subclass `BasePPOTrainer` (MLP) or `RecurrentBasePPOTrainer` (recurrent) and implement the abstract hooks:
   - MLP: `critic_values_from_step`, `critic_last_values`, `critic_forward_minibatch`
   - Recurrent: the above plus `critic_forward_chunk` and the `critic_batch` property
3. Add the appropriate critic network to `ppo/networks.py` or `ppo/recurrent_networks.py` (or import your own).
4. Register it in `train.py`'s `build_trainer()` function.

### Adding a new environment

1. Implement the env interface directly (see contract above), **or** wrap a PettingZoo parallel env with `PettingZooParallelAdapter`.
2. Create a factory function: `factory(num_agents, seed, max_cycles, **kwargs) -> (env, obs_dim, action_dim, num_agents)`.
3. Register it via `register_env("my_env", factory)` in `marlkit/envs/registry.py`, or dynamically at the CLI with `--pz-env`.

See `marlkit/examples/custom_env_example.py` for a complete working example.

---

## License

This project is for research and educational purposes.
