---
id: ppo-landing-agent
title: PPO landing agent (milestone 1b)
type: component
status: current
updated: 2026-11-15
summary: The ppo_mlp agent, its rsl_rl adapter, the landing reward, the training script, and how the design reached a policy that learns.
---

# PPO landing agent (milestone 1b)

## Files

| File | What it does |
| --- | --- |
| `src/lunarsim/agents/rsl_rl/policy_action_adapter.py` | Policy numbers ↔ Command and scaled observation (shared by training and flying) |
| `src/lunarsim/agents/rsl_rl/lander_vector_environment.py` | rsl_rl `VecEnv` around `LanderEnv`: decisions every `policy_period_steps`, logs success rate and reward terms |
| `src/lunarsim/agents/rsl_rl/training_configuration.py` | OnPolicyRunner configuration from `configs/agent/ppo_mlp.yaml`; TorchScript export of the actor |
| `src/lunarsim/agents/rsl_rl/ppo_policy_agent.py` | `agent: ppo_mlp`, flies `runs/<run>/policy.pt` |
| `src/lunarsim/env/reward_terms.py` | `LandingRewardTerms`, weights from `configs/training/landing_ppo.yaml` |
| `scripts/train_ppo.py` | `python scripts/train_ppo.py scenario=s1_terminal_truth num_envs=4096 seed=0 run_name=...` |

Flying a trained policy: `python scripts/evaluate_agent.py scenario=s1_terminal_truth agent=ppo_mlp
agent.policy_path=runs/<run>/policy.pt seed=1000 ...` (seed 1000 = held-out starts; ZEM/ZEV runs
on the same seed get the same first-episode starts).

## Design (all model choices, in the config files)

- **Actions** (`attitude_commands`, 3 values in [−1, 1]): throttle piecewise linear (−1 off, 0
  hover at start mass, +1 = 0.65) and a wanted tilt toward site +x/+y up to 15°, flown by the
  classical attitude controller at 50 Hz. The policy decides at 10 Hz (`policy_period_steps: 5`).
- **Observations**: actor = the 16-value navigation estimate (truth in S1), scaled; critic = the
  same 16 values from `TruthState` (`training.critic_observation: truth`).
- **Reward**: success +100; failed touchdown −100; no touchdown −300; touchdown −10 × Σ
  min((value/limit)², 4); velocity tracking −1 × |v − v_wanted| per second of flight (Gaudet,
  Linares, Furfaro 2020); tilt and rate −1 per rad (rad/s) per second; −0.05 per kg propellant.
- **PPO**: rsl_rl 5.4.1, MLP 256-256-128 ELU for actor and critic, empirical normalisation,
  24 decisions per env per iteration, 5 epochs × 4 mini-batches, lr 3e-4 adaptive (KL 0.01),
  γ 0.99, λ 0.95, clip 0.2, no entropy bonus, initial noise std 0.5.

## How the design got there (2026-11-15, 1,024 envs, 300 iterations each, seed 0)

| Trial | Change | Result |
| --- | --- | --- |
| 1 | Raw jet couples, potential shaping (γ = 0.995), uncapped touchdown term, entropy 0.005 | Noise std grew 0.5 → 2.6, touchdown term up to −620; no success |
| 2 | Cap the touchdown term, entropy 0.001 | Learned to hover (shaping pays per hovering step); no success |
| 3 | Plain-difference shaping, time cost, timeouts not bootstrapped | Tipped over in the air: attitude not learned |
| 4 | **Attitude commands** through the classical controller | Flew away: out of bounds (−100) was cheaper than a crash (−100 − touchdown) |
| 5 | **−300 without touchdown**, throttle centred on hover | Hovered until the propellant ran out (50 Hz, γ 0.995 ≈ 4 s horizon) |
| 6 | Velocity-tracking reward | Still hovering (same horizon problem) |
| 7 | **10 Hz decisions, γ 0.99** | Training success 0 % → 94 % between iterations 150 and 300; deterministic 94.4 % on 1,024 held-out starts (seed 1000) |

Tools: `scripts/export_policy.py` exports `policy.pt` from any checkpoint; `lunarsim compare`
prints the comparison table (checks that all runs flew the same starts).

## Policies with memory (milestone 3)

`ppo.model` in the agent file picks the actor; everything else (PPO settings, MLP head
256-256-128, Gaussian distribution, critic on truth, reward, 10 Hz decisions) is shared:

| Agent file | `ppo.model` | Encoder in front of the MLP head |
| --- | --- | --- |
| `ppo_mlp.yaml` | `mlp` | none |
| `ppo_gru.yaml` | `gru` | rsl_rl `RNNModel`, GRU 128, one layer |
| `ppo_attention.yaml` | `attention` | `HistoryAttentionModel` (`agents/rsl_rl/history_models.py`): last 32 decisions (3.2 s), per-feature normaliser on real steps, projection to 64, learned positions, 2-layer 4-head transformer encoder, padding masked; newest step's output to the head (user decision 2026-11-15: custom in rsl_rl) |

With `observation.include_navigation_health: true` (S3) the policy observation is the 16
navigation values plus the filter's health (failed flag and data age per aided sensor), 20 in
all. Exports (`export_policy_module`): MLP obs → action; GRU (obs, hidden) → (action, hidden),
one hidden state per lander; attention (history, mask) → action. The `ppo_mlp` agent binding flies
all three (hidden state or history kept per lander, cleared at reset). Tests: CPU
`test_history_models.py` (padding has no effect, exports equal the models), GPU smoke tests for
GRU and attention in S3.

Learning check, `ppo_gru` in S3 (1,024 envs, seed 0, faults on every episode): training success
0.3 % at iteration 100, 10 % at 200, 71 % at 300, 78 % at 400 (stochastic policy).

## Training runs

The five-seed runs (4,096 envs) run on `minsky` ([[workstation-setup-and-machines]]). They were
started with 1,500 iterations; the user cut them to **600** (2026-11-15): each run is stopped at
its `model_600.pt` and `policy.pt` is exported from it (`scripts/export_policy.py`). S2 policies
train for 600 iterations too. Results go to a results note when done.

- flies on: [[slot-loop-architecture]]
- compared with: [[classical-agents]]
- judged by: [[metrics-and-run-records]]
