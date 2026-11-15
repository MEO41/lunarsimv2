# SPDX-License-Identifier: BSD-3-Clause
"""rsl_rl runner configuration and policy export for the PPO agents (milestones 1b and 3).

The agent file's ``ppo.model`` picks the actor: ``mlp`` (rsl_rl ``MLPModel``), ``gru`` (rsl_rl
``RNNModel`` with a GRU) or ``attention`` (``HistoryAttentionModel`` over the observation
history). The critic is always an MLP on the ``critic`` group (truth) when there is one. PPO, the
action distribution and the training loop are the same for every kind.
"""

from __future__ import annotations

from pathlib import Path

import torch

POLICY_KINDS = ("mlp", "gru", "attention")
ATTENTION_MODEL_CLASS = "lunarsim.agents.rsl_rl.history_models:HistoryAttentionModel"


def policy_kind(ppo: dict) -> str:
    """Return the actor kind of a ``ppo`` section (default ``mlp``)."""
    kind = str(ppo.get("model", "mlp"))
    if kind not in POLICY_KINDS:
        raise ValueError(f"ppo.model must be one of {POLICY_KINDS}, not '{kind}'")
    return kind


def actor_groups(kind: str) -> list[str]:
    """Return the observation groups the actor of a kind reads."""
    from lunarsim.agents.rsl_rl.history_models import HISTORY_GROUP, HISTORY_MASK_GROUP

    return [HISTORY_GROUP, HISTORY_MASK_GROUP] if kind == "attention" else ["policy"]


def runner_configuration(ppo: dict, seed: int, run_name: str, has_critic_group: bool) -> dict:
    """Return the rsl_rl OnPolicyRunner configuration from the agent file's ``ppo:`` section."""
    actor, critic, algorithm = dict(ppo["actor"]), dict(ppo["critic"]), dict(ppo["algorithm"])
    initial_noise_std = float(actor.pop("initial_noise_std"))
    normalize = bool(ppo.get("empirical_normalization", True))
    kind = policy_kind(ppo)
    actor_settings = {**actor, "obs_normalization": normalize}
    if kind == "mlp":
        actor_settings["class_name"] = "MLPModel"
    elif kind == "gru":
        actor_settings.update({"class_name": "RNNModel", "rnn_type": "gru", **dict(ppo["recurrent"])})
    else:
        actor_settings.update({"class_name": ATTENTION_MODEL_CLASS, **dict(ppo["attention"])})
    actor_settings["distribution_cfg"] = {"class_name": "GaussianDistribution", "init_std": initial_noise_std, "std_type": "scalar"}
    return {
        "seed": seed,
        "device": "cuda:0",
        "num_steps_per_env": int(ppo["num_steps_per_env"]),
        "max_iterations": int(ppo["max_iterations"]),
        "save_interval": int(ppo["save_interval"]),
        "obs_groups": {"actor": actor_groups(kind), "critic": ["critic"] if has_critic_group else ["policy"]},
        "actor": actor_settings,
        "critic": {"class_name": "MLPModel", **critic, "obs_normalization": normalize},
        "algorithm": {"class_name": "PPO", **algorithm, "rnd_cfg": None, "symmetry_cfg": None},
        "logger": "tensorboard",
        "experiment_name": "lunarsim_ppo",
        "run_name": run_name,
        "check_for_nan": True,
    }


def export_policy(actor, path: Path, kind: str = "mlp") -> None:
    """Save the actor's deterministic part as TorchScript on the CPU (see ``history_models``)."""
    from lunarsim.agents.rsl_rl.history_models import export_policy_module

    was_training = actor.training
    exported = export_policy_module(actor, kind)
    exported.save(str(path))
    actor.train(was_training)
