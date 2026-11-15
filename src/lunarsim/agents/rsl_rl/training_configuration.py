# SPDX-License-Identifier: BSD-3-Clause
"""rsl_rl runner configuration and policy export for the ``ppo_mlp`` agent (milestone 1b).

Imports rsl_rl only inside the functions that need it, so the module loads without it.
"""

from __future__ import annotations

from pathlib import Path

import torch


def runner_configuration(ppo: dict, seed: int, run_name: str, has_critic_group: bool) -> dict:
    """Return the rsl_rl OnPolicyRunner configuration from the agent file's ``ppo:`` section."""
    actor, critic, algorithm = dict(ppo["actor"]), dict(ppo["critic"]), dict(ppo["algorithm"])
    initial_noise_std = float(actor.pop("initial_noise_std"))
    normalize = bool(ppo.get("empirical_normalization", True))
    return {
        "seed": seed,
        "device": "cuda:0",
        "num_steps_per_env": int(ppo["num_steps_per_env"]),
        "max_iterations": int(ppo["max_iterations"]),
        "save_interval": int(ppo["save_interval"]),
        "obs_groups": {"actor": ["policy"], "critic": ["critic"] if has_critic_group else ["policy"]},
        "actor": {
            "class_name": "MLPModel",
            **actor,
            "obs_normalization": normalize,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": initial_noise_std, "std_type": "scalar"},
        },
        "critic": {"class_name": "MLPModel", **critic, "obs_normalization": normalize},
        "algorithm": {"class_name": "PPO", **algorithm, "rnd_cfg": None, "symmetry_cfg": None},
        "logger": "tensorboard",
        "experiment_name": "lunarsim_ppo",
        "run_name": run_name,
        "check_for_nan": True,
    }


def export_policy(actor, path: Path) -> None:
    """Save the actor's deterministic part (normaliser, network, mean output) as TorchScript on the CPU."""
    was_training = actor.training
    actor.eval()
    with torch.no_grad():
        exported = torch.jit.script(actor.as_jit()).to("cpu")
    exported.save(str(path))
    actor.train(was_training)
