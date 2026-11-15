# SPDX-License-Identifier: BSD-3-Clause
"""Export the actor of an rsl_rl checkpoint as ``policy.pt`` for ``agent: ppo_mlp``, without Isaac.

Example::

    python scripts/export_policy.py run=runs/ppo_mlp_s1_seed0 checkpoint=model_500.pt

Reads the training settings from ``runs/<run>/manifest.yaml`` (written by ``train_ppo.py``),
builds the actor network with those settings, loads the checkpoint's actor weights (with the
observation normaliser) and writes ``policy.pt`` next to the checkpoint, or to ``output=``.
"""

import sys
from pathlib import Path

import torch
import yaml
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from lunarsim.agents.rsl_rl.policy_action_adapter import ACTION_DIMENSION_BY_MODE
from lunarsim.agents.rsl_rl.history_models import HISTORY_GROUP, HISTORY_MASK_GROUP
from lunarsim.agents.rsl_rl.training_configuration import export_policy, policy_kind
from lunarsim.core.message_types import NavigationStateLayout


def main() -> None:
    """Rebuild the actor from the manifest, load the checkpoint and export it."""
    options = dict(argument.split("=", 1) for argument in sys.argv[1:])
    run = Path(options["run"])
    checkpoint = run / options.get("checkpoint", "model_1500.pt")
    output = Path(options.get("output", run / "policy.pt"))
    with open(run / "manifest.yaml") as file:
        manifest = yaml.safe_load(file)
    runner = manifest["extra"]["runner"]
    agent_parameters = manifest["resolved_scenario"]["agent"]
    number_of_actions = ACTION_DIMENSION_BY_MODE[agent_parameters.get("action_mode", "attitude_commands")]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    kind = policy_kind(agent_parameters["ppo"])
    # The observation width is read from the saved normaliser (16 navigation values plus any health columns).
    state = saved["actor_state_dict"]
    normaliser = "step_normalizer._mean" if kind == "attention" else "obs_normalizer._mean"
    dimension = int(state[normaliser].shape[-1]) if normaliser in state else NavigationStateLayout.DIMENSION
    history = int(agent_parameters["ppo"].get("attention", {}).get("history_length", 1))
    observations = TensorDict(
        {
            "policy": torch.zeros(1, dimension),
            "critic": torch.zeros(1, NavigationStateLayout.DIMENSION),
            HISTORY_GROUP: torch.zeros(1, history * dimension),
            HISTORY_MASK_GROUP: torch.ones(1, history),
        },
        batch_size=[1],
    )
    actor_settings = dict(runner["actor"])
    actor_class = resolve_callable(actor_settings.pop("class_name"))
    actor = actor_class(observations, runner["obs_groups"], "actor", number_of_actions, **actor_settings)
    actor.load_state_dict(state)
    export_policy(actor, output, kind)
    print(f"[lunarsim] exported iteration {saved.get('iter')} of {checkpoint} to {output}")


if __name__ == "__main__":
    main()
