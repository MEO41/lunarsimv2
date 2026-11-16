# SPDX-License-Identifier: BSD-3-Clause
"""ONNX export of a trained policy and a runner with the TorchScript call signature (S10 policy node).

Design S10: the policy node "loads the ONNX export of ppo_gru". The export starts from the
``policy.pt`` that training writes (TorchScript, observation normaliser inside):

- ``mlp``: observation (N, d) -> action (N, a);
- ``gru``: (observation (N, d), hidden state (L, N, h)) -> (action, new hidden state).

The attention export is traced with a fixed history length and is not exported here (S10 uses
``ppo_gru``).

:class:`OnnxPolicy` runs the ONNX file with ``onnxruntime`` when it is installed, else with the
reference evaluator of the ``onnx`` package (pure Python, about 1.5 ms per GRU call on one
environment: enough for a 10 Hz policy). It takes and returns torch tensors, so an agent can use
it in place of its TorchScript policy.
"""

from __future__ import annotations

from pathlib import Path

import numpy
import torch

INPUT_NAMES = {"mlp": ["observation"], "gru": ["observation", "hidden_state"]}
OUTPUT_NAMES = {"mlp": ["action"], "gru": ["action", "new_hidden_state"]}


def policy_input_examples(policy: torch.jit.ScriptModule, hidden_size: int | None) -> tuple[str, tuple[torch.Tensor, ...]]:
    """Return the policy kind and example inputs, found by calling the policy (the observation size is read from the normaliser)."""
    state = policy.state_dict()
    means = [value for name, value in state.items() if name.endswith("obs_normalizer._mean") or name.endswith("_mean")]
    if not means:
        raise ValueError("the policy has no observation normaliser to read the observation size from")
    observation = torch.zeros(1, int(means[0].numel()))
    if hidden_size is None:
        return "mlp", (observation,)
    layers = int(sum(1 for name in state if name.startswith("rnn.weight_hh_l")))
    return "gru", (observation, torch.zeros(layers, 1, hidden_size))


def export_policy_to_onnx(policy_path: Path | str, onnx_path: Path | str, kind: str, hidden_size: int | None = None) -> dict:
    """Export ``policy.pt`` to ONNX (opset 17, any batch size) and check it against the TorchScript policy.

    Returns the largest difference of the outputs on random inputs (``maximum_output_difference``)
    and the names of the inputs and outputs.
    """
    policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
    found_kind, examples = policy_input_examples(policy, hidden_size if kind == "gru" else None)
    if found_kind != kind:
        raise ValueError(f"policy kind {found_kind} does not match {kind}")
    batch_axes = {"observation": {0: "environments"}, "action": {0: "environments"}}
    if kind == "gru":
        batch_axes.update({"hidden_state": {1: "environments"}, "new_hidden_state": {1: "environments"}})
    torch.onnx.export(
        policy, examples, str(onnx_path), dynamo=False, opset_version=17,
        input_names=INPUT_NAMES[kind], output_names=OUTPUT_NAMES[kind], dynamic_axes=batch_axes,
    )
    generator = torch.Generator().manual_seed(0)
    inputs = [torch.randn(3, examples[0].shape[1], generator=generator)]
    if kind == "gru":
        inputs.append(0.3 * torch.randn(examples[1].shape[0], 3, examples[1].shape[2], generator=generator))
    with torch.no_grad():
        expected = policy(*inputs)
    expected = expected if isinstance(expected, tuple) else (expected,)
    produced = OnnxPolicy(onnx_path)(*inputs)
    produced = produced if isinstance(produced, tuple) else (produced,)
    difference = max(float((a - b).abs().max()) for a, b in zip(expected, produced))
    return {"maximum_output_difference": difference, "inputs": INPUT_NAMES[kind], "outputs": OUTPUT_NAMES[kind], "kind": kind}


class OnnxPolicy:
    """Callable ONNX policy with the TorchScript signature (torch tensors in and out)."""

    def __init__(self, onnx_path: Path | str) -> None:
        """Open the ONNX file with onnxruntime if available, else with the onnx reference evaluator."""
        self.path = str(onnx_path)
        try:
            import onnxruntime

            self.session = onnxruntime.InferenceSession(self.path, providers=["CPUExecutionProvider"])
            self.input_names = [item.name for item in self.session.get_inputs()]
            self.engine = "onnxruntime"
        except ImportError:
            import onnx
            from onnx.reference import ReferenceEvaluator

            model = onnx.load(self.path)
            self.session = ReferenceEvaluator(model)
            self.input_names = [item.name for item in model.graph.input]
            self.engine = "onnx_reference"

    def eval(self) -> OnnxPolicy:
        """Do nothing (TorchScript compatibility)."""
        return self

    def __call__(self, *inputs: torch.Tensor):
        """Run the policy; return a tensor (MLP) or a tuple of tensors (GRU) on the inputs' device."""
        device = inputs[0].device
        feeds = {name: value.detach().to("cpu", torch.float32).numpy() for name, value in zip(self.input_names, inputs)}
        outputs = self.session.run(None, feeds)
        tensors = tuple(torch.from_numpy(numpy.asarray(output, dtype=numpy.float32)).to(device) for output in outputs)
        return tensors[0] if len(tensors) == 1 else tensors
