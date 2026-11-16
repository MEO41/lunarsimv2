# SPDX-License-Identifier: BSD-3-Clause
"""Export a trained policy (``policy.pt``) to ONNX for the S10 policy node (milestone 7). No Isaac needed.

Example::

    python scripts/export_policy_onnx.py policy=runs/ppo_gru_s3_seed0_to1500/policy.pt kind=gru hidden_size=128

Writes ``policy.onnx`` next to the policy (or ``output=``) and ``policy_onnx_check.yaml`` with the
largest output difference against the TorchScript policy on random inputs, and the ONNX engine
that was used (onnxruntime if installed, else the onnx reference evaluator).
"""

import sys
from pathlib import Path

import yaml

from lunarsim.bridge.onnx_policy import OnnxPolicy, export_policy_to_onnx

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Parse ``key=value`` arguments, export, check and write the check file."""
    options = {"policy": None, "kind": "gru", "hidden_size": "128", "output": None}
    for argument in sys.argv[1:]:
        key, _, value = argument.partition("=")
        if key not in options:
            raise SystemExit(f"unknown argument '{argument}' (policy=, kind=, hidden_size=, output=)")
        options[key] = value
    if not options["policy"]:
        raise SystemExit("give policy=<path to policy.pt>")
    policy = Path(options["policy"])
    policy = policy if policy.is_absolute() else REPOSITORY_ROOT / policy
    output = Path(options["output"]) if options["output"] else policy.with_suffix(".onnx")
    report = export_policy_to_onnx(policy, output, options["kind"], int(options["hidden_size"]) if options["kind"] == "gru" else None)
    report.update({"policy": str(policy.relative_to(REPOSITORY_ROOT)) if policy.is_relative_to(REPOSITORY_ROOT) else str(policy), "onnx": output.name, "engine": OnnxPolicy(output).engine})
    with open(output.parent / "policy_onnx_check.yaml", "w") as file:
        yaml.safe_dump(report, file, sort_keys=False)
    print(f"[export] {output}: {report}")


if __name__ == "__main__":
    main()
