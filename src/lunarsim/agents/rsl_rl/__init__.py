# SPDX-License-Identifier: BSD-3-Clause
"""Learned agents trained with rsl_rl (milestone 1b).

``policy_action_adapter`` maps between the policy's numbers and the environment's Observation and
Command (shared by training and evaluation). ``lander_vector_environment`` wraps ``LanderEnv`` for
the rsl_rl runner (imports rsl_rl, so only the training script loads it). ``ppo_policy_agent`` is
the ``ppo_mlp`` agent binding that flies an exported policy.
"""
