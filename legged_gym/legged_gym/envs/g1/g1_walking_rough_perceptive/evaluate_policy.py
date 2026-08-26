"""Evaluate either blind or perceptive rough walking with common metrics.

Select inputs through environment variables so this remains compatible with
the repository's existing Isaac Gym argument parser:

G1_EVAL_TASK, G1_EVAL_CHECKPOINT, G1_EVAL_NUM_ENVS, G1_EVAL_STEPS,
G1_PERCEPTIVE_MODE.
"""

import json
import os

import isaacgym  # must be imported before torch
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


SUPPORTED_TASKS = ("g1_walking_rough", "g1_walking_rough_perceptive")


def main():
    args = get_args(test=True)
    task_name = os.environ.get(
        "G1_EVAL_TASK", "g1_walking_rough_perceptive"
    )
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(
            "G1_EVAL_TASK must be one of {}".format(SUPPORTED_TASKS)
        )
    checkpoint_path = os.environ.get("G1_EVAL_CHECKPOINT")
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "Set G1_EVAL_CHECKPOINT to a trained {} checkpoint".format(
                task_name
            )
        )

    args.task = task_name
    args.resume = False
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = int(os.environ.get("G1_EVAL_NUM_ENVS", "256"))
    env_cfg.domain_rand.push_robots = (
        os.environ.get("G1_EVAL_ENABLE_PUSHES", "1") == "1"
    )
    if task_name == "g1_walking_rough_perceptive":
        env_cfg.perception.mode = os.environ.get(
            "G1_PERCEPTIVE_MODE", "noisy"
        ).lower()

    env, _ = task_registry.make_env(
        name=task_name, args=args, env_cfg=env_cfg
    )
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=None,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    # The repository runner falls back to cuda:0 when a CUDA checkpoint is
    # opened on CPU. Evaluation must instead follow the selected RL device.
    checkpoint = torch.load(checkpoint_path, map_location=runner.device)
    runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
    policy = runner.get_inference_policy(device=env.device)
    obs, _ = env.reset()

    steps = int(os.environ.get("G1_EVAL_STEPS", "5000"))
    tracking_error_sum = 0.0
    orientation_error_sum = 0.0
    slip_speed_sum = 0.0
    torque_sq_sum = 0.0
    fall_count = 0
    timeout_count = 0
    sample_count = 0
    perception_error_sum = 0.0

    with torch.inference_mode():
        for _ in range(steps):
            actions = policy(obs)
            obs, _, _, done, _ = env.step(actions)
            tracking_error_sum += torch.sum(
                torch.linalg.vector_norm(
                    env.base_lin_vel[:, :2] - env.commands[:, :2], dim=1
                )
            ).item()
            orientation_error_sum += torch.sum(
                torch.linalg.vector_norm(env.projected_gravity[:, :2], dim=1)
            ).item()
            contact = env.contact_forces[:, env.feet_indices, 2] > 1.0
            slip_speed_sum += torch.sum(
                torch.linalg.vector_norm(env.feet_vel[:, :, :2], dim=2)
                * contact
            ).item()
            torque_sq_sum += torch.sum(torch.square(env.torques)).item()
            timeout_count += int(torch.sum(done.bool() & env.time_out_buf).item())
            fall_count += int(torch.sum(done.bool() & ~env.time_out_buf).item())
            if hasattr(env, "last_actor_terrain_obs"):
                perception_error_sum += torch.sum(
                    torch.mean(
                        torch.abs(
                            env.last_actor_terrain_obs
                            - env.last_actor_terrain_truth
                        ),
                        dim=1,
                    )
                ).item()
            sample_count += env.num_envs

    result = {
        "task": task_name,
        "checkpoint": checkpoint_path,
        "num_envs": env.num_envs,
        "steps": steps,
        "falls": fall_count,
        "timeouts": timeout_count,
        "mean_velocity_tracking_error_mps": tracking_error_sum / sample_count,
        "mean_orientation_error": orientation_error_sum / sample_count,
        "mean_contact_foot_slip_speed_mps": slip_speed_sum / sample_count,
        "mean_squared_torque": torque_sq_sum
        / (sample_count * env.num_actions),
    }
    if hasattr(env, "last_actor_terrain_obs"):
        result["perception_mode"] = env.terrain_perception.mode
        result["mean_heightmap_error_m"] = (
            perception_error_sum / sample_count
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
