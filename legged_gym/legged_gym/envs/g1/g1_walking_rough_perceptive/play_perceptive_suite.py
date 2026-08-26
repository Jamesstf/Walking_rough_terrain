"""Benchmark and visualize a perceptive checkpoint on every hardest terrain."""

import json
import math
import os

import isaacgym  # must be imported before torch
import torch

from rsl_rl.modules import ActorCriticPerceptive
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

from legged_gym.envs.g1.g1_walking_rough.play_rough_suite import (
    TERRAIN_NAMES,
    VISUAL_ORDER,
    set_camera_for_env,
)


TASK_NAME = "g1_walking_rough_perceptive"


def load_policy(env, train_cfg, checkpoint_path):
    cfg = train_cfg.policy
    actor_critic = ActorCriticPerceptive(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_privileged_obs,
        num_actions=env.num_actions,
        proprio_obs_dim=cfg.proprio_obs_dim,
        terrain_obs_dim=cfg.terrain_obs_dim,
        terrain_encoder_dims=cfg.terrain_encoder_dims,
        actor_hidden_dims=cfg.actor_hidden_dims,
        critic_hidden_dims=cfg.critic_hidden_dims,
        activation=cfg.activation,
        init_noise_std=cfg.init_noise_std,
    ).to(env.device)
    checkpoint = torch.load(checkpoint_path, map_location=env.device)
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    actor_critic.eval()
    return actor_critic.act_inference, checkpoint.get("iter")


def main():
    args = get_args(test=True)
    args.task = TASK_NAME
    headless = os.environ.get("G1_PERCEPTIVE_SUITE_HEADLESS", "0") == "1"
    args.headless = headless
    if args.sim_device.startswith("cpu"):
        print("[perceptive-suite][warning] Use GPU PhysX for capability claims")

    checkpoint_path = os.environ.get(
        "G1_PERCEPTIVE_SUITE_CHECKPOINT",
        os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs/g1_walking_rough_perceptive/oracle_heightmap_34/model_10000.pt",
        ),
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    perception_mode = os.environ.get(
        "G1_PERCEPTIVE_SUITE_MODE", "oracle"
    ).lower()
    if perception_mode not in ("oracle", "noisy"):
        raise ValueError("G1_PERCEPTIVE_SUITE_MODE must be oracle or noisy")
    level = int(os.environ.get("G1_PERCEPTIVE_SUITE_LEVEL", "9"))
    if not 0 <= level <= 9:
        raise ValueError("G1_PERCEPTIVE_SUITE_LEVEL must be in [0, 9]")
    replicas = int(
        os.environ.get("G1_PERCEPTIVE_SUITE_REPLICAS", "8" if headless else "1")
    )
    command_x = float(os.environ.get("G1_PERCEPTIVE_SUITE_VX", "0.25"))
    focus_duration_s = float(
        os.environ.get("G1_PERCEPTIVE_SUITE_FOCUS_DURATION", "4.0")
    )
    default_duration = 10.0 if headless else focus_duration_s * len(VISUAL_ORDER)
    duration_s = float(
        os.environ.get("G1_PERCEPTIVE_SUITE_DURATION", str(default_duration))
    )
    evaluation_seed = int(
        os.environ.get("G1_PERCEPTIVE_SUITE_SEED", "33")
    )

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    # Use the blind baseline's seed by default so both policies see exactly the
    # same procedurally generated terrain layouts during comparison.
    env_cfg.seed = evaluation_seed
    train_cfg.seed = evaluation_seed
    env_cfg.env.num_envs = len(TERRAIN_NAMES) * replicas
    env_cfg.env.episode_length_s = max(duration_s + 5.0, 30.0)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = len(TERRAIN_NAMES)
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.perception.mode = perception_mode
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_x = [command_x, command_x]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    env.cfg.terrain.curriculum = False
    env.terrain_levels[:] = level
    env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
    obs, _ = env.reset()
    policy, checkpoint_iteration = load_policy(env, train_cfg, checkpoint_path)

    difficulty = level / 10.0
    terrain_parameters = {
        "level": level,
        "difficulty": difficulty,
        "rough_amplitude_m": 0.01 + 0.045 * difficulty,
        "slope_gradient": 0.04 + 0.20 * difficulty,
        "slope_degrees": math.degrees(math.atan(0.04 + 0.20 * difficulty)),
        "step_height_m": 0.025 + 0.11 * difficulty,
        "obstacle_height_m": 0.02 + 0.09 * difficulty,
    }
    print("[perceptive-suite] checkpoint={}".format(checkpoint_path))
    print(
        "[perceptive-suite] iteration={} mode={} parameters={}".format(
            checkpoint_iteration, perception_mode, terrain_parameters
        )
    )

    falls = torch.zeros(env.num_envs, device=env.device)
    tracking_error_sum = torch.zeros_like(falls)
    orientation_error_sum = torch.zeros_like(falls)
    slip_speed_sum = torch.zeros_like(falls)
    forward_progress = torch.zeros_like(falls)
    height_error_sum = torch.zeros_like(falls)
    step_count = 0
    last_focus = -1

    with torch.inference_mode():
        while step_count * env.dt < duration_s:
            if not headless:
                focus_slot = min(
                    int(step_count * env.dt / focus_duration_s),
                    len(VISUAL_ORDER) - 1,
                )
                if focus_slot != last_focus:
                    terrain_id = VISUAL_ORDER[focus_slot]
                    set_camera_for_env(env, terrain_id * replicas)
                    print(
                        "[perceptive-suite] viewing {} ({}/{})".format(
                            TERRAIN_NAMES[terrain_id], focus_slot + 1,
                            len(VISUAL_ORDER),
                        )
                    )
                    last_focus = focus_slot

            env.commands[:, 0] = command_x
            env.commands[:, 1:] = 0.0
            actions = policy(obs)
            obs, _, _, done, _ = env.step(actions)
            falls += (done.bool() & ~env.time_out_buf).float()
            tracking_error_sum += torch.linalg.vector_norm(
                env.base_lin_vel[:, :2] - env.commands[:, :2], dim=1
            )
            orientation_error_sum += torch.linalg.vector_norm(
                env.projected_gravity[:, :2], dim=1
            )
            contact = env.contact_forces[:, env.feet_indices, 2] > 1.0
            slip_speed_sum += torch.sum(
                torch.linalg.vector_norm(env.feet_vel[:, :, :2], dim=2) * contact,
                dim=1,
            )
            forward_progress += env.base_lin_vel[:, 0] * env.dt
            height_error_sum += torch.mean(
                torch.abs(
                    env.last_actor_terrain_obs - env.last_actor_terrain_truth
                ),
                dim=1,
            )
            step_count += 1

    results = []
    for terrain_id, terrain_name in enumerate(TERRAIN_NAMES):
        mask = env.terrain_types == terrain_id
        terrain_falls = falls[mask]
        results.append(
            {
                "terrain": terrain_name,
                "fall_free_rate": torch.mean((terrain_falls == 0).float()).item(),
                "mean_falls": torch.mean(terrain_falls).item(),
                "mean_tracking_error_mps": torch.mean(
                    tracking_error_sum[mask] / step_count
                ).item(),
                "mean_orientation_error": torch.mean(
                    orientation_error_sum[mask] / step_count
                ).item(),
                "mean_contact_slip_speed_mps": torch.mean(
                    slip_speed_sum[mask] / step_count
                ).item(),
                "mean_integrated_forward_progress_m": torch.mean(
                    forward_progress[mask]
                ).item(),
                "mean_heightmap_error_m": torch.mean(
                    height_error_sum[mask] / step_count
                ).item(),
            }
        )

    output = {
        "checkpoint": checkpoint_path,
        "checkpoint_iteration": checkpoint_iteration,
        "perception_mode": perception_mode,
        "evaluation_seed": evaluation_seed,
        "sim_device": args.sim_device,
        "terrain_parameters": terrain_parameters,
        "replicas_per_terrain": replicas,
        "command_x_mps": command_x,
        "duration_s": step_count * env.dt,
        "results": results,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
