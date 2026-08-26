"""Visualize a rough-walking checkpoint on one selected terrain family."""

import os

import isaacgym  # must be imported before torch
import torch

from rsl_rl.modules import ActorCritic
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


TASK_NAME = "g1_walking_rough"
TERRAIN_PROPORTIONS = {
    "flat": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "random_rough": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "slope_up": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "slope_down": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "stairs_up": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "stairs_down": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "obstacles": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}


def load_policy(env, train_cfg, checkpoint_path):
    policy_cfg = train_cfg.policy
    actor_critic = ActorCritic(
        env.num_obs,
        env.num_privileged_obs,
        env.num_actions,
        actor_hidden_dims=policy_cfg.actor_hidden_dims,
        critic_hidden_dims=policy_cfg.critic_hidden_dims,
        activation=policy_cfg.activation,
        init_noise_std=policy_cfg.init_noise_std,
    ).to(env.device)
    checkpoint = torch.load(checkpoint_path, map_location=env.device)
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    actor_critic.eval()
    return actor_critic.act_inference


def main():
    args = get_args(test=True)
    args.task = TASK_NAME
    args.headless = os.environ.get("G1_ROUGH_PLAY_HEADLESS", "0") == "1"

    checkpoint_path = os.environ.get(
        "G1_ROUGH_PLAY_CHECKPOINT",
        os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs/g1_walking_rough/rough_warmstart_33/model_10000.pt",
        ),
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "Rough walking checkpoint not found: {}".format(checkpoint_path)
        )

    terrain_name = os.environ.get(
        "G1_ROUGH_PLAY_TERRAIN", "random_rough"
    ).lower()
    if terrain_name not in TERRAIN_PROPORTIONS:
        raise ValueError(
            "G1_ROUGH_PLAY_TERRAIN must be one of {}".format(
                sorted(TERRAIN_PROPORTIONS)
            )
        )
    terrain_level = int(os.environ.get("G1_ROUGH_PLAY_LEVEL", "5"))
    if not 0 <= terrain_level <= 9:
        raise ValueError("G1_ROUGH_PLAY_LEVEL must be in [0, 9]")

    command_x = float(os.environ.get("G1_ROUGH_PLAY_VX", "0.35"))
    command_y = float(os.environ.get("G1_ROUGH_PLAY_VY", "0.0"))
    command_yaw = float(os.environ.get("G1_ROUGH_PLAY_YAW", "0.0"))
    duration_s = float(os.environ.get("G1_ROUGH_PLAY_DURATION", "0.0"))

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 30.0
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain.terrain_proportions = TERRAIN_PROPORTIONS[terrain_name]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_x = [command_x, command_x]
    env_cfg.commands.ranges.lin_vel_y = [command_y, command_y]
    env_cfg.commands.ranges.ang_vel_yaw = [command_yaw, command_yaw]

    env, _ = task_registry.make_env(
        name=TASK_NAME, args=args, env_cfg=env_cfg
    )
    # Generate all ten curriculum rows, then place the single robot on the
    # requested row and freeze terrain curriculum for repeatable playback.
    env.cfg.terrain.curriculum = False
    env.terrain_levels[:] = terrain_level
    env.env_origins[:] = env.terrain_origins[terrain_level, 0]
    obs, _ = env.reset()

    origin = env.env_origins[0].detach().cpu().numpy()
    env.set_camera(
        [origin[0] - 3.2, origin[1] - 4.2, origin[2] + 2.6],
        [origin[0] + 1.0, origin[1], origin[2] + 0.75],
    )
    policy = load_policy(env, train_cfg, checkpoint_path)

    print("[play] checkpoint={}".format(checkpoint_path))
    print(
        "[play] terrain={} level={} command=[{:.2f}, {:.2f}, {:.2f}]".format(
            terrain_name, terrain_level, command_x, command_y, command_yaw
        )
    )
    print("[play] close the viewer or press ESC to stop; press V to pause rendering")

    step_count = 0
    fall_count = 0
    tracking_error_sum = 0.0
    with torch.inference_mode():
        while True:
            env.commands[:, 0] = command_x
            env.commands[:, 1] = command_y
            env.commands[:, 2] = command_yaw
            actions = policy(obs)
            obs, _, _, done, _ = env.step(actions)
            tracking_error_sum += torch.linalg.vector_norm(
                env.base_lin_vel[0, :2] - env.commands[0, :2]
            ).item()
            step_count += 1
            if done[0] and not env.time_out_buf[0]:
                fall_count += 1
            if step_count % 250 == 0:
                print(
                    "[play] time={:.1f}s mean_tracking_error={:.3f}m/s falls={}".format(
                        step_count * env.dt,
                        tracking_error_sum / step_count,
                        fall_count,
                    )
                )
            if duration_s > 0.0 and step_count * env.dt >= duration_s:
                print(
                    "[play] completed {:.1f}s, mean_tracking_error={:.3f}m/s falls={}".format(
                        step_count * env.dt,
                        tracking_error_sum / step_count,
                        fall_count,
                    )
                )
                break


if __name__ == "__main__":
    main()
