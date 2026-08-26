"""Run one tiny CPU PPO update to validate the complete rough-training pipeline."""

import os
import tempfile
from types import SimpleNamespace

import isaacgym  # must be imported before torch
from isaacgym import gymapi
import torch
import wandb

from rsl_rl.runners import OnPolicyRunner
from legged_gym.utils.helpers import class_to_dict

from .g1_walking_rough import G1WalkingRough
from .g1_walking_rough_config import G1WalkingRoughCfg, G1WalkingRoughCfgPPO
from .train_rough import transfer_flat_actor


def main():
    env_cfg = G1WalkingRoughCfg()
    env_cfg.env.num_envs = 16
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.max_init_terrain_level = 1
    env_cfg.domain_rand.push_robots = False

    train_cfg = G1WalkingRoughCfgPPO()
    train_cfg.runner.num_steps_per_env = 8
    train_cfg.runner.max_iterations = 1
    train_cfg.runner.save_interval = 1
    train_cfg.algorithm.num_learning_epochs = 1
    train_cfg.algorithm.num_mini_batches = 2

    sim_params = gymapi.SimParams()
    sim_params.dt = env_cfg.sim.dt
    sim_params.substeps = env_cfg.sim.substeps
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(*env_cfg.sim.gravity)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = False
    sim_params.physx.num_position_iterations = env_cfg.sim.physx.num_position_iterations
    sim_params.physx.num_velocity_iterations = env_cfg.sim.physx.num_velocity_iterations
    sim_params.physx.contact_offset = env_cfg.sim.physx.contact_offset
    sim_params.physx.rest_offset = env_cfg.sim.physx.rest_offset

    env = G1WalkingRough(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=gymapi.SIM_PHYSX,
        sim_device="cpu",
        headless=True,
    )
    log_dir = tempfile.mkdtemp(prefix="g1_rough_smoke_", dir="/tmp")
    args = SimpleNamespace()
    wandb.init(project="g1_walking_rough_smoke", mode="disabled")
    runner = OnPolicyRunner(
        env,
        class_to_dict(train_cfg),
        log_dir=log_dir,
        device="cpu",
        args=args,
    )
    transfer_flat_actor(runner, train_cfg.runner.warm_start_path)
    before = runner.alg.actor_critic.actor[0].weight.detach().clone()
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
    after = runner.alg.actor_critic.actor[0].weight.detach()
    checkpoint = os.path.join(log_dir, "model_1.pt")
    if not os.path.isfile(checkpoint):
        raise RuntimeError(f"Smoke training did not create {checkpoint}")
    parameter_change = torch.mean(torch.abs(after - before)).item()
    print(
        f"rough PPO smoke training passed: checkpoint={checkpoint} "
        f"actor_parameter_change={parameter_change:.6g}"
    )


if __name__ == "__main__":
    main()
