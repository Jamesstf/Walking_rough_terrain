"""Create a tiny CPU rough-terrain simulation and validate observation/reward shapes."""

import isaacgym  # must be imported before torch
from isaacgym import gymapi
import torch

from .g1_walking_rough import G1WalkingRough
from .g1_walking_rough_config import G1WalkingRoughCfg


def main():
    cfg = G1WalkingRoughCfg()
    cfg.env.num_envs = 8
    cfg.terrain.num_rows = 2
    cfg.terrain.num_cols = 4
    cfg.terrain.max_init_terrain_level = 1
    cfg.domain_rand.push_robots = False

    sim_params = gymapi.SimParams()
    sim_params.dt = cfg.sim.dt
    sim_params.substeps = cfg.sim.substeps
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(*cfg.sim.gravity)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = False
    sim_params.physx.num_position_iterations = cfg.sim.physx.num_position_iterations
    sim_params.physx.num_velocity_iterations = cfg.sim.physx.num_velocity_iterations
    sim_params.physx.contact_offset = cfg.sim.physx.contact_offset
    sim_params.physx.rest_offset = cfg.sim.physx.rest_offset

    env = G1WalkingRough(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=gymapi.SIM_PHYSX,
        sim_device="cpu",
        headless=True,
    )
    obs, privileged = env.reset()
    assert obs.shape == (cfg.env.num_envs, 141), obs.shape
    assert privileged.shape == (cfg.env.num_envs, 113), privileged.shape
    for _ in range(5):
        obs, privileged, reward, done, _ = env.step(
            torch.zeros(cfg.env.num_envs, cfg.env.num_actions, device=env.device)
        )
        assert torch.isfinite(obs).all()
        assert torch.isfinite(privileged).all()
        assert torch.isfinite(reward).all()
    print(
        f"rough env validation passed: obs={tuple(obs.shape)} "
        f"critic={tuple(privileged.shape)} reward_mean={reward.mean().item():.4f} "
        f"resets={int(done.sum().item())}"
    )


if __name__ == "__main__":
    main()
