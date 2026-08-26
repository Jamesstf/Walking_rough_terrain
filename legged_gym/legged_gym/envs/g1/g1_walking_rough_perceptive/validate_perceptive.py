"""Validate the sensor, observation layout and actor network on CPU PhysX."""

from types import SimpleNamespace

import isaacgym  # must be imported before torch
from isaacgym import gymapi
import torch

from rsl_rl.modules import ActorCriticPerceptive

from .g1_walking_rough_perceptive import G1WalkingRoughPerceptive
from .g1_walking_rough_perceptive_config import G1WalkingRoughPerceptiveCfg


def make_cpu_sim_params(cfg):
    sim_params = gymapi.SimParams()
    sim_params.dt = cfg.sim.dt
    sim_params.substeps = cfg.sim.substeps
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(*cfg.sim.gravity)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = False
    sim_params.physx.num_position_iterations = (
        cfg.sim.physx.num_position_iterations
    )
    sim_params.physx.num_velocity_iterations = (
        cfg.sim.physx.num_velocity_iterations
    )
    sim_params.physx.contact_offset = cfg.sim.physx.contact_offset
    sim_params.physx.rest_offset = cfg.sim.physx.rest_offset
    return sim_params


def main():
    cfg = G1WalkingRoughPerceptiveCfg()
    cfg.env.num_envs = 8
    cfg.terrain.num_rows = 2
    cfg.terrain.num_cols = 4
    cfg.terrain.max_init_terrain_level = 1
    cfg.domain_rand.push_robots = False
    cfg.perception.mode = "noisy"

    env = G1WalkingRoughPerceptive(
        cfg=cfg,
        sim_params=make_cpu_sim_params(cfg),
        physics_engine=gymapi.SIM_PHYSX,
        sim_device="cpu",
        headless=True,
    )
    obs, privileged = env.reset()
    assert obs.shape == (cfg.env.num_envs, 204), obs.shape
    assert privileged.shape == (cfg.env.num_envs, 113), privileged.shape
    assert torch.isfinite(obs).all()
    assert torch.isfinite(privileged).all()

    model = ActorCriticPerceptive(
        num_actor_obs=204,
        num_critic_obs=113,
        num_actions=cfg.env.num_actions,
        proprio_obs_dim=141,
        terrain_obs_dim=63,
        terrain_encoder_dims=[128, 32],
        actor_hidden_dims=[512, 128],
        critic_hidden_dims=[512, 128],
        activation="elu",
        device="cpu",
        args=SimpleNamespace(),
    )
    with torch.no_grad():
        actions = model.act_inference(obs)
    assert actions.shape == (cfg.env.num_envs, cfg.env.num_actions)
    assert torch.isfinite(actions).all()

    for _ in range(5):
        obs, privileged, reward, done, _ = env.step(actions * 0.0)
        assert torch.isfinite(obs).all()
        assert torch.isfinite(privileged).all()
        assert torch.isfinite(reward).all()

    noisy_mae = torch.mean(
        torch.abs(
            env.last_actor_terrain_obs - env.last_actor_terrain_truth
        )
    ).item()
    env.terrain_perception.set_mode("oracle")
    env.compute_observations()
    oracle_mae = torch.mean(
        torch.abs(
            env.last_actor_terrain_obs - env.last_actor_terrain_truth
        )
    ).item()
    if oracle_mae > 1.0e-7:
        raise RuntimeError("Oracle height map is not exact: MAE={}".format(oracle_mae))
    print(
        "perceptive validation passed: obs={} critic={} actions={} "
        "noisy_mae={:.6f}m oracle_mae={:.6f}m resets={}".format(
            tuple(obs.shape),
            tuple(privileged.shape),
            tuple(actions.shape),
            noisy_mae,
            oracle_mae,
            int(done.sum().item()),
        )
    )


if __name__ == "__main__":
    main()

