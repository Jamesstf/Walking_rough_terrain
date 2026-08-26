"""Run one CPU PPO update for the complete perceptive training pipeline."""

import os
import tempfile
from types import SimpleNamespace

import isaacgym  # must be imported before torch
from isaacgym import gymapi
import torch
import wandb

from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner
from legged_gym.utils.helpers import class_to_dict

from .g1_walking_rough_perceptive import G1WalkingRoughPerceptive
from .g1_walking_rough_perceptive_config import (
    G1WalkingRoughPerceptiveCfg,
    G1WalkingRoughPerceptiveCfgPPO,
)
from .train_perceptive import transfer_rough_policy
from .validate_perceptive import make_cpu_sim_params


def main():
    env_cfg = G1WalkingRoughPerceptiveCfg()
    env_cfg.env.num_envs = 16
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.max_init_terrain_level = 1
    env_cfg.domain_rand.push_robots = False

    train_cfg = G1WalkingRoughPerceptiveCfgPPO()
    train_cfg.runner.num_steps_per_env = 8
    train_cfg.runner.max_iterations = 1
    train_cfg.runner.save_interval = 1
    train_cfg.algorithm.num_learning_epochs = 1
    train_cfg.algorithm.num_mini_batches = 2

    env = G1WalkingRoughPerceptive(
        cfg=env_cfg,
        sim_params=make_cpu_sim_params(env_cfg),
        physics_engine=gymapi.SIM_PHYSX,
        sim_device="cpu",
        headless=True,
    )
    log_dir = tempfile.mkdtemp(prefix="g1_perceptive_smoke_", dir="/tmp")
    args = SimpleNamespace()
    wandb.init(project="g1_perceptive_smoke", mode="disabled")
    runner = OnPolicyRunner(
        env,
        class_to_dict(train_cfg),
        log_dir=log_dir,
        device="cpu",
        args=args,
    )

    # The original flat checkpoint has the same 141D actor structure and can
    # validate transfer before the new rough checkpoint has finished training.
    source_path = G1WalkingRoughPerceptiveCfgPPO.runner.warm_start_path
    if not os.path.isfile(source_path):
        from legged_gym.envs.g1.g1_walking_rough.g1_walking_rough_config import (
            G1WalkingRoughCfgPPO,
        )
        source_path = G1WalkingRoughCfgPPO.runner.warm_start_path
    transfer_rough_policy(runner, source_path, transfer_critic=True)

    checkpoint = torch.load(source_path, map_location="cpu")
    source_model = ActorCritic(
        141,
        50,
        env_cfg.env.num_actions,
        actor_hidden_dims=[512, 128],
        critic_hidden_dims=[512, 128],
        activation="elu",
    )
    source_state = source_model.state_dict()
    compatible = {
        key: value for key, value in checkpoint["model_state_dict"].items()
        if key in source_state and source_state[key].shape == value.shape
    }
    source_model.load_state_dict(compatible, strict=False)
    test_proprio = torch.randn(32, 141)
    test_terrain = torch.randn(32, 63)
    with torch.no_grad():
        source_actions = source_model.act_inference(test_proprio)
        target_actions = runner.alg.actor_critic.act_inference(
            torch.cat((test_proprio, test_terrain), dim=-1)
        )
    transfer_error = torch.max(torch.abs(source_actions - target_actions)).item()
    if transfer_error > 1.0e-6:
        raise RuntimeError(
            "Residual warm start changed actor output: max error={}".format(
                transfer_error
            )
        )

    adapter_before = (
        runner.alg.actor_critic.actor.terrain_adapter.weight.detach().clone()
    )
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
    adapter_after = runner.alg.actor_critic.actor.terrain_adapter.weight.detach()
    adapter_change = torch.mean(
        torch.abs(adapter_after - adapter_before)
    ).item()
    saved_checkpoint = os.path.join(log_dir, "model_1.pt")
    if not os.path.isfile(saved_checkpoint):
        raise RuntimeError(
            "Smoke training did not create {}".format(saved_checkpoint)
        )
    if adapter_change <= 0.0:
        raise RuntimeError("Terrain residual adapter did not receive an update")
    print(
        "perceptive PPO smoke passed: checkpoint={} transfer_max_error={:.3g} "
        "terrain_adapter_change={:.6g}".format(
            saved_checkpoint, transfer_error, adapter_change
        )
    )


if __name__ == "__main__":
    main()

