"""Train rough-terrain G1 walking, warm-starting only the flat-policy actor."""

import os

import isaacgym  # must be imported before torch
import torch
import wandb

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


TASK_NAME = "g1_walking_rough"


def transfer_flat_actor(runner, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=runner.device)
    source = checkpoint["model_state_dict"]
    target = runner.alg.actor_critic.state_dict()
    transferred = []
    for key, value in source.items():
        if not (key.startswith("actor.") or key == "std"):
            continue
        if key not in target or target[key].shape != value.shape:
            continue
        target[key] = value.to(device=target[key].device, dtype=target[key].dtype)
        transferred.append(key)
    runner.alg.actor_critic.load_state_dict(target)
    expected = {"std", "actor.0.weight", "actor.0.bias", "actor.2.weight", "actor.2.bias", "actor.4.weight", "actor.4.bias"}
    missing = expected.difference(transferred)
    if missing:
        raise RuntimeError(f"Flat actor warm start missed parameters: {sorted(missing)}")
    print(f"[warm-start] loaded {len(transferred)} actor tensors from {checkpoint_path}")
    print("[warm-start] rough-terrain critic and optimizer remain freshly initialized")


def train(args):
    args.task = TASK_NAME
    wandb_mode = os.environ.get("WANDB_MODE", "offline")
    wandb.init(project=args.wandb, name=args.run_name, entity=args.entity, mode=wandb_mode)

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    if "G1_ROUGH_NUM_ENVS" in os.environ:
        env_cfg.env.num_envs = int(os.environ["G1_ROUGH_NUM_ENVS"])
    if "G1_ROUGH_MAX_ITERATIONS" in os.environ:
        train_cfg.runner.max_iterations = int(os.environ["G1_ROUGH_MAX_ITERATIONS"])
    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=None,
        args=args,
        train_cfg=train_cfg,
    )

    disable_warm_start = os.environ.get("G1_ROUGH_DISABLE_WARM_START", "0") == "1"
    if train_cfg.runner.warm_start and not args.resume and not disable_warm_start:
        checkpoint_path = os.environ.get(
            "G1_ROUGH_WARM_START",
            train_cfg.runner.warm_start_path,
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint_path}")
        transfer_flat_actor(runner, checkpoint_path)

    iterations = train_cfg.runner.max_iterations
    if args.resume and args.resume_stop_at_max:
        iterations = max(iterations - runner.current_learning_iteration, 0)
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    train(get_args())
