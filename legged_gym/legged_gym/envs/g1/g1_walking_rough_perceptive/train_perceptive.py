"""Train the perceptive rough-terrain policy from a rough walking checkpoint."""

import os

import isaacgym  # must be imported before torch
import torch
import wandb

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


TASK_NAME = "g1_walking_rough_perceptive"


def transfer_rough_policy(runner, checkpoint_path, transfer_critic=True):
    """Transfer the blind actor exactly and optionally reuse its terrain critic."""
    checkpoint = torch.load(checkpoint_path, map_location=runner.device)
    source = checkpoint["model_state_dict"]
    target = runner.alg.actor_critic.state_dict()
    actor_mapping = {
        "std": "std",
        "actor.0.weight": "actor.proprio_input.weight",
        "actor.0.bias": "actor.proprio_input.bias",
        "actor.2.weight": "actor.proprio_hidden.weight",
        "actor.2.bias": "actor.proprio_hidden.bias",
        "actor.4.weight": "actor.action_head.weight",
        "actor.4.bias": "actor.action_head.bias",
    }

    transferred_actor = []
    for source_key, target_key in actor_mapping.items():
        if source_key not in source:
            continue
        if target_key not in target or source[source_key].shape != target[target_key].shape:
            continue
        target[target_key] = source[source_key].to(
            device=target[target_key].device,
            dtype=target[target_key].dtype,
        )
        transferred_actor.append(source_key)

    missing = set(actor_mapping).difference(transferred_actor)
    if missing:
        raise RuntimeError(
            "Rough actor warm start missed parameters: {}".format(sorted(missing))
        )

    transferred_critic = []
    if transfer_critic:
        for key, value in source.items():
            if not key.startswith("critic."):
                continue
            if key not in target or target[key].shape != value.shape:
                continue
            target[key] = value.to(
                device=target[key].device, dtype=target[key].dtype
            )
            transferred_critic.append(key)

    runner.alg.actor_critic.load_state_dict(target)
    print(
        "[warm-start] loaded {} actor tensors and {} critic tensors from {}".format(
            len(transferred_actor), len(transferred_critic), checkpoint_path
        )
    )
    print(
        "[warm-start] terrain encoder is new; its zero adapter preserves the source actor initially"
    )
    return transferred_actor, transferred_critic


def train(args):
    args.task = TASK_NAME
    wandb_mode = os.environ.get("WANDB_MODE", "offline")
    wandb.init(
        project=args.wandb,
        name=args.run_name,
        entity=args.entity,
        mode=wandb_mode,
    )

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    if "G1_PERCEPTIVE_NUM_ENVS" in os.environ:
        env_cfg.env.num_envs = int(os.environ["G1_PERCEPTIVE_NUM_ENVS"])
    if "G1_PERCEPTIVE_MAX_ITERATIONS" in os.environ:
        train_cfg.runner.max_iterations = int(
            os.environ["G1_PERCEPTIVE_MAX_ITERATIONS"]
        )
    if "G1_PERCEPTIVE_MODE" in os.environ:
        env_cfg.perception.mode = os.environ["G1_PERCEPTIVE_MODE"].lower()

    env, _ = task_registry.make_env(
        name=TASK_NAME, args=args, env_cfg=env_cfg
    )
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=None,
        args=args,
        train_cfg=train_cfg,
    )

    disable_warm_start = (
        os.environ.get("G1_PERCEPTIVE_DISABLE_WARM_START", "0") == "1"
    )
    if train_cfg.runner.warm_start and not args.resume and not disable_warm_start:
        checkpoint_path = os.environ.get(
            "G1_PERCEPTIVE_WARM_START", train_cfg.runner.warm_start_path
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "Rough warm-start checkpoint is not ready: {}. Finish the blind "
                "rough run or set G1_PERCEPTIVE_WARM_START to an intermediate "
                "checkpoint.".format(checkpoint_path)
            )
        transfer_critic = (
            os.environ.get("G1_PERCEPTIVE_TRANSFER_CRITIC", "1") == "1"
        )
        transfer_rough_policy(
            runner, checkpoint_path, transfer_critic=transfer_critic
        )

    iterations = train_cfg.runner.max_iterations
    if args.resume and args.resume_stop_at_max:
        iterations = max(
            iterations - runner.current_learning_iteration, 0
        )
    runner.learn(
        num_learning_iterations=iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    train(get_args())

