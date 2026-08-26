import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import wandb

def train(args):
    wandb.init(project=args.wandb, name=args.run_name, entity=args.entity)

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    if args.resume and args.resume_stop_at_max:
        num_learning_iterations = max(train_cfg.runner.max_iterations - ppo_runner.current_learning_iteration, 0)
    else:
        num_learning_iterations = train_cfg.runner.max_iterations
    # ppo_runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True)
    ppo_runner.learn(num_learning_iterations=num_learning_iterations)

if __name__ == '__main__':
    args = get_args()
    train(args)
