# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


def split_and_pad_trajectories(tensor, dones):
    """Split trajectories at done indices.

    The input tensor is expected to have shape:
        [time, num_envs, additional_dims]

    The function returns padded trajectories and a mask indicating valid
    trajectory entries.
    """
    dones = dones.clone()
    dones[-1] = 1

    # Permute buffers to [num_envs, num_transitions_per_env, ...] before flattening.
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Compute trajectory lengths from done indices.
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()

    # Split the tensor into individual trajectories.
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)

    # Add one full-length zero trajectory so pad_sequence always pads to rollout length.
    trajectories = (*trajectories, torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device))

    # Pad trajectories and remove the temporary full-length zero trajectory.
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
    padded_trajectories = padded_trajectories[:, :-1]

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0],
                                                                    trajectories.shape[-1]).transpose(1, 0)
