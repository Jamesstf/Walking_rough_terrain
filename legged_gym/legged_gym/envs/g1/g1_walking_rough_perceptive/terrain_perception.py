"""Fast batched simulation of a local height-map sensor."""

import torch


class TerrainPerceptionModel:
    """Apply delay, bias, white noise, quantization and point dropout.

    Inputs and outputs are height residuals in metres. Scaling for the policy
    observation is deliberately kept in the environment.
    """

    VALID_MODES = ("oracle", "noisy")

    def __init__(self, cfg, num_envs, num_points, device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_points = num_points
        self.device = device
        self.mode = cfg.mode
        if self.mode not in self.VALID_MODES:
            raise ValueError("Unsupported perception mode: {}".format(self.mode))
        if cfg.max_delay_steps < 0:
            raise ValueError("max_delay_steps must be non-negative")

        self.buffer_length = cfg.max_delay_steps + 1
        self.history = torch.zeros(
            self.buffer_length,
            num_envs,
            num_points,
            dtype=torch.float,
            device=device,
        )
        self.write_index = 0
        self.env_indices = torch.arange(num_envs, device=device)
        self.delay_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.noise_std = torch.zeros(num_envs, 1, device=device)
        self.height_bias = torch.zeros(num_envs, 1, device=device)
        self.last_valid_mask = torch.ones(
            num_envs, num_points, dtype=torch.bool, device=device
        )

    def set_mode(self, mode):
        if mode not in self.VALID_MODES:
            raise ValueError("Unsupported perception mode: {}".format(mode))
        self.mode = mode

    def reset(self, env_ids, current_heights):
        if len(env_ids) == 0:
            return
        count = len(env_ids)
        if self.cfg.randomize_delay:
            self.delay_steps[env_ids] = torch.randint(
                low=0,
                high=self.cfg.max_delay_steps + 1,
                size=(count,),
                device=self.device,
            )
        else:
            self.delay_steps[env_ids] = self.cfg.max_delay_steps

        minimum_std, maximum_std = self.cfg.noise_std_range
        self.noise_std[env_ids] = minimum_std + (
            maximum_std - minimum_std
        ) * torch.rand(count, 1, device=self.device)
        self.height_bias[env_ids] = (
            2.0 * torch.rand(count, 1, device=self.device) - 1.0
        ) * self.cfg.max_height_bias
        self.history[:, env_ids, :] = current_heights[env_ids].unsqueeze(0)
        self.last_valid_mask[env_ids] = True

    def observe(self, true_heights):
        if true_heights.shape != (self.num_envs, self.num_points):
            raise RuntimeError(
                "Height map has shape {}, expected ({}, {})".format(
                    tuple(true_heights.shape), self.num_envs, self.num_points
                )
            )
        if self.mode == "oracle":
            self.last_valid_mask.fill_(True)
            return torch.clamp(
                true_heights, -self.cfg.clip_height, self.cfg.clip_height
            )

        self.history[self.write_index].copy_(true_heights)
        read_indices = torch.remainder(
            self.write_index - self.delay_steps, self.buffer_length
        )
        delayed = self.history[read_indices, self.env_indices].clone()
        self.write_index = (self.write_index + 1) % self.buffer_length

        noisy = delayed + self.height_bias
        noisy += torch.randn_like(noisy) * self.noise_std
        if self.cfg.quantization > 0.0:
            noisy = torch.round(noisy / self.cfg.quantization) * self.cfg.quantization

        self.last_valid_mask = (
            torch.rand_like(noisy) >= self.cfg.dropout_probability
        )
        fill = torch.full_like(noisy, self.cfg.dropout_fill_value)
        noisy = torch.where(self.last_valid_mask, noisy, fill)
        return torch.clamp(noisy, -self.cfg.clip_height, self.cfg.clip_height)

