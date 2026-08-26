"""G1 rough-terrain environment with proprioceptive actor and terrain-aware critic."""

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.g1.g1_walking_unitree.g1_walking_unitree import G1WalkingUnitree

from .g1_walking_rough_config import G1WalkingRoughCfg
from .rough_terrain import G1RoughTerrain


class G1WalkingRough(G1WalkingUnitree):
    def __init__(self, cfg: G1WalkingRoughCfg, sim_params, physics_engine, sim_device, headless):
        self._rough_initializing = True
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._rough_initializing = False
        self.cfg: G1WalkingRoughCfg

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym simulation")

        self.terrain = G1RoughTerrain(self.cfg.terrain, self.num_envs)
        if self.cfg.terrain.mesh_type == "trimesh":
            self._create_trimesh()
        elif self.cfg.terrain.mesh_type == "heightfield":
            self._create_heightfield()
        elif self.cfg.terrain.mesh_type == "plane":
            self._create_ground_plane()
        else:
            raise ValueError(f"Unsupported terrain mesh type: {self.cfg.terrain.mesh_type}")
        self._create_envs()

    def _init_buffers(self):
        super()._init_buffers()
        self.measured_heights = self._get_heights()

    def _post_physics_step_callback(self):
        # Retain the original phase and command logic, then refresh local terrain.
        super()._post_physics_step_callback()
        self.measured_heights = self._get_heights()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # The flat G1 task has a custom reset that omits terrain curriculum.
        # Call the generic implementation explicitly; dynamic dispatch still
        # retains G1's DOF/root randomization methods.
        curriculum_enabled = self.cfg.terrain.curriculum
        if self._rough_initializing:
            self.cfg.terrain.curriculum = False
        try:
            LeggedRobot.reset_idx(self, env_ids)
        finally:
            self.cfg.terrain.curriculum = curriculum_enabled
        if hasattr(self, "height_points"):
            current_heights = self._get_heights()
            self.measured_heights[env_ids] = current_heights[env_ids]
        if hasattr(self, "obs_history"):
            for frame in self.obs_history:
                frame[env_ids] = 0.0
            for frame in self.critic_history:
                frame[env_ids] = 0.0

    def _terrain_height_observation(self):
        relative_height = (
            self.root_states[:, 2].unsqueeze(1)
            - self.cfg.rewards.base_height_target
            - self.measured_heights
        )
        return torch.clip(relative_height, -0.50, 0.50) * self.obs_scales.height_measurements

    def compute_observations(self):
        self._update_gait_phase()
        sin_phase = torch.sin(2.0 * torch.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2.0 * torch.pi * self.phase).unsqueeze(1)
        command_obs = self.commands[:, :3] * self.commands_scale

        actor_now = torch.cat(
            (
                command_obs,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                sin_phase,
                cos_phase,
            ),
            dim=-1,
        )
        critic_now = torch.cat(
            (
                command_obs,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                sin_phase,
                cos_phase,
                self._terrain_height_observation(),
            ),
            dim=-1,
        )
        if actor_now.shape[1] != self.cfg.env.num_single_obs:
            raise RuntimeError(
                f"Rough actor frame is {actor_now.shape[1]}D, expected "
                f"{self.cfg.env.num_single_obs}D"
            )
        if critic_now.shape[1] != self.cfg.env.single_num_privileged_obs:
            raise RuntimeError(
                f"Rough critic frame is {critic_now.shape[1]}D, expected "
                f"{self.cfg.env.single_num_privileged_obs}D"
            )

        if self.add_noise:
            actor_now += (2.0 * torch.rand_like(actor_now) - 1.0) * self.noise_scale_vec
        self.obs_history.append(actor_now)
        self.critic_history.append(critic_now)
        self.obs_buf = torch.stack(tuple(self.obs_history), dim=1).reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat(tuple(self.critic_history), dim=1)

    def _sample_height_at_world_xy(self, points_xy):
        points = ((points_xy + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[..., 0].reshape(-1), 0, self.height_samples.shape[0] - 2)
        py = torch.clip(points[..., 1].reshape(-1), 0, self.height_samples.shape[1] - 2)
        heights = torch.minimum(
            self.height_samples[px, py],
            self.height_samples[px + 1, py],
        )
        heights = torch.minimum(heights, self.height_samples[px, py + 1])
        return heights.reshape(points_xy.shape[:-1]) * self.terrain.cfg.vertical_scale

    def _reward_base_height(self):
        center_index = self.cfg.terrain.center_height_index
        terrain_under_base = self.measured_heights[:, center_index]
        clearance = self.root_states[:, 2] - terrain_under_base
        return torch.square(clearance - self.cfg.rewards.base_height_target)

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        terrain_height = self._sample_height_at_world_xy(self.feet_pos[:, :, :2])
        clearance = self.feet_pos[:, :, 2] - terrain_height
        error = torch.square(clearance - self.cfg.rewards.swing_height_target) * ~contact
        return torch.sum(error, dim=1)

    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        tangential_speed_sq = torch.sum(torch.square(self.feet_vel[:, :, :2]), dim=2)
        return torch.sum(tangential_speed_sq * contact, dim=1)

    def _reward_feet_stumble(self):
        horizontal_force = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        vertical_force = torch.abs(self.contact_forces[:, self.feet_indices, 2])
        return torch.sum((horizontal_force > 4.0 * vertical_force).float(), dim=1)

    def _reward_feet_contact_forces(self):
        forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        excess = torch.clip(forces - self.cfg.rewards.max_contact_force, min=0.0)
        return torch.sum(excess, dim=1)
