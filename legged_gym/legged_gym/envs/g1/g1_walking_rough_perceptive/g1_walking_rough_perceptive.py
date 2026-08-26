"""G1 rough locomotion with a simulated local height-map sensor."""

import torch

from legged_gym.envs.g1.g1_walking_rough.g1_walking_rough import G1WalkingRough

from .g1_walking_rough_perceptive_config import G1WalkingRoughPerceptiveCfg
from .terrain_perception import TerrainPerceptionModel


class G1WalkingRoughPerceptive(G1WalkingRough):
    def __init__(
        self, cfg: G1WalkingRoughPerceptiveCfg, sim_params, physics_engine,
        sim_device, headless
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg: G1WalkingRoughPerceptiveCfg

    def _init_buffers(self):
        super()._init_buffers()
        self.terrain_perception = TerrainPerceptionModel(
            cfg=self.cfg.perception,
            num_envs=self.num_envs,
            num_points=self.cfg.env.num_terrain_observations,
            device=self.device,
        )
        self.last_actor_terrain_truth = torch.zeros(
            self.num_envs,
            self.cfg.env.num_terrain_observations,
            device=self.device,
        )
        self.last_actor_terrain_obs = torch.zeros_like(
            self.last_actor_terrain_truth
        )

    def _raw_terrain_height_residual(self):
        return torch.clip(
            self.root_states[:, 2].unsqueeze(1)
            - self.cfg.rewards.base_height_target
            - self.measured_heights,
            -self.cfg.perception.clip_height,
            self.cfg.perception.clip_height,
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0 or not hasattr(self, "terrain_perception"):
            return
        truth = self._raw_terrain_height_residual()
        self.terrain_perception.reset(env_ids, truth)
        self.last_actor_terrain_truth[env_ids] = truth[env_ids]
        self.last_actor_terrain_obs[env_ids] = truth[env_ids]

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
                "Perceptive proprioceptive frame is {}D, expected {}D".format(
                    actor_now.shape[1], self.cfg.env.num_single_obs
                )
            )
        if critic_now.shape[1] != self.cfg.env.single_num_privileged_obs:
            raise RuntimeError(
                "Perceptive critic frame is {}D, expected {}D".format(
                    critic_now.shape[1], self.cfg.env.single_num_privileged_obs
                )
            )

        if self.add_noise:
            actor_now += (
                2.0 * torch.rand_like(actor_now) - 1.0
            ) * self.noise_scale_vec
        self.obs_history.append(actor_now)
        self.critic_history.append(critic_now)

        proprioception = torch.stack(
            tuple(self.obs_history), dim=1
        ).reshape(self.num_envs, -1)
        truth = self._raw_terrain_height_residual()
        perceived = self.terrain_perception.observe(truth)
        terrain_scaled = perceived * self.obs_scales.height_measurements

        self.last_actor_terrain_truth.copy_(truth)
        self.last_actor_terrain_obs.copy_(perceived)
        self.obs_buf = torch.cat((proprioception, terrain_scaled), dim=-1)
        self.privileged_obs_buf = torch.cat(tuple(self.critic_history), dim=1)

        if self.obs_buf.shape[1] != self.cfg.env.num_observations:
            raise RuntimeError(
                "Perceptive actor observation is {}D, expected {}D".format(
                    self.obs_buf.shape[1], self.cfg.env.num_observations
                )
            )

