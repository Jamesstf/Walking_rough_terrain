"""MuJoCo raycast height-map sensor with the training-time degradation model."""

from __future__ import annotations

from collections import deque

import numpy as np

from deploy.astar_mpc_walking.walking_policy import yaw_from_quaternion


HEIGHT_POINTS_X = np.arange(-0.8, 0.81, 0.2, dtype=np.float64)
HEIGHT_POINTS_Y = np.arange(-0.6, 0.61, 0.2, dtype=np.float64)


class MuJoCoHeightmapSensor:
    """Sample 9 x 7 terrain rays and emit 63 height residuals in metres."""

    def __init__(self, mujoco, model, data, seed=33, mode="noisy"):
        if mode not in ("oracle", "noisy"):
            raise ValueError("mode must be oracle or noisy")
        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        grid_x, grid_y = np.meshgrid(HEIGHT_POINTS_X, HEIGHT_POINTS_Y, indexing="ij")
        self.local_points = np.column_stack(
            (grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size))
        )
        self.geom_group = np.zeros(6, dtype=np.uint8)
        self.geom_group[3] = 1
        self.terrain_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "rough_terrain"
        )
        if self.terrain_geom_id < 0:
            raise RuntimeError("rough_terrain geom was not found")
        self.max_delay_steps = 5
        self.history = deque(maxlen=self.max_delay_steps + 1)
        self.delay_steps = 0
        self.noise_std = 0.0
        self.height_bias = 0.0
        self.last_valid_mask = np.ones(63, dtype=bool)
        self.last_world_points = np.zeros((63, 3), dtype=np.float64)
        self.last_true_residual = np.zeros(63, dtype=np.float64)
        self.last_observation = np.zeros(63, dtype=np.float64)
        self.ray_misses = 0

    def _raycast_truth(self, qpos):
        yaw = yaw_from_quaternion(qpos[3:7])
        cosine, sine = np.cos(yaw), np.sin(yaw)
        local_x, local_y = self.local_points[:, 0], self.local_points[:, 1]
        world_x = qpos[0] + cosine * local_x - sine * local_y
        world_y = qpos[1] + sine * local_x + cosine * local_y
        ray_start_z = float(qpos[2]) + 1.0
        direction = np.array((0.0, 0.0, -1.0), dtype=np.float64)
        heights = np.empty(63, dtype=np.float64)
        misses = 0
        for index in range(63):
            point = np.array((world_x[index], world_y[index], ray_start_z), dtype=np.float64)
            geom_id = np.array((-1,), dtype=np.int32)
            distance = self.mujoco.mj_ray(
                self.model, self.data, point, direction, self.geom_group,
                1, -1, geom_id,
            )
            if distance < 0.0 or geom_id[0] != self.terrain_geom_id:
                heights[index] = float(qpos[2]) - 0.78
                misses += 1
            else:
                heights[index] = ray_start_z - distance
        self.ray_misses += misses
        self.last_world_points[:, 0] = world_x
        self.last_world_points[:, 1] = world_y
        self.last_world_points[:, 2] = heights
        return np.clip(float(qpos[2]) - 0.78 - heights, -0.5, 0.5)

    def reset(self, qpos):
        truth = self._raycast_truth(qpos)
        self.delay_steps = int(self.rng.integers(0, self.max_delay_steps + 1))
        self.noise_std = float(self.rng.uniform(0.0, 0.02))
        self.height_bias = float(self.rng.uniform(-0.015, 0.015))
        self.history.clear()
        for _ in range(self.max_delay_steps + 1):
            self.history.append(truth.copy())
        self.last_valid_mask[:] = True
        self.last_true_residual[:] = truth
        self.last_observation[:] = truth
        return truth

    def observe(self, qpos):
        truth = self._raycast_truth(qpos)
        self.history.append(truth.copy())
        delayed = self.history[-1 - self.delay_steps].copy()
        if self.mode == "oracle":
            observed = truth
            self.last_valid_mask[:] = True
        else:
            observed = delayed + self.height_bias
            observed += self.rng.normal(0.0, self.noise_std, size=observed.shape)
            observed = np.round(observed / 0.005) * 0.005
            self.last_valid_mask = self.rng.random(63) >= 0.08
            observed = np.where(self.last_valid_mask, observed, 0.0)
        observed = np.clip(observed, -0.5, 0.5)
        self.last_true_residual[:] = truth
        self.last_observation[:] = observed
        return observed.astype(np.float32)

    @property
    def mean_absolute_error(self):
        return float(np.mean(np.abs(self.last_observation - self.last_true_residual)))

