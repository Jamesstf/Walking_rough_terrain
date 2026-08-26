"""Exact 141-D observation adapter for the existing 12-DOF walking policy."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Optional

import numpy as np

from .config import WalkingPolicyConfig


def gravity_orientation(quaternion_wxyz: Iterable[float]) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quaternion_wxyz, dtype=np.float64)
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def yaw_from_quaternion(quaternion_wxyz: Iterable[float]) -> float:
    qw, qx, qy, qz = np.asarray(quaternion_wxyz, dtype=np.float64)
    return float(
        np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
    )


def world_to_body_velocity(velocity_world_xy: Iterable[float], yaw: float) -> np.ndarray:
    vx, vy = np.asarray(velocity_world_xy, dtype=np.float64)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array(
        [cosine * vx + sine * vy, -sine * vx + cosine * vy],
        dtype=np.float64,
    )


def robot_planar_state(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """Return [x, y, yaw, vx_body, vy_body, yaw_rate] from MuJoCo state."""
    yaw = yaw_from_quaternion(qpos[3:7])
    body_velocity = world_to_body_velocity(qvel[:2], yaw)
    return np.array(
        [qpos[0], qpos[1], yaw, body_velocity[0], body_velocity[1], qvel[5]],
        dtype=np.float64,
    )


class PlanarStateEstimator:
    """
    Estimate navigation velocity from pose increments over one MPC period.

    Instantaneous floating-base velocity oscillates strongly inside a walking
    gait.  Pose increments represent net locomotion and are therefore the state
    the outer-loop MPC should predict and control.
    """

    def __init__(self, dt: float, velocity_filter_alpha: float):
        self.dt = float(dt)
        self.alpha = float(velocity_filter_alpha)
        self._previous_pose: Optional[np.ndarray] = None
        self._filtered_velocity = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self._previous_pose = None
        self._filtered_velocity[:] = 0.0

    def update(self, qpos: np.ndarray) -> np.ndarray:
        yaw = yaw_from_quaternion(qpos[3:7])
        pose = np.array([qpos[0], qpos[1], yaw], dtype=np.float64)
        if self._previous_pose is not None:
            world_velocity = (pose[:2] - self._previous_pose[:2]) / self.dt
            body_velocity = world_to_body_velocity(world_velocity, yaw)
            yaw_delta = np.arctan2(
                np.sin(yaw - self._previous_pose[2]),
                np.cos(yaw - self._previous_pose[2]),
            )
            measured = np.array(
                [body_velocity[0], body_velocity[1], yaw_delta / self.dt],
                dtype=np.float64,
            )
            self._filtered_velocity = (
                (1.0 - self.alpha) * self._filtered_velocity + self.alpha * measured
            )
        self._previous_pose = pose
        return np.concatenate((pose, self._filtered_velocity)).astype(np.float64)


class WalkingObservationBuilder:
    """Build oldest-to-newest 3 x 47 observations exactly like navigation_low."""

    def __init__(self, config: WalkingPolicyConfig):
        self.config = config
        self.default_angles = np.asarray(config.default_angles, dtype=np.float32)
        self.command_scale = np.asarray(config.command_scale, dtype=np.float32)
        self.history: Deque[np.ndarray] = deque(maxlen=config.frame_stack)
        self.reset()

    def reset(self) -> None:
        self.history.clear()
        for _ in range(self.config.frame_stack):
            self.history.append(np.zeros(self.config.num_single_obs, dtype=np.float32))

    def build(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        previous_action: Iterable[float],
        command: Iterable[float],
        policy_step: int,
    ) -> np.ndarray:
        action = np.asarray(previous_action, dtype=np.float32)
        command = np.asarray(command, dtype=np.float32)
        qj = np.asarray(qpos[7:7 + self.config.num_actions], dtype=np.float32)
        dqj = np.asarray(qvel[6:6 + self.config.num_actions], dtype=np.float32)
        angular_velocity = np.asarray(qvel[3:6], dtype=np.float32)
        elapsed = policy_step * self.config.policy_dt
        phase = (elapsed % self.config.phase_period) / self.config.phase_period

        single = np.concatenate(
            (
                command * self.command_scale,                                      # 3
                angular_velocity * self.config.angular_velocity_scale,             # 3
                gravity_orientation(qpos[3:7]),                                    # 3
                (qj - self.default_angles) * self.config.dof_pos_scale,             # 12
                dqj * self.config.dof_vel_scale,                                   # 12
                action,                                                            # 12
                np.array([np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)]), # 2
            )
        ).astype(np.float32)
        if single.shape != (self.config.num_single_obs,):
            raise ValueError(
                f"Single observation has shape {single.shape}, expected "
                f"({self.config.num_single_obs},)"
            )
        single = np.clip(single, -self.config.clip_observations, self.config.clip_observations)
        self.history.append(single)
        stacked = np.stack(tuple(self.history), axis=0).reshape(-1).astype(np.float32)
        expected = self.config.frame_stack * self.config.num_single_obs
        if stacked.shape != (expected,):
            raise ValueError(f"Stacked observation has shape {stacked.shape}, expected ({expected},)")
        return stacked


class TorchWalkingPolicy:
    """Small lazy Torch wrapper so planning tests do not require Torch."""

    def __init__(self, config: WalkingPolicyConfig):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Torch is required only for the MuJoCo walking-policy runner. "
                "Run this entry in the project's legged_gym environment."
            ) from exc
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = torch.jit.load(config.policy_path, map_location=self.device)
        self.policy.eval()
        self.clip_actions = float(config.clip_actions)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        tensor = self.torch.from_numpy(observation).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            output = self.policy(tensor).detach().cpu().numpy().squeeze()
        action = np.asarray(output, dtype=np.float32)
        if action.shape != (12,):
            raise ValueError(f"Walking policy returned {action.shape}; expected (12,)")
        return np.clip(action, -self.clip_actions, self.clip_actions)
