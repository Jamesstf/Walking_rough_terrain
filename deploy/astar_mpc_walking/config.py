"""Configuration types for the independent navigation module."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class CircleObstacle:
    center: Tuple[float, float]
    radius: float
    height: float = 1.0


@dataclass(frozen=True)
class RectangleObstacle:
    center: Tuple[float, float]
    size: Tuple[float, float]
    height: float = 1.0


Obstacle = Union[CircleObstacle, RectangleObstacle]


@dataclass(frozen=True)
class MapConfig:
    x_bounds: Tuple[float, float] = (-1.0, 6.0)
    y_bounds: Tuple[float, float] = (-3.0, 3.0)
    resolution: float = 0.10
    robot_radius: float = 0.30
    execution_error_margin: float = 0.15
    localization_error_margin: float = 0.05
    obstacles: Tuple[Obstacle, ...] = ()

    @property
    def inflation_radius(self) -> float:
        return (
            self.robot_radius
            + self.execution_error_margin
            + self.localization_error_margin
        )

    @property
    def hard_safety_inflation_radius(self) -> float:
        """Hard collision boundary; execution margin remains available as a tracking tube."""
        return self.robot_radius + self.localization_error_margin


@dataclass(frozen=True)
class PathConfig:
    allow_diagonal: bool = True
    prevent_corner_cutting: bool = True
    shortcut: bool = True
    smooth_iterations: int = 30
    smooth_weight: float = 0.20
    point_spacing: float = 0.08
    reference_speed: float = 0.35
    goal_tolerance: float = 0.18
    stop_speed_tolerance: float = 0.10


@dataclass(frozen=True)
class WalkingResponseConfig:
    # Steady-state map: actual body velocity = gain_matrix @ command.
    gain_matrix: Tuple[Tuple[float, float, float], ...] = (
        (0.85, 0.00, 0.00),
        (0.00, 0.75, 0.00),
        (0.00, 0.00, 0.80),
    )
    time_constants: Tuple[float, float, float] = (0.35, 0.40, 0.30)
    command_delay_steps: int = 1
    disturbance_filter_gain: float = 0.12
    disturbance_limit: Tuple[float, float, float] = (0.12, 0.12, 0.20)


@dataclass(frozen=True)
class MPCConfig:
    dt: float = 0.10
    horizon: int = 12
    command_min: Tuple[float, float, float] = (-0.60, -0.35, -0.80)
    command_max: Tuple[float, float, float] = (0.60, 0.35, 0.80)
    command_delta_max: Tuple[float, float, float] = (0.12, 0.10, 0.20)
    state_weights: Tuple[float, ...] = (35.0, 35.0, 5.0, 2.0, 2.0, 1.0)
    terminal_weight_multiplier: float = 4.0
    command_weights: Tuple[float, float, float] = (0.20, 0.20, 0.10)
    command_delta_weights: Tuple[float, float, float] = (3.0, 3.0, 1.5)
    max_iterations: int = 80
    solver_ftol: float = 1.0e-4
    corridor_scale: float = 0.90
    corridor_max_half_width: float = 0.35
    feasibility_tolerance: float = 2.0e-3
    velocity_filter_alpha: float = 0.35


@dataclass(frozen=True)
class WalkingPolicyConfig:
    policy_path: str
    scene_path: str
    simulation_dt: float = 0.005
    control_decimation: int = 4
    phase_period: float = 0.8
    stand_height: float = 0.78
    num_actions: int = 12
    frame_stack: int = 3
    num_single_obs: int = 47
    default_angles: Tuple[float, ...] = (
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    )
    kps: Tuple[float, ...] = (
        100, 100, 100, 150, 40, 40,
        100, 100, 100, 150, 40, 40,
    )
    kds: Tuple[float, ...] = (2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2)
    locked_joint_idx: Tuple[int, ...] = tuple(range(12, 23))
    locked_kps: Tuple[float, ...] = (300, 100, 100, 50, 50, 20, 100, 100, 50, 50, 20)
    locked_kds: Tuple[float, ...] = (3, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1)
    locked_target: Tuple[float, ...] = (
        0, 0.3, 0.25, 0, 0.97, 0.15,
        0.3, -0.25, 0, 0.97, -0.15,
    )
    action_scale: float = 0.25
    command_scale: Tuple[float, float, float] = (2.0, 2.0, 0.25)
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    angular_velocity_scale: float = 0.25
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    camera_follow: bool = True
    camera_azimuth: float = 180.0
    camera_elevation: float = -24.0
    camera_distance: float = 4.5
    camera_lookat_height: float = 0.60
    camera_lookahead: float = 0.80

    @property
    def policy_dt(self) -> float:
        return self.simulation_dt * self.control_decimation


@dataclass(frozen=True)
class NavigationConfig:
    map: MapConfig
    path: PathConfig
    walking_response: WalkingResponseConfig
    mpc: MPCConfig
    walking_policy: Optional[WalkingPolicyConfig] = None
    start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal: Tuple[float, float] = (5.0, 0.0)


def _tuple(values: Sequence[Any]) -> tuple:
    return tuple(values)


def _parse_obstacle(raw: Dict[str, Any]) -> Obstacle:
    kind = raw["type"].lower()
    if kind == "circle":
        return CircleObstacle(
            center=_tuple(raw["center"]),
            radius=float(raw["radius"]),
            height=float(raw.get("height", 1.0)),
        )
    if kind in ("rectangle", "box"):
        return RectangleObstacle(
            center=_tuple(raw["center"]),
            size=_tuple(raw["size"]),
            height=float(raw.get("height", 1.0)),
        )
    raise ValueError(f"Unsupported obstacle type: {raw['type']!r}")


def _construct_config(raw: Dict[str, Any], config_path: Path) -> NavigationConfig:
    map_raw = raw.get("map", {})
    map_cfg = MapConfig(
        x_bounds=_tuple(map_raw.get("x_bounds", (-1.0, 6.0))),
        y_bounds=_tuple(map_raw.get("y_bounds", (-3.0, 3.0))),
        resolution=float(map_raw.get("resolution", 0.10)),
        robot_radius=float(map_raw.get("robot_radius", 0.30)),
        execution_error_margin=float(map_raw.get("execution_error_margin", 0.15)),
        localization_error_margin=float(map_raw.get("localization_error_margin", 0.05)),
        obstacles=tuple(_parse_obstacle(item) for item in map_raw.get("obstacles", [])),
    )

    path_raw = raw.get("path", {})
    path_cfg = PathConfig(**path_raw)

    response_raw = dict(raw.get("walking_response", {}))
    for key in ("gain_matrix", "time_constants", "disturbance_limit"):
        if key in response_raw:
            response_raw[key] = tuple(
                tuple(row) if isinstance(row, list) else row
                for row in response_raw[key]
            )
    response_cfg = WalkingResponseConfig(**response_raw)

    mpc_raw = dict(raw.get("mpc", {}))
    tuple_fields = (
        "command_min", "command_max", "command_delta_max", "state_weights",
        "command_weights", "command_delta_weights",
    )
    for key in tuple_fields:
        if key in mpc_raw:
            mpc_raw[key] = _tuple(mpc_raw[key])
    mpc_cfg = MPCConfig(**mpc_raw)

    policy_cfg = None
    if "walking_policy" in raw:
        policy_raw = dict(raw["walking_policy"])
        for key in (
            "default_angles", "kps", "kds", "locked_joint_idx", "locked_kps",
            "locked_kds", "locked_target", "command_scale",
        ):
            if key in policy_raw:
                policy_raw[key] = _tuple(policy_raw[key])
        project_root = Path(__file__).resolve().parents[2]
        for key in ("policy_path", "scene_path"):
            value = str(policy_raw[key]).replace("{PROJECT_ROOT}", str(project_root))
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = (config_path.parent / candidate).resolve()
            policy_raw[key] = str(candidate)
        policy_cfg = WalkingPolicyConfig(**policy_raw)

    cfg = NavigationConfig(
        map=map_cfg,
        path=path_cfg,
        walking_response=response_cfg,
        mpc=mpc_cfg,
        walking_policy=policy_cfg,
        start=_tuple(raw.get("start", (0.0, 0.0, 0.0))),
        goal=_tuple(raw.get("goal", (5.0, 0.0))),
    )
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: NavigationConfig) -> None:
    if cfg.map.resolution <= 0.0:
        raise ValueError("map.resolution must be positive")
    if cfg.map.x_bounds[0] >= cfg.map.x_bounds[1] or cfg.map.y_bounds[0] >= cfg.map.y_bounds[1]:
        raise ValueError("Map bounds must be strictly increasing")
    if cfg.mpc.dt <= 0.0 or cfg.mpc.horizon < 2:
        raise ValueError("MPC dt must be positive and horizon must be at least 2")
    if not 0.0 < cfg.mpc.velocity_filter_alpha <= 1.0:
        raise ValueError("mpc.velocity_filter_alpha must be in (0, 1]")
    if np.any(np.asarray(cfg.walking_response.time_constants) <= cfg.mpc.dt):
        raise ValueError("walking response time constants must be greater than mpc.dt")
    if cfg.walking_response.command_delay_steps < 0:
        raise ValueError("command_delay_steps cannot be negative")
    if cfg.walking_policy is not None:
        policy = cfg.walking_policy
        if policy.num_single_obs != 47 or policy.frame_stack != 3:
            raise ValueError("The supplied G1 walking policy expects 3 x 47 = 141 observations")
        ratio = cfg.mpc.dt / policy.policy_dt
        if not np.isclose(ratio, round(ratio), atol=1.0e-8):
            raise ValueError("mpc.dt must be an integer multiple of the walking policy period")


def load_config(path: Union[str, Path]) -> NavigationConfig:
    """Load and validate a JSON configuration file."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    return _construct_config(raw, config_path)
