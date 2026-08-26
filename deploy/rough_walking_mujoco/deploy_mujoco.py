"""Play model_10000.pt on procedurally generated rough terrain in MuJoCo."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from deploy.astar_mpc_walking.config import WalkingPolicyConfig
from deploy.astar_mpc_walking.walking_policy import (
    WalkingObservationBuilder,
    robot_planar_state,
)
from .checkpoint_policy import CheckpointWalkingPolicy
from .terrain_scene import TERRAIN_NAMES, build_rough_scene_xml, generate_terrain


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "legged_gym/logs/g1_walking_rough/rough_warmstart_33/model_10000.pt"
DEFAULT_SCENE = ROOT / "legged_gym/resources/robots/g1/scene.xml"


def _pd_torque(target, position, kp, velocity, kd):
    return (target - position) * kp - velocity * kd


def _roll_pitch(quaternion_wxyz: np.ndarray):
    qw, qx, qy, qz = np.asarray(quaternion_wxyz, dtype=np.float64)
    roll = float(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
    pitch = float(np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))
    return roll, pitch


def run(args):
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required; run this command in safe_track") from exc

    checkpoint = Path(args.checkpoint).resolve()
    scene = Path(args.scene).resolve()
    terrain = generate_terrain(args.terrain, args.level, args.seed)
    scene_xml, normalized_heights = build_rough_scene_xml(str(scene), terrain)
    model = mujoco.MjModel.from_xml_string(scene_xml)
    if model.nhfield != 1:
        raise RuntimeError(f"Expected one MuJoCo height field, got {model.nhfield}")
    model.hfield_data[:] = normalized_heights
    model.opt.timestep = 0.005
    model.opt.iterations = 50
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    data = mujoco.MjData(model)

    config = WalkingPolicyConfig(policy_path=str(checkpoint), scene_path=str(scene))
    policy = CheckpointWalkingPolicy(str(checkpoint), args.device, config.clip_actions)
    observations = WalkingObservationBuilder(config)
    default = np.asarray(config.default_angles, dtype=np.float64)
    kps = np.asarray(config.kps, dtype=np.float64)
    kds = np.asarray(config.kds, dtype=np.float64)
    locked_idx = np.asarray(config.locked_joint_idx, dtype=int)
    locked_target = np.asarray(config.locked_target, dtype=np.float64)
    locked_kps = np.asarray(config.locked_kps, dtype=np.float64)
    locked_kds = np.asarray(config.locked_kds, dtype=np.float64)
    command = np.array((args.vx, args.vy, args.yaw_rate), dtype=np.float64)
    initial_qpos = np.zeros(model.nq, dtype=np.float64)
    initial_qpos[:3] = (0.0, 0.0, terrain.height_at(0.0, 0.0) + 0.82)
    initial_qpos[3] = 1.0
    initial_qpos[7:19] = default
    initial_qpos[7 + locked_idx] = locked_target

    action = np.zeros(config.num_actions, dtype=np.float32)
    target = default.copy()
    policy_step = 0
    falls = 0
    fall_pending = False
    tracking_error_sum = 0.0
    tracking_samples = 0
    min_clearance = float("inf")
    max_tilt = 0.0
    start_xy = initial_qpos[:2].copy()

    def reset_robot(count_fall: bool):
        nonlocal action, target, policy_step, falls, fall_pending
        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial_qpos
        data.qvel[:] = 0.0
        action[:] = 0.0
        target[:] = default
        policy_step = 0
        observations.reset()
        fall_pending = False
        if count_fall:
            falls += 1
        mujoco.mj_forward(model, data)

    reset_robot(False)

    def simulation_step(counter: int):
        nonlocal action, target, policy_step, fall_pending
        nonlocal tracking_error_sum, tracking_samples, min_clearance, max_tilt
        joints = data.qpos[7:]
        joint_velocities = data.qvel[6:]
        torques = np.zeros(model.nu, dtype=np.float64)
        torques[:12] = _pd_torque(target, joints[:12], kps, joint_velocities[:12], kds)
        torques[locked_idx] = _pd_torque(
            locked_target, joints[locked_idx], locked_kps,
            joint_velocities[locked_idx], locked_kds,
        )
        data.ctrl[:] = torques
        mujoco.mj_step(model, data)

        terrain_height = terrain.height_at(float(data.qpos[0]), float(data.qpos[1]))
        clearance = float(data.qpos[2]) - terrain_height
        roll, pitch = _roll_pitch(data.qpos[3:7])
        min_clearance = min(min_clearance, clearance)
        max_tilt = max(max_tilt, abs(roll), abs(pitch))
        if clearance < 0.43 or abs(roll) > 1.0 or abs(pitch) > 1.0:
            fall_pending = True

        if counter % config.control_decimation == 0:
            state = robot_planar_state(data.qpos, data.qvel)
            tracking_error_sum += float(np.linalg.norm(state[3:5] - command[:2]))
            tracking_samples += 1
            observation = observations.build(data.qpos, data.qvel, action, command, policy_step)
            action = policy(observation)
            target = action.astype(np.float64) * config.action_scale + default
            policy_step += 1

        # Reset one simulation step after detecting a fall so the bad state is
        # visible in metrics without allowing numerical instability to spread.
        if fall_pending:
            reset_robot(True)

    print(
        f"[mujoco] checkpoint={checkpoint}\n"
        f"[mujoco] checkpoint_iteration={policy.iteration} terrain={args.terrain} "
        f"level={args.level} parameters={terrain.parameters}\n"
        f"[mujoco] command=({args.vx:.2f}, {args.vy:.2f}, {args.yaw_rate:.2f})m/s "
        f"duration={args.duration:.1f}s device={policy.device}"
    )
    total_steps = int(round(args.duration / model.opt.timestep))
    if args.headless:
        for counter in range(total_steps):
            simulation_step(counter)
    else:
        start_time = time.monotonic()
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth = 165.0
            viewer.cam.elevation = -22.0
            viewer.cam.distance = 4.5
            counter = 0
            while viewer.is_running() and counter < total_steps:
                step_start = time.monotonic()
                simulation_step(counter)
                viewer.cam.lookat[:] = (data.qpos[0] + 0.7, data.qpos[1], 0.55)
                viewer.sync()
                counter += 1
                remaining = model.opt.timestep - (time.monotonic() - step_start)
                if remaining > 0.0:
                    time.sleep(remaining)
        print(f"[mujoco] wall_time={time.monotonic() - start_time:.1f}s")

    final_state = robot_planar_state(data.qpos, data.qvel)
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": policy.iteration,
        "terrain": args.terrain,
        "level": args.level,
        "duration_s": args.duration,
        "command": command.tolist(),
        "falls": falls,
        "mean_planar_tracking_error_mps": tracking_error_sum / max(tracking_samples, 1),
        "minimum_base_clearance_m": min_clearance,
        "maximum_tilt_degrees": float(np.degrees(max_tilt)),
        "final_position_xy_m": final_state[:2].tolist(),
        "net_progress_x_m": float(final_state[0] - start_xy[0]),
    }
    print("[summary] " + json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--terrain", choices=TERRAIN_NAMES, default="random_rough")
    parser.add_argument("--level", type=int, choices=range(10), default=9)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--vx", type=float, default=0.25)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
