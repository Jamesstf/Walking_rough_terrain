"""Deploy the Noisy 204-D perceptive walking policy in MuJoCo."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from deploy.astar_mpc_walking.config import WalkingPolicyConfig
from deploy.astar_mpc_walking.walking_policy import WalkingObservationBuilder, robot_planar_state
from .heightmap_sensor import MuJoCoHeightmapSensor
from .perceptive_policy import PerceptiveCheckpointWalkingPolicy
from .terrain_scene import TERRAIN_NAMES, build_rough_scene_xml, generate_terrain


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "legged_gym/logs/g1_walking_rough_perceptive/noisy_heightmap_34/model_10000.pt"
DEFAULT_SCENE = ROOT / "legged_gym/resources/robots/g1/scene.xml"


def _pd_torque(target, position, kp, velocity, kd):
    return (target - position) * kp - velocity * kd


def _roll_pitch(quaternion_wxyz):
    qw, qx, qy, qz = np.asarray(quaternion_wxyz, dtype=np.float64)
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    return float(roll), float(pitch)


def _draw_sensor(mujoco, scene, sensor):
    scene.ngeom = 0
    for index, point in enumerate(sensor.last_world_points):
        if scene.ngeom >= scene.maxgeom:
            break
        valid = sensor.last_valid_mask[index]
        rgba = np.array((0.10, 0.90, 0.30, 0.90) if valid else (1.0, 0.15, 0.10, 0.95), dtype=np.float32)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array((0.018, 0.0, 0.0), dtype=np.float64),
            point + np.array((0.0, 0.0, 0.015)),
            np.eye(3).reshape(-1), rgba,
        )
        scene.ngeom += 1


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
    model.hfield_data[:] = normalized_heights
    model.opt.timestep = 0.005
    model.opt.iterations = 50
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    data = mujoco.MjData(model)

    config = WalkingPolicyConfig(policy_path=str(checkpoint), scene_path=str(scene))
    policy = PerceptiveCheckpointWalkingPolicy(
        str(checkpoint), args.device, config.clip_actions
    )
    proprioception = WalkingObservationBuilder(config)
    sensor = MuJoCoHeightmapSensor(
        mujoco, model, data, seed=args.sensor_seed, mode=args.perception_mode
    )
    default = np.asarray(config.default_angles, dtype=np.float64)
    kps, kds = np.asarray(config.kps), np.asarray(config.kds)
    locked_idx = np.asarray(config.locked_joint_idx, dtype=int)
    locked_target = np.asarray(config.locked_target, dtype=np.float64)
    locked_kps, locked_kds = np.asarray(config.locked_kps), np.asarray(config.locked_kds)
    command = np.array((args.vx, args.vy, args.yaw_rate), dtype=np.float64)

    initial_qpos = np.zeros(model.nq, dtype=np.float64)
    initial_qpos[:3] = (0.0, 0.0, terrain.height_at(0.0, 0.0) + 0.82)
    initial_qpos[3] = 1.0
    initial_qpos[7:19] = default
    initial_qpos[7 + locked_idx] = locked_target
    action = np.zeros(12, dtype=np.float32)
    target = default.copy()
    policy_step = 0
    falls = 0
    tracking_error_sum = 0.0
    sensor_error_sum = 0.0
    sensor_samples = 0
    minimum_clearance = float("inf")
    maximum_tilt = 0.0

    def reset_robot(count_fall=False):
        nonlocal action, target, policy_step, falls
        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial_qpos
        data.qvel[:] = 0.0
        action[:] = 0.0
        target[:] = default
        policy_step = 0
        proprioception.reset()
        mujoco.mj_forward(model, data)
        sensor.reset(data.qpos)
        if count_fall:
            falls += 1

    reset_robot()

    def simulation_step(counter):
        nonlocal action, target, policy_step, tracking_error_sum
        nonlocal sensor_error_sum, sensor_samples, minimum_clearance, maximum_tilt
        joints, joint_velocities = data.qpos[7:], data.qvel[6:]
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
        minimum_clearance = min(minimum_clearance, clearance)
        maximum_tilt = max(maximum_tilt, abs(roll), abs(pitch))
        fell = clearance < 0.43 or abs(roll) > 1.0 or abs(pitch) > 1.0

        if counter % config.control_decimation == 0:
            state = robot_planar_state(data.qpos, data.qvel)
            tracking_error_sum += float(np.linalg.norm(state[3:5] - command[:2]))
            prop = proprioception.build(data.qpos, data.qvel, action, command, policy_step)
            terrain_observation = sensor.observe(data.qpos)
            observation = np.concatenate((prop, terrain_observation * 5.0)).astype(np.float32)
            action = policy(observation)
            target = action.astype(np.float64) * config.action_scale + default
            sensor_error_sum += sensor.mean_absolute_error
            sensor_samples += 1
            policy_step += 1
        if fell:
            reset_robot(True)

    print(
        f"[perceptive-mujoco] checkpoint={checkpoint}\n"
        f"[perceptive-mujoco] iteration={policy.iteration} mode={args.perception_mode} "
        f"terrain={args.terrain} level={args.level} parameters={terrain.parameters}\n"
        f"[perceptive-mujoco] delay={sensor.delay_steps * config.policy_dt * 1000:.0f}ms "
        f"noise_std={sensor.noise_std * 100:.2f}cm bias={sensor.height_bias * 100:.2f}cm"
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
                _draw_sensor(mujoco, viewer.user_scn, sensor)
                viewer.cam.lookat[:] = (data.qpos[0] + 0.7, data.qpos[1], 0.55)
                viewer.sync()
                counter += 1
                remaining = model.opt.timestep - (time.monotonic() - step_start)
                if remaining > 0.0:
                    time.sleep(remaining)
        print(f"[perceptive-mujoco] wall_time={time.monotonic() - start_time:.1f}s")

    state = robot_planar_state(data.qpos, data.qvel)
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": policy.iteration,
        "perception_mode": args.perception_mode,
        "terrain": args.terrain,
        "level": args.level,
        "duration_s": args.duration,
        "falls": falls,
        "mean_planar_tracking_error_mps": tracking_error_sum / max(sensor_samples, 1),
        "mean_heightmap_error_m": sensor_error_sum / max(sensor_samples, 1),
        "ray_misses": sensor.ray_misses,
        "minimum_base_clearance_m": minimum_clearance,
        "maximum_tilt_degrees": float(np.degrees(maximum_tilt)),
        "final_position_xy_m": state[:2].tolist(),
    }
    print("[summary] " + json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--terrain", choices=TERRAIN_NAMES, default="up_stairs")
    parser.add_argument("--level", type=int, choices=range(10), default=9)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--vx", type=float, default=0.25)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--sensor-seed", type=int, default=34)
    parser.add_argument("--perception-mode", choices=("oracle", "noisy"), default="noisy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
