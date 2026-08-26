from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import math

import isaacgym
from isaacgym import gymapi
from isaacgym import gymutil
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, set_seed
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

H, W = 480, 640

ROBOT_LIST = ['h1', 'g1', 'h1_2']

WALKING = [f"{robot}_walking" for robot in ROBOT_LIST] + ["g1_walking_unitree", "g1_walking_rough"]
REACHING = [f"{robot}_reaching" for robot in ROBOT_LIST]
STEPPING = [f"{robot}_stepping" for robot in ROBOT_LIST]
SQUATTING = [f"{robot}_squatting" for robot in ROBOT_LIST]
TASK_REACH = [f"{robot}_task_reach" for robot in ROBOT_LIST]
TASK_BOX = [f"{robot}_task_box" for robot in ROBOT_LIST]
TASK_BUTTON = [f"{robot}_task_button" for robot in ROBOT_LIST]
TASK_LIFT = [f"{robot}_task_lift" for robot in ROBOT_LIST]
TASK_BALL = [f"{robot}_task_ball" for robot in ROBOT_LIST]
TASK_CARRY = [f"{robot}_task_carry" for robot in ROBOT_LIST]
TASK_TRANSFER = [f"{robot}_task_transfer" for robot in ROBOT_LIST]
TASK_CABINET = [f"{robot}_task_cabinet" for robot in ROBOT_LIST]

def visualize_task(task, env):
    """only be used when with display"""
    ### Low-level skills
    if task in WALKING:
        env.gym.clear_lines(env.viewer)
        commands = env.commands.clone()
        commands[:, 2] = 0
        root_states = env.ori_root_states[:, :3]
        root_end = root_states + 1e3 * commands[:, :3]
        for i in range(env.num_envs):
            gymutil.draw_line(gymapi.Vec3(root_states[i, 0], root_states[i, 1], root_states[i, 2]), gymapi.Vec3(root_end[i, 0], root_end[i, 1], root_end[i, 2]), gymapi.Vec3(1, 0, 0), env.gym, env.viewer, env.envs[i])
    elif task in REACHING+TASK_REACH:
        env.gym.clear_lines(env.viewer)
        # Create helper geometry used for visualization
        # Create an wireframe axis
        axes_geom = gymutil.AxesGeometry(0.15)
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        yellow_geom = gymutil.WireframeSphereGeometry(0.05, 12, 12, sphere_pose, color=(1, 1, 0))
        purple_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(1, 0, 1))
        wrist_pos = env.rigid_state[:, env.wrist_indices, :7] # [num_envs, 2, 7]
        ref_wrist_pos = env.ref_wrist_pos # [num_envs, 2, 7]
        for i in range(env.num_envs):
            wrist_pos_i = wrist_pos[i] # [2, 7]
            ref_wrist_pos_i = ref_wrist_pos[i] # [2, 7]
            for j in range(2):
                wrist_pos_ij = wrist_pos_i[j] # [7]
                ref_wrist_pos_ij = ref_wrist_pos_i[j] # [7]
                ori_wrist_pos_ij = env.ori_wrist_pos[i, j] # [7]
                # convert to gymapi.Transform
                wrist_pos_ij = gymapi.Transform(gymapi.Vec3(wrist_pos_ij[0], wrist_pos_ij[1], wrist_pos_ij[2]), gymapi.Quat())
                ref_wrist_pos_ij = gymapi.Transform(gymapi.Vec3(ref_wrist_pos_ij[0], ref_wrist_pos_ij[1], ref_wrist_pos_ij[2]), gymapi.Quat())
                ori_wrist_pos_ij = gymapi.Transform(gymapi.Vec3(ori_wrist_pos_ij[0], ori_wrist_pos_ij[1], ori_wrist_pos_ij[2]), gymapi.Quat())
                # current wrist pos and ref wrist pos
                gymutil.draw_lines(axes_geom, env.gym, env.viewer, env.envs[i], wrist_pos_ij)
                # gymutil.draw_lines(sphere_geom, env.gym, env.viewer, env.envs[i], wrist_pos_ij)
                # gymutil.draw_lines(axes_geom, env.gym, env.viewer, env.envs[i], ref_wrist_pos_ij)
                gymutil.draw_lines(yellow_geom, env.gym, env.viewer, env.envs[i], ref_wrist_pos_ij)
                # original wrist pos
                # gymutil.draw_lines(axes_geom, env.gym, env.viewer, env.envs[i], ori_wrist_pos_ij)
                gymutil.draw_lines(purple_geom, env.gym, env.viewer, env.envs[i], ori_wrist_pos_ij)
    elif task in STEPPING:
        env.gym.clear_lines(env.viewer)
        # Create helper geometry used for visualization
        # Create an wireframe axis
        axes_geom = gymutil.AxesGeometry(0.15)
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        yellow_geom = gymutil.WireframeSphereGeometry(0.05, 12, 12, sphere_pose, color=(1, 1, 0))
        purple_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(1, 0, 1))
        feet_pos = env.rigid_state[:, env.feet_indices, :2] # [num_envs, 2, 2]
        ref_feet_pos = env.ref_feet_pos # [num_envs, 2, 2]
        for i in range(env.num_envs):
            feet_pos_i = feet_pos[i] # [2, 2]
            ref_feet_pos_i = ref_feet_pos[i] # [2, 2]
            for j in range(2):
                feet_pos_ij = feet_pos_i[j] # [2]
                ref_feet_pos_ij = ref_feet_pos_i[j] # [2]
                ori_feet_pos_ij = env.ori_feet_pos[i, j] # [2]
                # convert to gymapi.Transform
                feet_pos_ij = gymapi.Transform(gymapi.Vec3(feet_pos_ij[0], feet_pos_ij[1], 0), gymapi.Quat())
                ref_feet_pos_ij = gymapi.Transform(gymapi.Vec3(ref_feet_pos_ij[0], ref_feet_pos_ij[1], 0), gymapi.Quat())
                ori_feet_pos_ij = gymapi.Transform(gymapi.Vec3(ori_feet_pos_ij[0], ori_feet_pos_ij[1], 0), gymapi.Quat())
                # current feet pos and ref feet pos
                gymutil.draw_lines(axes_geom, env.gym, env.viewer, env.envs[i], feet_pos_ij)
                gymutil.draw_lines(yellow_geom, env.gym, env.viewer, env.envs[i], ref_feet_pos_ij)
                # original feet pos
                gymutil.draw_lines(purple_geom, env.gym, env.viewer, env.envs[i], ori_feet_pos_ij)
    elif task in SQUATTING:
        env.gym.clear_lines(env.viewer)
        # Create helper geometry used for visualization
        # Create an wireframe axis
        axes_geom_cur = gymutil.AxesGeometry(0.25)
        axes_geom_tgt = gymutil.AxesGeometry(0.5)
        root_pos = env.root_states[:, :3].clone() # [num_envs, 3]
        ref_root_pos = env.root_states[:, :3].clone() # [num_envs, 3]
        ref_root_pos[:, 2] = env.ref_root_height
        for i in range(env.num_envs):
            root_pos_i = root_pos[i] # [3]
            ref_root_pos_i = ref_root_pos[i] # [3]
            # convert to gymapi.Transform
            root_pos_i = gymapi.Transform(gymapi.Vec3(root_pos_i[0], root_pos_i[1], root_pos_i[2]), gymapi.Quat())
            ref_root_pos_i = gymapi.Transform(gymapi.Vec3(ref_root_pos_i[0], ref_root_pos_i[1], ref_root_pos_i[2]), gymapi.Quat())
            # current root pos and ref root pos
            gymutil.draw_lines(axes_geom_cur, env.gym, env.viewer, env.envs[i], root_pos_i)
            gymutil.draw_lines(axes_geom_tgt, env.gym, env.viewer, env.envs[i], ref_root_pos_i)
    
    ### High-level tasks
    elif task in TASK_BOX+TASK_TRANSFER:
        env.gym.clear_lines(env.viewer)
        # Create helper geometry used for visualization
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        red_geom = gymutil.WireframeSphereGeometry(0.05, 12, 12, sphere_pose, color=(1, 0, 0))
        for i in range(env.num_envs):
            box_goal_pos = env.box_goal_pos[i, :3]
            box_goal_i = gymapi.Transform(gymapi.Vec3(box_goal_pos[0], box_goal_pos[1], box_goal_pos[2]), gymapi.Quat())
            gymutil.draw_lines(red_geom, env.gym, env.viewer, env.envs[i], box_goal_i)
    elif task in TASK_BUTTON:
        env.gym.clear_lines(env.viewer)
        # Create helper geometry used for visualization
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        red_geom = gymutil.WireframeSphereGeometry(0.05, 12, 12, sphere_pose, color=(1, 0, 0))
        for i in range(env.num_envs):
            button_goal_pos = env.button_goal_pos[i, :3]
            button_goal_i = gymapi.Transform(gymapi.Vec3(button_goal_pos[0], button_goal_pos[1], button_goal_pos[2]), gymapi.Quat())
            gymutil.draw_lines(red_geom, env.gym, env.viewer, env.envs[i], button_goal_i)
    elif task in TASK_LIFT+TASK_CARRY:
        env.gym.clear_lines(env.viewer)
        # Create helper geometry used for visualization
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        yellow_geom = gymutil.WireframeSphereGeometry(0.1, 12, 12, sphere_pose, color=(1, 1, 0))
        red_geom = gymutil.WireframeSphereGeometry(0.1, 12, 12, sphere_pose, color=(1, 0, 0))
        for i in range(env.num_envs):
            box_pos = env.box_root_states[i, :3]
            box_goal_pos = env.box_goal_pos[i]
            box_handle_left = box_pos.clone()
            box_handle_right = box_pos.clone()
            gymutil.draw_lines(yellow_geom, env.gym, env.viewer, env.envs[i], gymapi.Transform(gymapi.Vec3(box_handle_left[0], box_handle_left[1], box_handle_left[2]), gymapi.Quat()))
            gymutil.draw_lines(yellow_geom, env.gym, env.viewer, env.envs[i], gymapi.Transform(gymapi.Vec3(box_handle_right[0], box_handle_right[1], box_handle_right[2]), gymapi.Quat()))
            gymutil.draw_lines(red_geom, env.gym, env.viewer, env.envs[i], gymapi.Transform(gymapi.Vec3(box_goal_pos[0], box_goal_pos[1], box_goal_pos[2]), gymapi.Quat()))
    elif task in TASK_BALL:
        env.gym.clear_lines(env.viewer)
        # Create helper geometry used for visualization
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        red_geom = gymutil.WireframeSphereGeometry(0.2, 12, 12, sphere_pose, color=(1, 0, 0))
        for i in range(env.num_envs):
            ball_pos = env.ball_root_states[i, :3]
            goal_pos = env.goal_pos[i, :3]
            gymutil.draw_lines(red_geom, env.gym, env.viewer, env.envs[i], gymapi.Transform(gymapi.Vec3(goal_pos[0], goal_pos[1], goal_pos[2]), gymapi.Quat()))
    elif task in TASK_CABINET:
        # no need to visualize
        pass
    elif task in ['h1_imitation', 'h1_exbody']:
        pass
    else:
        raise NotImplementedError(f"Task {task} is not supported for visualization")
    
def override_env_cfg(env_cfg: LeggedRobotCfg, args):
    print('====> URDF file:', env_cfg.asset.file)
    # override some parameters for testing
    default_num_envs = 50
    if args.task in WALKING:
        default_num_envs = 1
        env_cfg.env.episode_length_s = 24
    elif args.task in TASK_REACH:
        env_cfg.env.episode_length_s = 20
        env_cfg.human.freq = 2
    elif args.task in TASK_BOX+TASK_TRANSFER+TASK_BUTTON:
        env_cfg.env.episode_length_s = 2.5
    elif args.task in TASK_LIFT+TASK_BALL+TASK_CARRY+TASK_CABINET:
        env_cfg.env.episode_length_s = 2.5
    else:
        env_cfg.env.episode_length_s = 8
    if args.visualize:
        default_num_envs = 1
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, default_num_envs)
    # env_cfg.env.mesh_type = "plane"
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    # env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = False
    # env_cfg.noise.noise_level = 0.5
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # env_cfg.commands.ranges.l_wrist_pos_x = [-0.5, 0.5]
    # env_cfg.commands.ranges.l_wrist_pos_y = [-0.5, 0.5]
    # env_cfg.commands.ranges.l_wrist_pos_z = [-0.5, 0.5]
    # env_cfg.commands.ranges.r_wrist_pos_x = [-0.5, 0.5]
    # env_cfg.commands.ranges.r_wrist_pos_y = [-0.5, 0.5]
    # env_cfg.commands.ranges.r_wrist_pos_z = [-0.5, 0.5]
    # env_cfg.commands.ranges.wrist_max_radius = 0.5
    
    return env_cfg

def get_camera_pose(task):
    if not EGO_CENTRIC:
        if task in TASK_BUTTON+TASK_BALL+TASK_CABINET:
            camera_offset = gymapi.Vec3(-1, -2, 1)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                        np.deg2rad(45))
        else:
            camera_offset = gymapi.Vec3(1, -1, 1)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
    else:
        camera_offset = gymapi.Vec3(0.1, 0, 0.9)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0),
                                                    np.deg2rad(45))
    return gymapi.Transform(camera_offset, camera_rotation)

def play_walking(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task, load_run=args.load_run, experiment_name=args.experiment_name)
    # Use the Unitree walking play settings.
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    for _ in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

def play(args):
    if args.task in WALKING:
        play_walking(args)
        return

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task, load_run=args.load_run, experiment_name=args.experiment_name)
    HRL = "_task_" in args.task and "Hierarchical" in train_cfg.runner.policy_class_name
    
    env_cfg = override_env_cfg(env_cfg, args)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.run_name = 'play'

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device, hrl=HRL)
    
    model_name = f'{args.load_run}_{train_cfg.runner.resume_path.split("_")[-1].split(".")[0]}'
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        filename_pt = f'{model_name}.pt'
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, filename_pt)
        print('Exported policy as jit script to: ', path)

    robot_index = 0 # which robot (env) is used for logging
    joint_index = 1 # which joint pos is used for logging
    state_log_interval = 1000 # number of steps before plotting states
    rew_log_interval = env.max_episode_length - 1 # number of steps before print average episode rewards
    N_rollouts = 10
    if RECORD_FRAMES:
        frame_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
        os.makedirs(frame_path, exist_ok=True)
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = W
        camera_properties.height = H
        cam = env.gym.create_camera_sensor(env.envs[robot_index], camera_properties)
        camera_pose = get_camera_pose(args.task)
        actor_handle = env.gym.get_actor_handle(env.envs[robot_index], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[robot_index], actor_handle, 0)
        env.gym.attach_camera_to_body(
            cam, 
            env.envs[robot_index], 
            body_handle,
            camera_pose,
            gymapi.FOLLOW_POSITION if not EGO_CENTRIC else gymapi.FOLLOW_TRANSFORM
        )

    max_steps = int(env.max_episode_length)
    if args.task in REACHING+TASK_REACH:
        last_ref_wrist_pos = env.ref_wrist_pos[robot_index][:,:3].cpu().numpy()
    if args.task in STEPPING:
        last_ref_feet_pos = env.ref_feet_pos[robot_index][:,:2].cpu().numpy()
    if args.task in SQUATTING:
        last_ref_root_height = env.ref_root_height[robot_index].cpu().numpy()
    if args.task in TASK_BOX+TASK_BUTTON+TASK_LIFT+TASK_BALL+TASK_CARRY+TASK_CABINET+TASK_TRANSFER:
        max_steps = int(env.max_episode_length) * 10

    for i_rollout in range(N_rollouts):
        print(f"====> Rollout {i_rollout+1}/{N_rollouts}")
        robot_index = i_rollout % env_cfg.env.num_envs
        logger = Logger(env.dt)
        env.ori_root_states = env.root_states.clone()
        if RECORD_FRAMES:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            filename_mp4 = f'{args.task}_{model_name}_{i_rollout}.mp4'
            video = cv2.VideoWriter(os.path.join(frame_path, filename_mp4), fourcc, 25.0, (W, H))
        for i in tqdm(range(max_steps)):
            # import pdb; pdb.set_trace()
            # visualize task-related information
            visualize_task(args.task, env)

            actions = policy(obs.detach())
            actions = actions["actions_mean"] if HRL else actions
            # import pdb; pdb.set_trace()

            if FIX_COMMAND:
                env.commands[:, 0] = 1.0
                env.commands[:, 1] = 0.0
                env.commands[:, 2] = 0.0
                env.commands[:, 3] = 0.0

            # import pdb;pdb.set_trace()
            # actions[:] = 0.0
            obs, _, rews, dones, infos = env.step(actions.detach())

            if (args.task == 'g1_task_cabinet' or args.task == 'h1_2_task_cabinet') and i <= 1000:
                continue # still need to fix accuracy, but all methods are 0.000 so it's okay
            
            if RECORD_FRAMES:
                if i % 4 == 0:
                    env.gym.fetch_results(env.sim, True)
                    env.gym.step_graphics(env.sim)
                    env.gym.render_all_camera_sensors(env.sim)
                    img = env.gym.get_camera_image(env.sim, env.envs[robot_index], cam, gymapi.IMAGE_COLOR)
                    img = np.reshape(img, (H, W, 4))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    video.write(img[..., :3])
                
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, :].detach().cpu().numpy() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, :].detach().cpu().numpy(),
                    'dof_vel': env.dof_vel[robot_index, :].detach().cpu().numpy(),
                    'dof_torque': env.torques[robot_index, :].detach().cpu().numpy(),
                    'command_x': env.commands[robot_index, 0].detach().cpu().numpy(),
                    'command_y': env.commands[robot_index, 1].detach().cpu().numpy(),
                    'command_yaw': env.commands[robot_index, 2].detach().cpu().numpy(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].detach().cpu().numpy(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].detach().cpu().numpy(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].detach().cpu().numpy(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].detach().cpu().numpy(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].detach().cpu().numpy(),
                    'base_roll': env.base_euler_xyz[robot_index, 0].detach().cpu().numpy(),
                    'base_pitch': env.base_euler_xyz[robot_index, 1].detach().cpu().numpy(),
                    'base_height': env.root_states[robot_index, 2].detach().cpu().numpy(),
                }
            )

            if args.task in REACHING+TASK_REACH:
                ref_wrist_pos = env.ref_wrist_pos[robot_index][:,:3].cpu().numpy()
                wrist_pos = env.rigid_state[robot_index, env.wrist_indices, :3].cpu().numpy()
                
                # if wrist_pos updates, which means that ref_wrist_pos is not last_wrist_pos
                if not np.allclose(ref_wrist_pos, last_ref_wrist_pos):
                    print("=> update reach goal")
                    wrist_pos_diff = np.mean(np.abs(last_ref_wrist_pos - wrist_pos))
                    logger.log_metrics(
                        {
                            'wrist_pos_diff': wrist_pos_diff
                        }
                    )
                    last_ref_wrist_pos = ref_wrist_pos
                    print('wrist_pos_diff:', wrist_pos_diff)
                    
            if args.task in STEPPING:
                ref_feet_pos = env.ref_feet_pos[robot_index][:,:2].cpu().numpy()
                feet_pos = env.rigid_state[robot_index, env.feet_indices, :2].cpu().numpy()
                
                # if feet_pos updates, which means that ref_feet_pos is not last_feet_pos
                if not np.allclose(ref_feet_pos, last_ref_feet_pos):
                    print("=> update kicking goal")
                    feet_pos_diff = np.mean(np.abs(last_ref_feet_pos - feet_pos))
                    logger.log_metrics(
                        {
                            'feet_pos_diff': feet_pos_diff
                        }
                    )
                    last_ref_feet_pos = ref_feet_pos
                    print('feet_pos_diff:', feet_pos_diff)
            
            if args.task in SQUATTING:
                ref_root_height = env.ref_root_height[robot_index].cpu().numpy()
                root_height = env.root_states[robot_index, 2].cpu().numpy()
                
                # if root_height updates, which means that ref_root_height is not last_root_height
                if not np.allclose(ref_root_height, last_ref_root_height):
                    print("=> update squating goal")
                    root_height_diff = np.mean(np.abs(last_ref_root_height - root_height))
                    logger.log_metrics(
                        {
                            'root_height_diff': root_height_diff
                        }
                    )
                    last_ref_root_height = ref_root_height
                    print('root_height_diff:', root_height_diff)
                    
            if args.task in TASK_BOX+TASK_TRANSFER:
                if i > 0 and i % (rew_log_interval-1) == 0:
                    box_pos = env.box_root_states[robot_index, :3].cpu().numpy()
                    box_goal_pos = env.box_goal_pos[robot_index, :3].cpu().numpy()
                    box_pos_diff = np.mean(np.abs(box_pos - box_goal_pos))
                    logger.log_metrics(
                        {
                            'box_pos_diff': box_pos_diff
                        }
                    )
                    print('box_pos_diff:', box_pos_diff)
                    
            if args.task in TASK_BUTTON:
                if i > 0 and i % (rew_log_interval-1) == 0:
                    button_goal_pos = env.button_goal_pos[robot_index, 1:3].cpu().numpy()
                    wrist_pos = env.rigid_state[robot_index, env.wrist_indices, 1:3].cpu().numpy()
                    wrist_pos = wrist_pos[0] # left hand
                    wrist_pos_diff = np.mean(np.abs(button_goal_pos - wrist_pos))
                    logger.log_metrics(
                        {
                            'button_pos_diff': wrist_pos_diff
                        }
                    )
                    print('button_pos_diff:', wrist_pos_diff)
                    
            if args.task in TASK_LIFT:
                if i > 0 and i % (rew_log_interval-1) == 0:
                    box_goal_pos = env.box_goal_pos[robot_index].cpu().numpy()
                    box_pos = env.box_root_states[robot_index, :3].cpu().numpy()
                    box_pos_diff = np.mean(np.abs(box_pos - box_goal_pos)[2:3]) # z axis only
                    logger.log_metrics(
                        {
                            'box_pos_diff': box_pos_diff
                        }
                    )
                    print('box_pos_diff:', box_pos_diff)
            
            if args.task in TASK_BALL:
                if i > 0 and i % (rew_log_interval-1) == 0:
                    ball_goal_pos = env.goal_pos[robot_index].cpu().numpy()
                    ball_pos = env.ball_root_states[robot_index, :3].cpu().numpy()
                    ball_pos_diff = np.mean(np.abs(ball_pos - ball_goal_pos))
                    logger.log_metrics(
                        {
                            'ball_pos_diff': ball_pos_diff
                        }
                    )
                    print('ball_pos_diff:', ball_pos_diff)
            
            if args.task in TASK_CARRY:
                if i > 0 and i % (rew_log_interval-1) == 0:
                    box_goal_pos = env.box_goal_pos[robot_index].cpu().numpy()
                    box_pos = env.box_root_states[robot_index, :3].cpu().numpy()
                    box_pos_diff = np.mean(np.abs(box_pos - box_goal_pos))
                    logger.log_metrics(
                        {
                            'box_pos_diff': box_pos_diff
                        }
                    )
                    print('box_pos_diff:', box_pos_diff)
            
            if args.task in TASK_CABINET:
                if i > 0 and i % (rew_log_interval-1) == 0:
                    arti_obj_dof_diff = (env.arti_obj_dof_state[:, :, 0] - env.arti_obj_dof_goal).cpu().numpy() # [num_envs, 2]
                    arti_obj_dof_diff = np.mean(np.abs(arti_obj_dof_diff))
                    logger.log_metrics(
                        {
                            'arti_obj_dof_diff': arti_obj_dof_diff
                        }
                    )
                    print('arti_obj_dof_diff:', arti_obj_dof_diff)
                
            ### logging end ###

            if i > 0 and i % state_log_interval == 0:
                logger.plot_states()

            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
            if i > 0 and i % rew_log_interval == 0:
                logger.print_rewards()

        fig = logger._plot()
        filename_png = f'{model_name}_{i_rollout}.png'
        fig.savefig(os.path.join(frame_path, filename_png))
        video.release()
        import pdb; pdb.set_trace()
        del logger

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = True
    EGO_CENTRIC = False
    FIX_COMMAND = True
    args = get_args(test=True)
    play(args)
