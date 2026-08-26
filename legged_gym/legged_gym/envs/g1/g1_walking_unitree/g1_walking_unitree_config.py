from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1WalkingUnitreeCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_actions = 12
        frame_stack = 3
        c_frame_stack = 1
        command_dim = 3
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 50
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        test = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10.
        heading_command = True

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1, 1]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8]
        default_joint_angles = {
            'left_hip_yaw_joint': 0.,
            'left_hip_roll_joint': 0,
            'left_hip_pitch_joint': -0.1,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0,
            'right_hip_yaw_joint': 0.,
            'right_hip_roll_joint': 0,
            'right_hip_pitch_joint': -0.1,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0,
            'torso_joint': 0.,
        }

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {
            'hip_yaw': 100,
            'hip_roll': 100,
            'hip_pitch': 100,
            'knee': 150,
            'ankle': 40,
        }
        damping = {
            'hip_yaw': 2,
            'hip_roll': 2,
            'hip_pitch': 2,
            'knee': 4,
            'ankle': 2,
        }
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof_modified.urdf'  # g1_12dof_modified.urdf
        name = "g1"
        foot_name = "ankle_roll"
        knee_name = "knee"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.78
        tracking_sigma = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0  # 1.0
            tracking_ang_vel = 0.5  # 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05  # -0.05
            orientation = -1.0  # -1.0
            base_height = -10.0
            torques = -0.00001  # -0.00001
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2  # -0.2
            feet_swing_height = -20.0
            contact = 0.18  # 0.18

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.
        clip_actions = 100.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1


class G1WalkingUnitreeCfgPPO(LeggedRobotCfgPPO):
    seed = 32
    runner_class_name = 'OnPolicyRunner'  # DWLOnPolicyRunner
    class policy:
        # init_noise_std = 0.8
        # actor_hidden_dims = [32]
        # critic_hidden_dims = [32]
        # activation = 'elu'
        # rnn_type = 'lstm'
        # rnn_hidden_size = 64
        # rnn_num_layers = 1
        init_noise_std = 0.8
        actor_hidden_dims = [512, 128]
        critic_hidden_dims = [512, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        # entropy_coef = 0.01
        # learning_rate = 1e-4
        # num_learning_epochs = 2
        # gamma = 0.994
        # lam = 0.9
        # num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        # policy_class_name = "ActorCriticRecurrent"
        policy_class_name = "ActorCritic"
        max_iterations = 20000
        save_interval = 1000
        run_name = ''
        experiment_name = 'g1_walking_unitree'
