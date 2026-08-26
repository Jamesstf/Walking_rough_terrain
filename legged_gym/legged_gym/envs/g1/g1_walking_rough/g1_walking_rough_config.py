"""Training configuration for the G1 rough-terrain walking policy."""

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.g1.g1_walking_unitree.g1_walking_unitree_config import (
    G1WalkingUnitreeCfg,
    G1WalkingUnitreeCfgPPO,
)


HEIGHT_POINTS_X = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
HEIGHT_POINTS_Y = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
NUM_HEIGHT_POINTS = len(HEIGHT_POINTS_X) * len(HEIGHT_POINTS_Y)  # 63


class G1WalkingRoughCfg(G1WalkingUnitreeCfg):
    class env(G1WalkingUnitreeCfg.env):
        # Actor remains bit-for-bit compatible with the flat walking policy.
        num_single_obs = 47
        frame_stack = 3
        num_observations = frame_stack * num_single_obs  # 141

        # Critic sees current proprioception (50) plus a 9 x 7 local height map.
        c_frame_stack = 1
        single_num_privileged_obs = 50 + NUM_HEIGHT_POINTS
        num_privileged_obs = c_frame_stack * single_num_privileged_obs  # 113
        episode_length_s = 20

    class terrain(G1WalkingUnitreeCfg.terrain):
        mesh_type = "trimesh"
        curriculum = True
        measure_heights = True
        horizontal_scale = 0.10
        vertical_scale = 0.005
        border_size = 20.0
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measured_points_x = HEIGHT_POINTS_X
        measured_points_y = HEIGHT_POINTS_Y
        center_height_index = (
            HEIGHT_POINTS_X.index(0.0) * len(HEIGHT_POINTS_Y)
            + HEIGHT_POINTS_Y.index(0.0)
        )
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 1
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 10
        # [flat, random rough, up slope, down slope, up stairs,
        #  down stairs, discrete obstacles]
        terrain_proportions = [0.10, 0.20, 0.15, 0.15, 0.15, 0.15, 0.10]
        slope_treshold = 0.75

    class commands(G1WalkingUnitreeCfg.commands):
        curriculum = True
        max_curriculum = 1.0
        resampling_time = 8.0
        heading_command = True

        class ranges:
            # Warm start begins with moderate commands. Existing curriculum code
            # expands forward speed as tracking performance improves.
            lin_vel_x = [-0.35, 0.60]
            lin_vel_y = [-0.30, 0.30]
            ang_vel_yaw = [-0.60, 0.60]
            heading = [-3.14, 3.14]

    class init_state(G1WalkingUnitreeCfg.init_state):
        pos = [0.0, 0.0, 0.82]

    class domain_rand(G1WalkingUnitreeCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.45, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]
        push_robots = True
        push_interval_s = 7.0
        max_push_vel_xy = 0.55

    class rewards(G1WalkingUnitreeCfg.rewards):
        base_height_target = 0.78
        tracking_sigma = 0.25
        max_contact_force = 350.0
        swing_height_target = 0.10

        class scales(G1WalkingUnitreeCfg.rewards.scales):
            tracking_lin_vel = 1.20
            tracking_ang_vel = 0.60
            lin_vel_z = -1.25
            ang_vel_xy = -0.08
            orientation = -1.20
            base_height = -4.0
            torques = -1.5e-5
            dof_acc = -1.5e-7
            dof_vel = -5.0e-4
            feet_air_time = 0.0
            collision = -0.60
            action_rate = -0.02
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -0.50
            contact_no_vel = 0.0
            feet_slip = -0.18
            feet_stumble = -0.45
            feet_contact_forces = -2.0e-4
            feet_swing_height = -8.0
            contact = 0.12

    class noise(G1WalkingUnitreeCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(G1WalkingUnitreeCfg.noise.noise_scales):
            dof_pos = 0.015
            dof_vel = 1.5
            lin_vel = 0.10
            ang_vel = 0.25
            gravity = 0.06
            height_measurements = 0.10

    class viewer(G1WalkingUnitreeCfg.viewer):
        ref_env = 0
        pos = [4.0, -6.0, 3.0]
        lookat = [4.0, 0.0, 0.6]


class G1WalkingRoughCfgPPO(G1WalkingUnitreeCfgPPO):
    seed = 33

    class policy(G1WalkingUnitreeCfgPPO.policy):
        init_noise_std = 0.55
        actor_hidden_dims = [512, 128]
        critic_hidden_dims = [512, 128]
        activation = "elu"

    class algorithm(G1WalkingUnitreeCfgPPO.algorithm):
        entropy_coef = 0.008
        learning_rate = 5.0e-4
        num_learning_epochs = 5
        num_mini_batches = 4
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01

    class runner(G1WalkingUnitreeCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 10000
        save_interval = 500
        experiment_name = "g1_walking_rough"
        run_name = "rough_warmstart_33"
        resume = False
        warm_start = True
        warm_start_path = (
            LEGGED_GYM_ROOT_DIR
            + "/logs/g1_walking_unitree/seed_31_1/model_20000.pt"
        )
