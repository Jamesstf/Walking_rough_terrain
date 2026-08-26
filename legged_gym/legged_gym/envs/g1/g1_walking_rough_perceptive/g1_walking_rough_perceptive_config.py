"""Configuration for perceptive G1 rough-terrain locomotion."""

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.g1.g1_walking_rough.g1_walking_rough_config import (
    NUM_HEIGHT_POINTS,
    G1WalkingRoughCfg,
    G1WalkingRoughCfgPPO,
)


PROPRIO_OBS_DIM = 141
TERRAIN_OBS_DIM = NUM_HEIGHT_POINTS


class G1WalkingRoughPerceptiveCfg(G1WalkingRoughCfg):
    class env(G1WalkingRoughCfg.env):
        # Keep each proprioceptive frame at 47D and append one current 63D map
        # only after the three frames have been stacked.
        num_single_obs = 47
        frame_stack = 3
        num_proprio_observations = PROPRIO_OBS_DIM
        num_terrain_observations = TERRAIN_OBS_DIM
        num_observations = PROPRIO_OBS_DIM + TERRAIN_OBS_DIM  # 204

        # The critic receives exact current state and exact terrain as before.
        c_frame_stack = 1
        single_num_privileged_obs = 50 + NUM_HEIGHT_POINTS
        num_privileged_obs = single_num_privileged_obs  # 113

    class perception:
        # "oracle" gives exact local heights; "noisy" activates all sensor
        # degradation below. Training defaults to the deployable noisy mode.
        mode = "noisy"
        max_delay_steps = 5
        randomize_delay = True
        noise_std_range = [0.0, 0.02]  # metres, sampled once per episode
        max_height_bias = 0.015  # metres, correlated across all map points
        dropout_probability = 0.08
        dropout_fill_value = 0.0
        quantization = 0.005  # metres
        clip_height = 0.50  # metres


class G1WalkingRoughPerceptiveCfgPPO(G1WalkingRoughCfgPPO):
    seed = 34

    class policy(G1WalkingRoughCfgPPO.policy):
        proprio_obs_dim = PROPRIO_OBS_DIM
        terrain_obs_dim = TERRAIN_OBS_DIM
        terrain_encoder_dims = [128, 32]
        actor_hidden_dims = [512, 128]
        critic_hidden_dims = [512, 128]
        activation = "elu"

    class runner(G1WalkingRoughCfgPPO.runner):
        policy_class_name = "ActorCriticPerceptive"
        experiment_name = "g1_walking_rough_perceptive"
        run_name = "noisy_heightmap_34"
        warm_start = True
        # The file can be overridden with G1_PERCEPTIVE_WARM_START. It is
        # intentionally separate from the currently running rough training.
        warm_start_path = (
            LEGGED_GYM_ROOT_DIR
            + "/logs/g1_walking_rough/rough_warmstart_33/model_10000.pt"
        )

