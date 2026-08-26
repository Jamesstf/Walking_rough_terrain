import os


LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LEGGED_GYM_SRC_DIR = os.path.dirname(os.path.realpath(__file__))
LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_SRC_DIR, "envs")
LEGGED_GYM_DEPLOY_DIR = os.path.dirname(LEGGED_GYM_ROOT_DIR)
