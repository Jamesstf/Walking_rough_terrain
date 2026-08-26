"""Focused task registry for G1 rough-terrain locomotion."""

from .g1.g1_walking_unitree.g1_walking_unitree import G1WalkingUnitree
from .g1.g1_walking_unitree.g1_walking_unitree_config import (
    G1WalkingUnitreeCfg,
    G1WalkingUnitreeCfgPPO,
)
from .g1.g1_walking_rough.g1_walking_rough import G1WalkingRough
from .g1.g1_walking_rough.g1_walking_rough_config import (
    G1WalkingRoughCfg,
    G1WalkingRoughCfgPPO,
)
from .g1.g1_walking_rough_perceptive.g1_walking_rough_perceptive import (
    G1WalkingRoughPerceptive,
)
from .g1.g1_walking_rough_perceptive.g1_walking_rough_perceptive_config import (
    G1WalkingRoughPerceptiveCfg,
    G1WalkingRoughPerceptiveCfgPPO,
)
from legged_gym.utils.task_registry import task_registry


task_registry.register(
    "g1_walking_unitree",
    G1WalkingUnitree,
    G1WalkingUnitreeCfg(),
    G1WalkingUnitreeCfgPPO(),
    "g1/g1_walking_unitree",
)
task_registry.register(
    "g1_walking_rough",
    G1WalkingRough,
    G1WalkingRoughCfg(),
    G1WalkingRoughCfgPPO(),
    "g1/g1_walking_rough",
)
task_registry.register(
    "g1_walking_rough_perceptive",
    G1WalkingRoughPerceptive,
    G1WalkingRoughPerceptiveCfg(),
    G1WalkingRoughPerceptiveCfgPPO(),
    "g1/g1_walking_rough_perceptive",
)
