"""Perceptive rough-terrain walking task for Unitree G1."""

from .g1_walking_rough_perceptive import G1WalkingRoughPerceptive
from .g1_walking_rough_perceptive_config import (
    G1WalkingRoughPerceptiveCfg,
    G1WalkingRoughPerceptiveCfgPPO,
)

__all__ = [
    "G1WalkingRoughPerceptive",
    "G1WalkingRoughPerceptiveCfg",
    "G1WalkingRoughPerceptiveCfgPPO",
]

