"""G1 proprioceptive walking policy for curriculum rough terrain."""

from .g1_walking_rough import G1WalkingRough
from .g1_walking_rough_config import G1WalkingRoughCfg, G1WalkingRoughCfgPPO

__all__ = ["G1WalkingRough", "G1WalkingRoughCfg", "G1WalkingRoughCfgPPO"]
