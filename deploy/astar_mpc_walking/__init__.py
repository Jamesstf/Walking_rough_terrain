"""Shared walking-policy configuration used by the MuJoCo deployment."""

from .config import NavigationConfig, WalkingPolicyConfig, load_config

__all__ = [
    "NavigationConfig",
    "WalkingPolicyConfig",
    "load_config",
]
