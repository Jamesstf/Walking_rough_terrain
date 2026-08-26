"""Difficulty-scaled terrain curriculum for G1 rough-terrain walking."""

from isaacgym import terrain_utils

from legged_gym.utils.terrain import Terrain


class G1RoughTerrain(Terrain):
    """
    Terrain columns contain different families and rows increase difficulty.

    The curriculum deliberately excludes gaps and pits in the first version:
    those are foothold-planning tasks and are unnecessarily destructive for a
    proprioceptive policy that has no exteroceptive terrain input.
    """

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "g1_rough_terrain",
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )

        # All families retain a sufficiently large, flat spawn platform.
        platform_size = 2.0
        rough_height = 0.01 + 0.045 * difficulty
        slope = 0.04 + 0.20 * difficulty
        step_height = 0.025 + 0.11 * difficulty
        obstacle_height = 0.02 + 0.09 * difficulty

        if choice < self.proportions[0]:
            # Flat patches stabilize early training and prevent forgetting the
            # nominal walking gait during warm start.
            pass
        elif choice < self.proportions[1]:
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-rough_height,
                max_height=rough_height,
                step=0.005,
                downsampled_scale=0.20,
            )
        elif choice < self.proportions[2]:
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=platform_size
            )
        elif choice < self.proportions[3]:
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=-slope, platform_size=platform_size
            )
        elif choice < self.proportions[4]:
            terrain_utils.pyramid_stairs_terrain(
                terrain,
                step_width=0.35,
                step_height=step_height,
                platform_size=platform_size,
            )
        elif choice < self.proportions[5]:
            terrain_utils.pyramid_stairs_terrain(
                terrain,
                step_width=0.35,
                step_height=-step_height,
                platform_size=platform_size,
            )
        else:
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                max_height=obstacle_height,
                min_size=0.45,
                max_size=1.20,
                num_rects=24,
                platform_size=platform_size,
            )
        return terrain
