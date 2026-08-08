from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SpacecraftEnv(gym.Env):
    """A basic 2D spacecraft navigation environment."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()

        # Actions:
        # 0 = no thrust
        # 1 = thrust left
        # 2 = thrust right
        # 3 = thrust up
        # 4 = thrust down
        self.action_space = spaces.Discrete(5)

        # Observation:
        # [
        #   relative_target_x,
        #   relative_target_y,
        #   velocity_x,
        #   velocity_y
        # ]
        #
        # Everything is normalized to approximately [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # World size
        self.window_width = 900
        self.window_height = 700

        # Object sizes
        self.spacecraft_radius = 12
        self.target_radius = 18

        # Spacecraft physics
        self.thrust_strength = 0.16
        self.maximum_speed = 6.0
        self.drag = 0.995

        # Episode settings
        self.maximum_steps = 500
        self.current_step = 0

        # Used for calculating progress reward
        self.previous_distance = 0.0

        # Spacecraft state
        self.spacecraft_position = np.zeros(
            2,
            dtype=np.float32,
        )

        self.spacecraft_velocity = np.zeros(
            2,
            dtype=np.float32,
        )

        # Target state
        self.target_position = np.zeros(
            2,
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        """
        Return the observation seen by the RL agent.

        Instead of giving the agent absolute coordinates,
        give it the target position relative to the spacecraft.
        """

        relative_position = (
            self.target_position
            - self.spacecraft_position
        )

        observation = np.array(
            [
                relative_position[0] / self.window_width,
                relative_position[1] / self.window_height,
                self.spacecraft_velocity[0] / self.maximum_speed,
                self.spacecraft_velocity[1] / self.maximum_speed,
            ],
            dtype=np.float32,
        )

        return np.clip(
            observation,
            -1.0,
            1.0,
        )

    def _get_distance_to_target(self) -> float:
        """Calculate distance from spacecraft to target."""

        return float(
            np.linalg.norm(
                self.target_position
                - self.spacecraft_position
            )
        )

    def _get_info(self) -> dict:
        """Return useful diagnostic information."""

        return {
            "distance_to_target":
                self._get_distance_to_target(),
            "step": self.current_step,
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Start a new episode.
        """

        super().reset(seed=seed)

        # Fixed starting position for now
        self.spacecraft_position = np.array(
            [150.0, 550.0],
            dtype=np.float32,
        )

        # Start with zero velocity
        self.spacecraft_velocity = np.array(
            [0.0, 0.0],
            dtype=np.float32,
        )

        # Fixed target for now
        self.target_position = np.array(
            [750.0, 150.0],
            dtype=np.float32,
        )

        self.current_step = 0

        # Store starting distance for progress reward
        self.previous_distance = (
            self._get_distance_to_target()
        )

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict,
    ]:
        """
        Advance the simulation by one timestep.
        """

        self.current_step += 1

        # Start each step with no thrust
        thrust = np.array(
            [0.0, 0.0],
            dtype=np.float32,
        )

        # Convert action into thrust
        if action == 1:
            thrust[0] -= self.thrust_strength

        elif action == 2:
            thrust[0] += self.thrust_strength

        elif action == 3:
            thrust[1] -= self.thrust_strength

        elif action == 4:
            thrust[1] += self.thrust_strength

        # Apply thrust
        self.spacecraft_velocity += thrust

        # Apply slight drag
        self.spacecraft_velocity *= self.drag

        # Limit maximum velocity
        self.spacecraft_velocity = np.clip(
            self.spacecraft_velocity,
            -self.maximum_speed,
            self.maximum_speed,
        )

        # Update spacecraft position
        self.spacecraft_position += (
            self.spacecraft_velocity
        )

        # Prevent spacecraft from leaving the world
        self._keep_spacecraft_in_bounds()

        # Calculate new distance
        distance = self._get_distance_to_target()

        # Check whether the target was reached
        reached_target = (
            distance
            < self.spacecraft_radius
            + self.target_radius
        )

        # Check whether episode ran too long
        exceeded_time_limit = (
            self.current_step >= self.maximum_steps
        )

        terminated = reached_target
        truncated = exceeded_time_limit

        # --------------------------------
        # Reward system
        # --------------------------------

        # Positive if spacecraft moved closer
        # Negative if spacecraft moved farther away
        progress = (
            self.previous_distance
            - distance
        )

        reward = progress

        # Small penalty for taking too long
        reward -= 0.01

        # Update previous distance
        self.previous_distance = distance

        # Large reward for reaching target
        if reached_target:
            reward += 100.0

        observation = self._get_observation()
        info = self._get_info()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def _keep_spacecraft_in_bounds(self) -> None:
        """
        Prevent the spacecraft from leaving the simulation area.
        """

        minimum_x = float(
            self.spacecraft_radius
        )

        maximum_x = float(
            self.window_width
            - self.spacecraft_radius
        )

        minimum_y = float(
            self.spacecraft_radius
        )

        maximum_y = float(
            self.window_height
            - self.spacecraft_radius
        )

        if self.spacecraft_position[0] < minimum_x:
            self.spacecraft_position[0] = minimum_x
            self.spacecraft_velocity[0] = 0.0

        if self.spacecraft_position[0] > maximum_x:
            self.spacecraft_position[0] = maximum_x
            self.spacecraft_velocity[0] = 0.0

        if self.spacecraft_position[1] < minimum_y:
            self.spacecraft_position[1] = minimum_y
            self.spacecraft_velocity[1] = 0.0

        if self.spacecraft_position[1] > maximum_y:
            self.spacecraft_position[1] = maximum_y
            self.spacecraft_velocity[1] = 0.0