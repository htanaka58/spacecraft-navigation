from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SpacecraftEnv(gym.Env):
    """A basic 2D spacecraft navigation environment."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()

        # 0 = no thrust
        # 1 = thrust left
        # 2 = thrust right
        # 3 = thrust up
        # 4 = thrust down
        self.action_space = spaces.Discrete(5)

        # Observation:
        # [spacecraft_x, spacecraft_y,
        #  velocity_x, velocity_y,
        #  target_x, target_y]
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, -6, -6, 0, 0],
                dtype=np.float32,
            ),
            high=np.array(
                [900, 700, 6, 6, 900, 700],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.window_width = 900
        self.window_height = 700

        self.spacecraft_radius = 12
        self.target_radius = 18

        self.thrust_strength = 0.16
        self.maximum_speed = 6.0
        self.drag = 0.995

        self.maximum_steps = 500
        self.current_step = 0

        self.spacecraft_position = np.zeros(
            2,
            dtype=np.float32,
        )

        self.spacecraft_velocity = np.zeros(
            2,
            dtype=np.float32,
        )

        self.target_position = np.zeros(
            2,
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        """Return the current environment observation."""

        return np.array(
            [
                self.spacecraft_position[0],
                self.spacecraft_position[1],
                self.spacecraft_velocity[0],
                self.spacecraft_velocity[1],
                self.target_position[0],
                self.target_position[1],
            ],
            dtype=np.float32,
        )

    def _get_distance_to_target(self) -> float:
        """Calculate the distance from spacecraft to target."""

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
        """Start a new episode."""

        super().reset(seed=seed)

        self.spacecraft_position = np.array(
            [150.0, 550.0],
            dtype=np.float32,
        )

        self.spacecraft_velocity = np.array(
            [0.0, 0.0],
            dtype=np.float32,
        )

        self.target_position = np.array(
            [750.0, 150.0],
            dtype=np.float32,
        )

        self.current_step = 0

        return self._get_observation(), self._get_info()

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
        """Advance the simulation by one time step."""

        self.current_step += 1

        thrust = np.array(
            [0.0, 0.0],
            dtype=np.float32,
        )

        if action == 1:
            thrust[0] -= self.thrust_strength

        elif action == 2:
            thrust[0] += self.thrust_strength

        elif action == 3:
            thrust[1] -= self.thrust_strength

        elif action == 4:
            thrust[1] += self.thrust_strength

        self.spacecraft_velocity += thrust
        self.spacecraft_velocity *= self.drag

        self.spacecraft_velocity = np.clip(
            self.spacecraft_velocity,
            -self.maximum_speed,
            self.maximum_speed,
        )

        self.spacecraft_position += (
            self.spacecraft_velocity
        )

        self._keep_spacecraft_in_bounds()

        distance = self._get_distance_to_target()

        reached_target = (
            distance
            < self.spacecraft_radius + self.target_radius
        )

        exceeded_time_limit = (
            self.current_step >= self.maximum_steps
        )

        terminated = reached_target
        truncated = exceeded_time_limit

        # The agent receives a negative reward when far away.
        reward = -distance / 100.0

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
        """Prevent the spacecraft from leaving the world."""

        minimum_x = float(self.spacecraft_radius)
        maximum_x = float(
            self.window_width - self.spacecraft_radius
        )

        minimum_y = float(self.spacecraft_radius)
        maximum_y = float(
            self.window_height - self.spacecraft_radius
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