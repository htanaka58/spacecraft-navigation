import sys

import numpy as np
import pygame
from stable_baselines3 import PPO

from spacecraft_env import SpacecraftEnv


WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700

BACKGROUND_COLOR = (8, 12, 25)
SPACECRAFT_COLOR = (235, 240, 255)
TARGET_COLOR = (80, 220, 120)
TEXT_COLOR = (230, 230, 230)
VELOCITY_COLOR = (100, 180, 255)

ACTION_NAMES = {
    0: "COAST",
    1: "LEFT",
    2: "RIGHT",
    3: "UP",
    4: "DOWN",
}


def main():
    # -------------------------
    # Start Pygame
    # -------------------------

    pygame.init()

    screen = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT)
    )

    pygame.display.set_caption(
        "PPO Spacecraft Navigation"
    )

    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 30)
    large_font = pygame.font.Font(None, 48)

    # -------------------------
    # Create environment
    # -------------------------

    env = SpacecraftEnv()

    observation, info = env.reset()

    # -------------------------
    # Load trained PPO agent
    # -------------------------

    print("Loading PPO agent...")

    model = PPO.load(
        "models/spacecraft_ppo"
    )

    print("Agent loaded!")
    print("Starting autonomous flight...")

    running = True

    terminated = False
    truncated = False

    current_action = 0

    # -------------------------
    # Simulation loop
    # -------------------------

    while running:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            # Press R to restart
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_r:
                    observation, info = env.reset()

                    terminated = False
                    truncated = False

        # -------------------------
        # AI chooses action
        # -------------------------

        if not terminated and not truncated:

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            current_action = int(action)

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(current_action)

        # -------------------------
        # Draw background
        # -------------------------

        screen.fill(BACKGROUND_COLOR)

        # -------------------------
        # Draw target
        # -------------------------

        target = env.target_position.astype(int)

        pygame.draw.circle(
            screen,
            TARGET_COLOR,
            target,
            env.target_radius,
            width=3,
        )

        # -------------------------
        # Draw spacecraft
        # -------------------------

        spacecraft = (
            env.spacecraft_position.astype(int)
        )

        pygame.draw.circle(
            screen,
            SPACECRAFT_COLOR,
            spacecraft,
            env.spacecraft_radius,
        )

        # -------------------------
        # Draw velocity vector
        # -------------------------

        velocity_end = (
            env.spacecraft_position
            + env.spacecraft_velocity * 10
        ).astype(int)

        pygame.draw.line(
            screen,
            VELOCITY_COLOR,
            spacecraft,
            velocity_end,
            width=3,
        )

        # -------------------------
        # Information
        # -------------------------

        distance = info["distance_to_target"]

        speed = float(
            np.linalg.norm(
                env.spacecraft_velocity
            )
        )

        action_text = ACTION_NAMES[
            current_action
        ]

        lines = [
            "PPO AUTONOMOUS CONTROL",
            f"Distance: {distance:.1f}",
            f"Speed: {speed:.2f}",
            f"Action: {action_text}",
            f"Step: {env.current_step}",
            "Press R to restart",
        ]

        y = 20

        for line in lines:

            text = font.render(
                line,
                True,
                TEXT_COLOR,
            )

            screen.blit(
                text,
                (20, y),
            )

            y += 32

        # -------------------------
        # Mission success
        # -------------------------

        if terminated:

            success_text = large_font.render(
                "WAYPOINT REACHED",
                True,
                TARGET_COLOR,
            )

            rectangle = success_text.get_rect(
                center=(
                    WINDOW_WIDTH // 2,
                    WINDOW_HEIGHT // 2,
                )
            )

            screen.blit(
                success_text,
                rectangle,
            )

        elif truncated:

            failure_text = large_font.render(
                "MISSION TIMEOUT",
                True,
                TEXT_COLOR,
            )

            rectangle = failure_text.get_rect(
                center=(
                    WINDOW_WIDTH // 2,
                    WINDOW_HEIGHT // 2,
                )
            )

            screen.blit(
                failure_text,
                rectangle,
            )

        pygame.display.flip()

        # Slow simulation down so
        # we can actually watch it.
        clock.tick(30)

    env.close()

    pygame.quit()

    sys.exit()


if __name__ == "__main__":
    main()