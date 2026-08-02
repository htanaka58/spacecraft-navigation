import sys

import numpy as np
import pygame


WINDOW_SIZE = 800
WORLD_MIN = -1.0
WORLD_MAX = 1.0

BACKGROUND_COLOR = (10, 15, 30)
SPACECRAFT_COLOR = (240, 240, 255)
TARGET_COLOR = (70, 220, 120)
TEXT_COLOR = (230, 230, 230)


def world_to_screen(position: np.ndarray) -> tuple[int, int]:
    """Convert a world position from [-1, 1] into screen pixels."""

    x_normalized = (position[0] - WORLD_MIN) / (WORLD_MAX - WORLD_MIN)
    y_normalized = (position[1] - WORLD_MIN) / (WORLD_MAX - WORLD_MIN)

    screen_x = int(x_normalized * WINDOW_SIZE)

    # Pygame's y-axis points downward, so invert the world y-coordinate.
    screen_y = int((1.0 - y_normalized) * WINDOW_SIZE)

    return screen_x, screen_y


def main() -> None:
    pygame.init()

    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("2D Spacecraft Navigation")

    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)

    spacecraft_position = np.array([-0.7, -0.5], dtype=np.float32)
    spacecraft_velocity = np.zeros(2, dtype=np.float32)
    target_position = np.array([0.6, 0.5], dtype=np.float32)

    thrust_strength = 0.0015
    drag = 0.995
    maximum_speed = 0.02

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        thrust = np.zeros(2, dtype=np.float32)

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            thrust[0] -= thrust_strength

        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            thrust[0] += thrust_strength

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            thrust[1] -= thrust_strength

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            thrust[1] += thrust_strength

        spacecraft_velocity += thrust
        spacecraft_velocity *= drag

        spacecraft_velocity = np.clip(
            spacecraft_velocity,
            -maximum_speed,
            maximum_speed,
        )

        spacecraft_position += spacecraft_velocity

        # Prevent the spacecraft from leaving the simulation area.
        for axis in range(2):
            if spacecraft_position[axis] <= WORLD_MIN:
                spacecraft_position[axis] = WORLD_MIN
                spacecraft_velocity[axis] = 0.0

            if spacecraft_position[axis] >= WORLD_MAX:
                spacecraft_position[axis] = WORLD_MAX
                spacecraft_velocity[axis] = 0.0

        distance_to_target = float(
            np.linalg.norm(target_position - spacecraft_position)
        )

        screen.fill(BACKGROUND_COLOR)

        target_screen_position = world_to_screen(target_position)
        spacecraft_screen_position = world_to_screen(spacecraft_position)

        pygame.draw.circle(
            screen,
            TARGET_COLOR,
            target_screen_position,
            18,
        )

        pygame.draw.circle(
            screen,
            SPACECRAFT_COLOR,
            spacecraft_screen_position,
            12,
        )

        distance_text = font.render(
            f"Distance to target: {distance_to_target:.3f}",
            True,
            TEXT_COLOR,
        )

        controls_text = font.render(
            "Controls: Arrow keys or WASD",
            True,
            TEXT_COLOR,
        )

        screen.blit(distance_text, (20, 20))
        screen.blit(controls_text, (20, 55))

        if distance_to_target < 0.08:
            success_text = font.render(
                "Waypoint reached!",
                True,
                TARGET_COLOR,
            )

            screen.blit(
                success_text,
                (
                    WINDOW_SIZE // 2 - success_text.get_width() // 2,
                    80,
                ),
            )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()