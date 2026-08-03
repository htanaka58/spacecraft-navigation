import sys

import numpy as np
import pygame


WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700

BACKGROUND_COLOR = (8, 12, 25)
SPACECRAFT_COLOR = (235, 240, 255)
TARGET_COLOR = (80, 220, 120)
TEXT_COLOR = (230, 230, 230)
BORDER_COLOR = (80, 90, 115)

SPACECRAFT_RADIUS = 12
TARGET_RADIUS = 18


def clamp_position(
    position: np.ndarray,
    velocity: np.ndarray,
) -> None:
    """Keep the spacecraft inside the window."""

    if position[0] < SPACECRAFT_RADIUS:
        position[0] = SPACECRAFT_RADIUS
        velocity[0] = 0

    if position[0] > WINDOW_WIDTH - SPACECRAFT_RADIUS:
        position[0] = WINDOW_WIDTH - SPACECRAFT_RADIUS
        velocity[0] = 0

    if position[1] < SPACECRAFT_RADIUS:
        position[1] = SPACECRAFT_RADIUS
        velocity[1] = 0

    if position[1] > WINDOW_HEIGHT - SPACECRAFT_RADIUS:
        position[1] = WINDOW_HEIGHT - SPACECRAFT_RADIUS
        velocity[1] = 0


def main() -> None:
    pygame.init()

    screen = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT)
    )
    pygame.display.set_caption("Spacecraft Navigation Simulator")

    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)
    large_font = pygame.font.Font(None, 48)

    spacecraft_position = np.array(
        [150.0, 550.0],
        dtype=np.float32,
    )

    spacecraft_velocity = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    target_position = np.array(
        [750.0, 150.0],
        dtype=np.float32,
    )

    thrust_strength = 0.16
    maximum_speed = 6.0
    drag = 0.995

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        thrust = np.array([0.0, 0.0], dtype=np.float32)

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            thrust[0] -= thrust_strength

        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            thrust[0] += thrust_strength

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            thrust[1] -= thrust_strength

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            thrust[1] += thrust_strength

        spacecraft_velocity += thrust
        spacecraft_velocity *= drag

        spacecraft_velocity = np.clip(
            spacecraft_velocity,
            -maximum_speed,
            maximum_speed,
        )

        spacecraft_position += spacecraft_velocity

        clamp_position(
            spacecraft_position,
            spacecraft_velocity,
        )

        distance_to_target = float(
            np.linalg.norm(
                target_position - spacecraft_position
            )
        )

        waypoint_reached = (
            distance_to_target
            < SPACECRAFT_RADIUS + TARGET_RADIUS
        )

        screen.fill(BACKGROUND_COLOR)

        pygame.draw.rect(
            screen,
            BORDER_COLOR,
            (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT),
            width=4,
        )

        pygame.draw.circle(
            screen,
            TARGET_COLOR,
            target_position.astype(int),
            TARGET_RADIUS,
            width=3,
        )

        pygame.draw.circle(
            screen,
            SPACECRAFT_COLOR,
            spacecraft_position.astype(int),
            SPACECRAFT_RADIUS,
        )

        velocity_end = (
            spacecraft_position
            + spacecraft_velocity * 10
        ).astype(int)

        pygame.draw.line(
            screen,
            SPACECRAFT_COLOR,
            spacecraft_position.astype(int),
            velocity_end,
            width=2,
        )

        distance_text = font.render(
            f"Distance to target: {distance_to_target:.1f}",
            True,
            TEXT_COLOR,
        )

        speed = float(np.linalg.norm(spacecraft_velocity))

        speed_text = font.render(
            f"Speed: {speed:.2f}",
            True,
            TEXT_COLOR,
        )

        controls_text = font.render(
            "Controls: WASD or arrow keys",
            True,
            TEXT_COLOR,
        )

        screen.blit(distance_text, (20, 20))
        screen.blit(speed_text, (20, 55))
        screen.blit(controls_text, (20, 90))

        if waypoint_reached:
            success_text = large_font.render(
                "Waypoint reached!",
                True,
                TARGET_COLOR,
            )

            success_rectangle = success_text.get_rect(
                center=(WINDOW_WIDTH // 2, 70)
            )

            screen.blit(success_text, success_rectangle)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()