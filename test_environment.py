from stable_baselines3.common.env_checker import check_env

from spacecraft_env import SpacecraftEnv


def main() -> None:
    env = SpacecraftEnv()

    print("Checking environment...")

    check_env(
        env,
        warn=True,
    )

    print("Environment check passed!")

    observation, info = env.reset(seed=42)

    print("\nInitial observation:")
    print(observation)

    print("\nInitial information:")
    print(info)

    print("\nRunning random actions:")

    for step_number in range(20):
        action = env.action_space.sample()

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        print(
            f"Step {step_number + 1:02d} | "
            f"Action: {action} | "
            f"Position: {observation[:2]} | "
            f"Velocity: {observation[2:4]} | "
            f"Distance: "
            f"{info['distance_to_target']:.1f} | "
            f"Reward: {reward:.2f}"
        )

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()