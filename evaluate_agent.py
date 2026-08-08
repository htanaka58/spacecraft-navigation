from stable_baselines3 import PPO

from spacecraft_env import SpacecraftEnv


def main() -> None:
    env = SpacecraftEnv()

    print("Loading trained model...")

    model = PPO.load(
        "models/spacecraft_ppo",
        env=env,
    )

    observation, info = env.reset()

    total_reward = 0.0

    print("\nStarting evaluation...")
    print(
        f"Initial distance: "
        f"{info['distance_to_target']:.2f}"
    )

    for step_number in range(500):

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        if step_number % 20 == 0:
            print(
                f"Step {step_number:03d} | "
                f"Action: {action} | "
                f"Distance: "
                f"{info['distance_to_target']:.2f}"
            )

        if terminated:
            print("\nSUCCESS!")
            print(
                f"Reached waypoint after "
                f"{step_number + 1} steps."
            )
            break

        if truncated:
            print("\nEpisode timed out.")
            break

    print(
        f"\nFinal distance: "
        f"{info['distance_to_target']:.2f}"
    )

    print(
        f"Total reward: {total_reward:.2f}"
    )

    env.close()


if __name__ == "__main__":
    main()