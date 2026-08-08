from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from spacecraft_env import SpacecraftEnv


def main() -> None:
    print("Creating spacecraft environment...")

    env = SpacecraftEnv()
    env = Monitor(env)

    print("Creating PPO agent...")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    print("Starting training...")

    model.learn(
        total_timesteps=300_000,
        progress_bar=True,
    )

    print("Training complete!")

    model.save("models/spacecraft_ppo")

    print("Model saved to models/spacecraft_ppo")

    env.close()


if __name__ == "__main__":
    main()