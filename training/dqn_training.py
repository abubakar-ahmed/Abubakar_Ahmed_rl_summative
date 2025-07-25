import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.f1_env import Formula1PathEnv


class SimpleEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=5000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.timesteps = []
        self.rewards = []
        self.success_rates = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=10, deterministic=True
            )
            self.timesteps.append(self.n_calls)
            self.rewards.append(mean_reward)

            # Success rate = how often it reached the goal
            successes = 0
            for _ in range(10):
                obs, _ = self.eval_env.reset()
                done = False
                for _ in range(100):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _, _ = self.eval_env.step(action)
                    if done:
                        successes += 1
                        break
            self.success_rates.append(successes / 10)

            if self.verbose:
                print(f"Step {self.n_calls}: Reward={mean_reward:.2f}, Success={successes}/10")
        return True


def train_dqn(total_timesteps=50000):
    train_env = Formula1PathEnv()
    eval_env = Formula1PathEnv()

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=5e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.15,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./logs/"
    )

    callback = SimpleEvalCallback(eval_env, eval_freq=5000, verbose=1)

    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save("models/dqn/dqn.zip")
    print("Model saved to models/dqn/")

    plot_training(callback)


def plot_training(callback):
    if not callback.timesteps:
        print("No data to plot")
        return

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(callback.timesteps, callback.rewards, label="Mean Reward")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Reward Over Time")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(callback.timesteps, [s * 100 for s in callback.success_rates], label="Success Rate (%)")
    plt.xlabel("Timesteps")
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rate Over Time")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("models/training_plot.png", dpi=300)
    plt.show()
    print("Plot saved to models/training_plot.png")


if __name__ == "__main__":
    train_dqn()
