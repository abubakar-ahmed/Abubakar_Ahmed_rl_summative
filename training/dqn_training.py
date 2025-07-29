import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.f1_env import ImprovedF1Env
import torch

class BasicCallback(BaseCallback):
    """Simplified callback for tracking training progress"""
    def __init__(self, eval_env, eval_freq=5000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.rewards = []
        self.timesteps = []
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=5)
            self.rewards.append(mean_reward)
            self.timesteps.append(self.n_calls)
            print(f"Step: {self.n_calls}, Mean Reward: {mean_reward:.1f}")
        return True

def train_f1_dqn(total_timesteps=50000):
    """Basic training function for F1 DQN"""
    print("Starting F1 DQN Training")
    
    # Create environments
    train_env = ImprovedF1Env()
    eval_env = ImprovedF1Env()
    
    # Model configuration
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=5e-4,
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log="./logs/f1_dqn/",
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU,
        )
    )
    
    # Training with basic callback
    callback = BasicCallback(eval_env)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/f1_dqn_basic")
    print("Training completed and model saved")
    
    return model

def test_model(model_path="models/f1_dqn_basic", episodes=10):
    """Basic model testing function"""
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print("Testing model...")
    model = DQN.load(model_path)
    env = ImprovedF1Env()
    
    successes = 0
    rewards = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                if terminated:
                    successes += 1
                break
        
        rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward: {episode_reward:.1f}, {'Success' if terminated else 'Timeout'}")
    
    print(f"\nTest Results:")
    print(f"Success Rate: {successes}/{episodes} ({successes/episodes:.1%})")
    print(f"Average Reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")

if __name__ == "__main__":
    # Simple training and testing
    trained_model = train_f1_dqn(total_timesteps=50000)
    test_model()