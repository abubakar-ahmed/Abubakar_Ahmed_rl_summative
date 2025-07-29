import os
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.f1_env import ImprovedF1Env  

class F1TrainingCallback(BaseCallback):
    """Fixed callback that properly handles the environment API"""
    
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=10, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        
        # Metrics storage
        self.evaluation_results = {
            'timesteps': [],
            'mean_rewards': [],
            'std_rewards': [],
            'success_rates': [],
            'episode_lengths': []
        }
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluation - use SB3's evaluate_policy which handles API differences
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            
            # Calculate success rate
            success_count = 0
            episode_lengths = []
            
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                episode_length = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, done, info = self.eval_env.step(action)
                    episode_length += 1
                    
                    # Fix: Handle vectorized environment info
                    if done:
                        # info is a list when using vectorized environments
                        if isinstance(info, list):
                            # Get the info dict from the first (and only) environment
                            info_dict = info[0] if len(info) > 0 else {}
                        else:
                            info_dict = info
                        
                        if info_dict.get('is_success', False):
                            success_count += 1
                
                episode_lengths.append(episode_length)
            
            success_rate = success_count / self.n_eval_episodes
            
            # Store metrics
            self.evaluation_results['timesteps'].append(self.num_timesteps)
            self.evaluation_results['mean_rewards'].append(mean_reward)
            self.evaluation_results['std_rewards'].append(std_reward)
            self.evaluation_results['success_rates'].append(success_rate)
            self.evaluation_results['episode_lengths'].append(np.mean(episode_lengths))
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Ensure models directory exists
                os.makedirs("models", exist_ok=True)
                self.model.save(os.path.join("models", "best_model"))
            
            if self.verbose > 0:
                print(f"Step {self.num_timesteps:,}:")
                print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
                print(f"  Success rate: {success_rate:.1%}")
                print(f"  Avg episode length: {np.mean(episode_lengths):.1f}")
        
        return True

def train_policy_agent(algo='ppo', total_timesteps=100000, save_dir="models"):
    """
    Fixed training function that properly handles environment API
    """
    print(f"Training {algo.upper()} agent for {total_timesteps:,} timesteps")
    
    # Create environments with proper wrapping
    train_env = DummyVecEnv([lambda: Monitor(ImprovedF1Env())])
    eval_env = DummyVecEnv([lambda: Monitor(ImprovedF1Env())])
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Algorithm parameters
    if algo.lower() == 'ppo':
        model = PPO(
            'MlpPolicy',
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=os.path.join(save_dir, "tensorboard")
        )
    elif algo.lower() == 'a2c':
        model = A2C(
            'MlpPolicy',
            train_env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=os.path.join(save_dir, "tensorboard")
        )
    else:
        raise ValueError("Algorithm must be 'ppo' or 'a2c'")

    # Create callback
    callback = F1TrainingCallback(
        eval_env,
        eval_freq=max(10000, total_timesteps//10),
        n_eval_episodes=10
    )

    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
        raise

    # Save the final model
    model_path = os.path.join(save_dir, f"{algo}_f1_final")
    model.save(model_path)
    print(f"Final model saved to: {model_path}")
    
    # Final evaluation
    try:
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=20
        )
        print(f"Final {algo.upper()} performance:")
        print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    except Exception as e:
        print(f"Final evaluation error: {e}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model, callback

if __name__ == "__main__":
    # Fix module import path
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'a2c'])
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--save_dir', type=str, default="models")
    args = parser.parse_args()
    
    model, callback = train_policy_agent(
        algo=args.algo,
        total_timesteps=args.timesteps,
        save_dir=args.save_dir
    )