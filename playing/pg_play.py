import os
import sys
import imageio
import pygame
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.f1_env import ImprovedF1Env
from environment.rendering import ImprovedF1Renderer

class F1ModelPlayer:
    """Fixed player for trained F1 policy gradient models."""
    
    def __init__(self):
        self.env = None
        self.renderer = None
        self.frames = []
        
    def load_model(self, model_path, algorithm='PPO'):
        """Load a trained model."""
        try:
            if algorithm.upper() == 'PPO':
                model = PPO.load(model_path)
            elif algorithm.upper() == 'A2C':
                model = A2C.load(model_path)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def setup_environment(self):
        """Setup the environment and renderer."""
        try:
            self.env = DummyVecEnv([lambda: ImprovedF1Env()])
            self.renderer = ImprovedF1Renderer()
            return True
        except Exception as e:
            print(f"Environment setup failed: {e}")
            return False
    
    def play_episode(self, model, algorithm_name, record_gif=True, fps=10):
        """Play a single episode with the trained model."""
        if not self.env or not self.renderer:
            return None
            
        obs = self.env.reset()
        done = False
        total_reward = 0
        step = 0
        self.frames = []
        
        while not done and step < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            
            # Handle vectorized environment returns
            reward_val = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            info_dict = info[0] if isinstance(info, list) and len(info) > 0 else {}
                
            total_reward += reward_val
            step += 1
            
            # Get car position
            try:
                if hasattr(self.env.envs[0], 'get_car_position'):
                    car_position = self.env.envs[0].get_car_position()
                else:
                    car_position = obs[0][:2] if len(obs[0]) >= 2 else [0, 0]
            except:
                car_position = [0, 0]
            
            # Render and capture frames
            try:
                self.renderer.render(
                    state=car_position,
                    step=step,
                    reward=reward_val,
                    total_reward=total_reward,
                    info=info_dict
                )
                
                if record_gif:
                    surface = pygame.display.get_surface()
                    if surface:
                        frame = pygame.surfarray.array3d(surface)
                        frame = np.transpose(frame, (1, 0, 2))
                        self.frames.append(frame)
                
                pygame.time.wait(max(1, 1000 // fps))
                
            except Exception as e:
                pass  # Continue without rendering
        
        success = info_dict.get('is_success', False)
        print(f"Steps: {step}, Reward: {total_reward:.2f}, Success: {success}")
        
        return {
            'algorithm': algorithm_name,
            'steps': step,
            'total_reward': total_reward,
            'success': success,
            'frames': len(self.frames)
        }
    
    def save_gif(self, filename, fps=6):
        """Save recorded frames as GIF."""
        if not self.frames:
            return False
            
        try:
            output_dir = os.path.dirname(filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            imageio.mimsave(filename, self.frames, fps=fps, duration=1/fps)
            print(f"GIF saved: {filename}")
            return True
        except Exception as e:
            print(f"Error saving GIF: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        if self.renderer:
            self.renderer.close()
        if self.env:
            self.env.close()

def find_model_files():
    """Find available model files in the models directory."""
    model_files = []
    model_dirs = ["models", "models/pg", "."]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.zip'):
                    full_path = os.path.join(model_dir, file)
                    
                    # Determine algorithm from filename
                    if 'ppo' in file.lower():
                        algorithm = 'PPO'
                    elif 'a2c' in file.lower():
                        algorithm = 'A2C'
                    else:
                        # Try to infer from file structure
                        algorithm = 'PPO'  # Default
                    
                    model_files.append((full_path, algorithm, file))
    
    return model_files

def main():
    """Main function for playing trained models."""
    print("F1 Model Player")
    
    # Find available models
    available_models = find_model_files()
    
    if not available_models:
        print("No model files found!")
        return
    
    # Initialize player
    player = F1ModelPlayer()
    
    if not player.setup_environment():
        print("Environment setup failed.")
        return
    
    try:
        # Test each model
        for model_path, algorithm, filename in available_models:
            print(f"\nTesting {algorithm}: {filename}")
            
            model = player.load_model(model_path, algorithm)
            if model:
                result = player.play_episode(model, algorithm, record_gif=True, fps=8)
                
                if result and result['frames'] > 0:
                    gif_filename = f"gifs/{algorithm.lower()}_demo.gif"
                    os.makedirs('gifs', exist_ok=True)
                    player.save_gif(gif_filename, fps=6)
        
    except KeyboardInterrupt:
        print("\nCancelled by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        player.cleanup()

if __name__ == "__main__":
    main()