import imageio
import pygame
import numpy as np
from stable_baselines3 import DQN
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.f1_env import ImprovedF1Env
from environment.rendering import ImprovedF1Renderer

def play_model(model_path="models/dqn/f1_dqn_basic", gif_path="dqn_agent_run.gif"):
    """Play the trained model and save as GIF"""
    
    # Load the trained model
    try:
        model = DQN.load(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create environment and renderer
    env = ImprovedF1Env()
    renderer = ImprovedF1Renderer(grid_size=env.grid_size)
    
    # For saving GIF frames
    frames = []
    
    # Reset the environment
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    print("Starting simulation... (Press ESC to exit early)")
    
    while not done:
        # Predict action using trained model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step through environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        # Prepare rendering data
        pos_state = obs[:2].astype(int)
        training_info = {
            'completion_rate': info.get('progress', 0),
            'episode_count': 1
        }

        # Render live
        renderer.render(
            state=pos_state,
            step=step,
            reward=reward,
            total_reward=total_reward,
            info=info,
            training_info=training_info
        )

        # Capture frame for GIF
        surface = pygame.display.get_surface()
        if surface:
            frame = pygame.surfarray.array3d(surface)
            frame = np.transpose(frame, (1, 0, 2)) 
            frames.append(frame)

        # Handle pygame events (for early exit)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                done = True

        # Control playback speed (100ms = 10 FPS)
        pygame.time.wait(100)

    # Save to GIF if we captured frames
    if frames:
        try:
            imageio.mimsave(gif_path, frames, duration=0.1)
            print(f"Saved GIF to {gif_path}")
        except Exception as e:
            print(f"Error saving GIF: {e}")

    # Clean up
    renderer.close()
    env.close()
    print(f"Simulation complete. Total reward: {total_reward:.1f}")

if __name__ == "__main__":
    play_model()