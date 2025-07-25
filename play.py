import imageio
import pygame
import numpy as np
from stable_baselines3 import DQN
from environment.f1_env import Formula1PathEnv
from environment.rendering import RaceTrackRenderer

# Load the trained model
model = DQN.load("models/dqn/dqn.zip")

# Create environment and renderer
env = Formula1PathEnv()
renderer = RaceTrackRenderer()

# For saving GIF frames
frames = []

# Reset the environment
obs, info = env.reset()
done = False
total_reward = 0
step = 0

while not done:
    # Predict action using trained model
    action, _ = model.predict(obs, deterministic=True)
    
    # Step through environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    step += 1

    # Render live (pygame display)
    renderer.render(obs, step, reward, total_reward, info)

    # Capture frame for GIF
    surface = pygame.display.get_surface()
    if surface:  # Ensure surface exists
        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2)) 
        frames.append(frame)

    # Add a delay for live playback (100ms = 10 FPS)
    pygame.time.wait(100)

# Save to GIF
gif_path = "agent_run.gif"
imageio.mimsave(gif_path, frames, duration=0.15) 
print(f"🎞️ Saved GIF to {gif_path}")

# Clean up
renderer.close()
env.close()
