import os
import imageio
import pygame
import numpy as np
import torch
from training.reinforce_training import REINFORCEAgent
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.f1_env import ImprovedF1Env
from environment.rendering import ImprovedF1Renderer

def visualize_agent(model_path="models/reinforce/reinforce.zip", gif_path="reinforce_agent_run.gif"):
    """Visualize the trained REINFORCE agent"""
    
    # Initialize environment and agent
    env = ImprovedF1Env()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = REINFORCEAgent(state_size, action_size)
    
    try:
        agent.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize rendering
    renderer = ImprovedF1Renderer()
    frames = []
    
    # Run episode
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    max_steps = 200

    print("Running agent visualization... (ESC to exit early)")
    
    with torch.no_grad():
        agent.policy.eval()
        while not done and step < max_steps:
            # Get action
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_probs = agent.policy(state_tensor)
            action = torch.argmax(action_probs).item()
            
            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Extract position from observation (assuming first 2 elements are x,y)
            car_position = obs[:2].astype(int)  # This may need adjustment based on your state representation
            
            # Render
            renderer.render(
                state=car_position,  # Pass just the position coordinates
                step=step,
                reward=reward,
                total_reward=total_reward,
                info=info
            )

            # Capture frame
            surface = pygame.display.get_surface()
            if surface:
                frame = pygame.surfarray.array3d(surface)
                frame = np.transpose(frame, (1, 0, 2)) 
                frames.append(frame)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    done = True
            
            pygame.time.wait(100)  # 10 FPS
            step += 1
            obs = next_obs

    # Results and cleanup
    result = "SUCCESS" if terminated else ("TIMEOUT" if step >= max_steps else "TRUNCATED")
    print(f"Episode finished: {result}")
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.2f}")

    if frames:
        try:
            imageio.mimsave(gif_path, frames, duration=0.1)
            print(f"Saved visualization to {gif_path}")
        except Exception as e:
            print(f"Error saving GIF: {e}")

    renderer.close()
    env.close()

if __name__ == "__main__":
    visualize_agent()