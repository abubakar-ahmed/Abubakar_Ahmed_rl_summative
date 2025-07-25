import gymnasium as gym
import imageio
import numpy as np
import pygame
import sys
import os
import time

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.f1_env import Formula1PathEnv
from environment.rendering import RaceTrackRenderer


def run_random_agent_demo(episodes=3, max_steps_per_episode=100, create_gif=True):
    """
    Run the F1 environment with a random agent for demonstration.
    
    Args:
        episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        create_gif: Whether to create and save a GIF
    """
    print("Starting Formula 1 Path Optimization Demo")
    print("=" * 50)
    
    # Initialize environment and renderer
    env = Formula1PathEnv()
    renderer = RaceTrackRenderer()
    
    # Storage for GIF frames
    all_frames = []
    episode_stats = []
    
    try:
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")
            print("-" * 30)
            
            # Reset environment
            state, info = env.reset()
            total_reward = 0
            episode_frames = []
            
            # Episode loop
            for step in range(max_steps_per_episode):
                # Take random action
                action = env.action_space.sample()
                
                # Execute action
                next_state, reward, terminated, truncated, step_info = env.step(action)
                total_reward += reward
                
                # Create info dictionary for renderer
                render_info = {
                    'on_track': step_info.get('on_track', True),
                    'distance_to_goal': step_info.get('distance_to_goal', 0),
                    'steps_taken': step + 1,
                    'success_rate': step_info.get('success_rate', 0)
                }
                
                # Render frame
                frame = renderer.render(state, step + 1, reward, total_reward, render_info)
                
                if create_gif:
                    episode_frames.append(frame.copy())
                
                # Print step information
                action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
                print(f"  Step {step+1:2d}: {action_names[action]:>5} → "
                      f"({state[1]},{state[0]}) | "
                      f"Reward: {reward:+5.1f} | "
                      f"Total: {total_reward:+6.1f} | "
                      f"{'✅ ON TRACK' if render_info['on_track'] else '❌ OFF TRACK'}")
                
                # Update state
                state = next_state
                
                # Check if episode ended
                if terminated:
                    print(f"  SUCCESS! Reached finish line in {step + 1} steps!")
                    # Add a few extra frames to show the victory
                    for _ in range(5):
                        frame = renderer.render(state, step + 1, reward, total_reward, render_info)
                        if create_gif:
                            episode_frames.append(frame.copy())
                    break
                elif truncated:
                    print(f"  Episode timeout after {step + 1} steps")
                    break
                
                # Small delay for human viewing
                time.sleep(0.1)
            
            # Episode summary
            episode_stats.append({
                'episode': episode + 1,
                'steps': step + 1,
                'total_reward': total_reward,
                'success': terminated,
                'final_position': tuple(state)
            })
            
            print(f" Episode Summary: {step + 1} steps, "
                  f"{total_reward:.1f} total reward, "
                  f"{'SUCCESS' if terminated else 'TIMEOUT'}")
            
            # Add episode frames to overall collection
            if create_gif:
                all_frames.extend(episode_frames)
                # Add separator frames between episodes
                if episode < episodes - 1:
                    separator_frame = create_episode_separator_frame(episode + 1, episodes)
                    for _ in range(10):  # Show separator for 1 second at 10fps
                        all_frames.append(separator_frame)
        
        # Print overall statistics
        print("\n" + "=" * 50)
        print("SIMULATION STATISTICS")
        print("=" * 50)
        
        successful_episodes = sum(1 for stat in episode_stats if stat['success'])
        total_steps = sum(stat['steps'] for stat in episode_stats)
        avg_reward = sum(stat['total_reward'] for stat in episode_stats) / episodes
        
        print(f"Episodes completed: {episodes}")
        print(f"Successful episodes: {successful_episodes}/{episodes} ({successful_episodes/episodes*100:.1f}%)")
        print(f"Average steps per episode: {total_steps/episodes:.1f}")
        print(f"Average total reward: {avg_reward:.2f}")
        
        print("\nDetailed Results:")
        for stat in episode_stats:
            status = "✅ SUCCESS" if stat['success'] else "❌ TIMEOUT"
            print(f"  Episode {stat['episode']}: {stat['steps']:2d} steps, "
                  f"{stat['total_reward']:+6.1f} reward, "
                  f"ended at {stat['final_position']} - {status}")
        
        # Create and save GIF
        if create_gif and all_frames:
            print(f"\nCreating GIF with {len(all_frames)} frames...")
            
            # Convert frames to the format imageio expects
            gif_frames = []
            for frame in all_frames:
                # Ensure frame is in the correct format (height, width, channels)
                if frame.shape[2] == 3:  # RGB
                    gif_frames.append(frame)
                else:
                    print(f"Warning: Unexpected frame shape: {frame.shape}")
            
            if gif_frames:
                try:
                    imageio.mimsave('random_agent_demo.gif', gif_frames, fps=10, loop=0)
                    print(f"GIF saved as 'random_agent_demo.gif' ({len(gif_frames)} frames)")
                except Exception as e:
                    print(f"Error saving GIF: {e}")
            else:
                print("No valid frames to save")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        renderer.close()
        env.close()
        pygame.quit()
        print("\nDemo completed. Thank you for watching!")


def create_episode_separator_frame(completed_episodes, total_episodes):
    """Create a frame showing episode transition."""
    # This is a simple black frame with text - you could make it more elaborate
    frame = np.zeros((680, 600, 3), dtype=np.uint8)  # Match renderer dimensions
    
    # Add text using pygame (simplified version)
    pygame.init()
    font = pygame.font.SysFont('arial', 36, bold=True)
    
    # Create a temporary surface for text
    temp_surface = pygame.Surface((600, 680))
    temp_surface.fill((0, 0, 0))
    
    text = f"Episode {completed_episodes} Complete"
    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(300, 340))
    temp_surface.blit(text_surface, text_rect)
    
    # Convert to numpy array
    surface_array = pygame.surfarray.array3d(temp_surface)
    frame = np.transpose(surface_array, (1, 0, 2))
    
    return frame


def main():
    """Main function to run the demo."""
    print("Formula 1 Path Optimization RL Simulator")
    print("Objective: Navigate from (0,0) to (5,5) on the racing track")
    print("Agent: Random action selection")
    print("Output: Live visualization + GIF recording")
    
    # Check if user wants to skip GIF creation (for faster testing)
    create_gif = True
    if len(sys.argv) > 1 and sys.argv[1] == '--no-gif':
        create_gif = False
        print("GIF creation disabled")
    
    # Run the demo
    try:
        run_random_agent_demo(episodes=3, max_steps_per_episode=100, create_gif=create_gif)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()