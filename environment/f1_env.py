import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.f1_env import ImprovedF1Env

def test_basic_functionality():
    """Test basic environment functionality"""
    print("Testing Basic Environment Functionality")
    print("=" * 50)
    
    env = ImprovedF1Env()
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test a few steps
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step+1}: Action={action}, Reward={reward:.2f}, Position={info['position']}")
        
        if terminated or truncated:
            break
    
    print(f"Total reward after {step+1} steps: {total_reward:.2f}")
    env.render()

def test_optimal_path():
    """Test environment with optimal path"""
    print("\n\nTesting Optimal Path")
    print("=" * 50)
    
    env = ImprovedF1Env()
    obs, info = env.reset()
    
    # Optimal path based on track layout
    # Track: [(1,1), (2,1), (3,1), (4,1), (5,1), (5,2), (5,3), (5,4), (4,4), (3,4), (2,4), (1,4), (1,3), (1,2)]
    # Actions: 0=Up, 1=Right, 2=Down, 3=Left
    optimal_actions = [2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 3, 3]
    
    total_reward = 0
    print("Following optimal path...")
    
    for i, action in enumerate(optimal_actions):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, "
              f"Position={info['position']}, Progress={info['progress']:.2%}")
        
        if terminated:
            print(f"\n🏁 Track completed! Total reward: {total_reward:.2f}")
            break
        elif truncated:
            print(f"\n⏰ Time limit reached. Total reward: {total_reward:.2f}")
            break
    
    env.render()
    print(f"Training info: {env.get_training_info()}")

def test_random_agent():
    """Test environment with random agent for multiple episodes"""
    print("\n\nTesting Random Agent (5 episodes)")
    print("=" * 50)
    
    env = ImprovedF1Env()
    episode_results = []
    
    for episode in range(5):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 50:  # Limit steps per episode
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        success = info['progress'] >= 1.0
        episode_results.append({
            'episode': episode + 1,
            'steps': steps,
            'reward': total_reward,
            'progress': info['progress'],
            'success': success
        })
        
        print(f"Episode {episode+1}: Steps={steps}, Reward={total_reward:.2f}, "
              f"Progress={info['progress']:.2%}, Success={success}")
    
    # Summary
    print(f"\nRandom Agent Summary:")
    avg_reward = np.mean([r['reward'] for r in episode_results])
    avg_progress = np.mean([r['progress'] for r in episode_results])
    success_rate = np.mean([r['success'] for r in episode_results])
    
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average progress: {avg_progress:.2%}")
    print(f"Success rate: {success_rate:.2%}")

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("\n\nTesting Edge Cases")
    print("=" * 50)
    
    env = ImprovedF1Env()
    obs, info = env.reset()
    
    print("Testing boundary collisions...")
    
    # Test going out of bounds
    test_cases = [
        (0, "Up - should hit boundary"),
        (0, "Up - should hit boundary again"),
        (3, "Left - should hit boundary"),
        (1, "Right - should move"),
        (2, "Down - should move"),
    ]
    
    total_reward = 0
    for action, description in test_cases:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Action {action} ({description}): Reward={reward:.2f}, Position={info['position']}")
        
        if terminated or truncated:
            break
    
    print(f"Total reward: {total_reward:.2f}")
    env.render()

def test_reward_system():
    """Test the reward system in detail"""
    print("\n\nTesting Reward System")
    print("=" * 50)
    
    env = ImprovedF1Env()
    obs, info = env.reset()
    
    print("Starting position:", info['position'])
    print("Track layout:", env.get_track_layout()[:5], "...")  # Show first 5 positions
    
    # Test different scenarios
    scenarios = [
        (2, "Move to next track position"),
        (1, "Move off track"),
        (0, "Move back toward track"),
        (2, "Continue on track"),
    ]
    
    for action, description in scenarios:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"{description}: Action={action}, Reward={reward:.2f}, "
              f"Position={info['position']}, On Track={info['on_track']}")
        
        if terminated or truncated:
            break

def run_all_tests():
    """Run all test functions"""
    print("F1 Environment Test Suite")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_optimal_path()
        test_random_agent()
        test_edge_cases()
        test_reward_system()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\n Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()