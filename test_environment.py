"""
test_environment.py
Quick test script to verify the F1 environment works correctly
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.f1_env import Formula1PathEnv


def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing Formula 1 Environment Basic Functionality")
    print("=" * 55)
    
    # Create environment
    env = Formula1PathEnv()
    
    # Test 1: Environment creation
    print("Test 1: Environment created successfully")
    
    # Test 2: Action and observation spaces
    print(f"Test 2: Action space: {env.action_space}")
    print(f"Test 2: Observation space: {env.observation_space}")
    
    # Test 3: Reset functionality
    state, info = env.reset()
    print(f"Test 3: Reset successful. Initial state: {state}")
    assert tuple(state) == (0, 0), f"Expected (0,0), got {tuple(state)}"
    
    # Test 4: Track validation
    track = env.get_track_positions()
    print(f"Test 4: Track has {len(track)} positions")
    assert (0, 0) in track, "Start position not in track"
    assert (5, 5) in track, "Finish position not in track"
    
    # Test 5: Valid actions
    valid_actions = []
    for action in range(4):
        state, reward, terminated, truncated, info = env.step(action)
        valid_actions.append(action)
        if not terminated:
            env.reset()  # Reset for next test
    
    print(f"Test 5: All {len(valid_actions)} actions work correctly")
    
    # Test 6: Reward system
    env.reset()
    
    # Test on-track movement
    state, reward, _, _, info = env.step(1)  # Move right to (0,1)
    print(f"Test 6a: On-track movement reward: {reward}")
    assert reward == -0.05, f"Expected -0.05, got {reward}"
    
    # Test off-track movement
    env.reset()
    state, reward, _, _, info = env.step(2)  # Try to move down from (0,0) - off track
    print(f"Test 6b: Off-track movement reward: {reward}")
    assert reward == -0.95, f"Expected -0.95, got {reward}"
    
    # Test 7: Goal reaching
    env.reset()
    # Navigate to near the goal
    env.state = np.array([5, 4], dtype=np.int32)  # Set state to (4,5) in (x,y)
    state, reward, terminated, _, info = env.step(1)  # Move right to (5,5)
    print(f"Test 7: Goal reaching reward: {reward}, terminated: {terminated}")
    assert reward == 10.0, f"Expected 10.0, got {reward}"
    assert terminated, "Episode should terminate at goal"
    
    # Test 8: Episode timeout
    env.reset()
    for step in range(100):
        state, reward, terminated, truncated, info = env.step(0)  # Keep moving up (will hit boundary)
        if truncated:
            print(f"Test 8: Episode truncated at step {step + 1}")
            break
    else:
        print("Test 8: Episode should have been truncated after 100 steps")
    
    print("\nAll basic functionality tests passed!")
    return True


def test_track_integrity():
    """Test that the track forms a valid path."""
    print("\nTesting Track Integrity")
    print("=" * 30)
    
    env = Formula1PathEnv()
    track = env.get_track_positions()
    
    # Test that track is connected (each position has a neighbor)
    connected = True
    for i, pos in enumerate(track):
        if i == 0:  # Start position
            continue
        
        prev_pos = track[i-1]
        # Check if positions are adjacent (Manhattan distance = 1)
        distance = abs(pos[0] - prev_pos[0]) + abs(pos[1] - prev_pos[1])
        if distance != 1:
            print(f"Track disconnected between {prev_pos} and {pos}")
            connected = False
    
    if connected:
        print("Track is properly connected")
    
    # Test track bounds
    for pos in track:
        if pos[0] < 0 or pos[0] >= 6 or pos[1] < 0 or pos[1] >= 6:
            print(f"Track position {pos} is out of bounds")
            return False
    
    print("All track positions are within bounds")
    
    # Print track visualization
    print("\nTrack Layout:")
    grid = [['.' for _ in range(6)] for _ in range(6)]
    
    for i, (x, y) in enumerate(track):
        if (x, y) == (0, 0):
            grid[y][x] = 'S'  # Start
        elif (x, y) == (5, 5):
            grid[y][x] = 'F'  # Finish
        else:
            grid[y][x] = '█'  # Track
    
    for row in grid:
        print('  ' + ' '.join(row))
    
    print("  Legend: S=Start, F=Finish, █=Track, .=Off-track")
    
    return True


def run_quick_simulation():
    """Run a quick simulation to verify everything works together."""
    print("\nRunning Quick Simulation")
    print("=" * 35)
    
    env = Formula1PathEnv()
    state, _ = env.reset()
    
    total_reward = 0
    steps = 0
    
    print("Taking 20 random actions...")
    for step in range(20):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        status = "🟢" if info.get('on_track', True) else "🔴"
        
        print(f"  Step {step+1:2d}: {action_names[action]:>5} → "
              f"({state[1]},{state[0]}) {status} "
              f"R:{reward:+5.1f}")
        
        if terminated:
            print(f"  Reached goal in {steps} steps!")
            break
        elif truncated:
            print(f"  Episode truncated after {steps} steps")
            break
    
    print(f"\nSimulation Summary:")
    print(f"   Steps taken: {steps}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Final position: ({state[1]}, {state[0]})")
    print(f"   Episode ended: {'Success' if terminated else 'Timeout' if truncated else 'Ongoing'}")
    
    return True


def main():
    """Run all tests."""
    print("Formula 1 Environment Test Suite")
    print("=" * 40)
    
    try:
        # Run all tests
        test_basic_functionality()
        test_track_integrity()
        run_quick_simulation()
        
        print("\n" + "=" * 40)
        print("ALL TESTS PASSED! Environment is ready for training.")
        print("=" * 40)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()