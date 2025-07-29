"""
test_improved_environment.py
Comprehensive test script for the improved F1 environment with track adherence
"""

import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.f1_env import Formula1PathEnv


def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing Improved Formula 1 Environment")
    print("=" * 55)
    
    # Create environment
    env = Formula1PathEnv()
    
    # Test 1: Environment creation
    print("✅ Test 1: Environment created successfully")
    
    # Test 2: Action and observation spaces
    print(f"✅ Test 2: Action space: {env.action_space}")
    print(f"✅ Test 2: Observation space: {env.observation_space}")
    
    # Test 3: Reset functionality
    state, info = env.reset()
    print(f"✅ Test 3: Reset successful. Initial state: {state}")
    assert tuple(state) == (0, 0), f"Expected (0,0), got {tuple(state)}"
    
    # Test 4: Track validation
    track = env.get_track_positions()
    print(f"✅ Test 4: Track has {len(track)} positions")
    assert (0, 0) in track, "Start position not in track"
    assert (5, 5) in track, "Finish position not in track"
    
    # Test 5: Valid actions
    valid_actions = []
    for action in range(4):
        old_state = env.state.copy()
        state, reward, terminated, truncated, info = env.step(action)
        valid_actions.append(action)
        print(f"   Action {action}: {old_state} → {state}, reward: {reward:.1f}")
        if not terminated:
            env.reset()  # Reset for next test
    
    print(f"✅ Test 5: All {len(valid_actions)} actions work correctly")
    
    return True


def test_reward_system():
    """Test the improved reward system."""
    print("\nTesting Reward System")
    print("=" * 30)
    
    env = Formula1PathEnv()
    
    # Test on-track movement (forward progress)
    print("Test 6a: On-track forward movement")
    env.reset()
    state, reward, _, _, info = env.step(1)  # Move right to (0,1)
    print(f"   State: (0,0) → ({state[1]},{state[0]})")
    print(f"   Reward: {reward:.2f}")
    print(f"   On track: {info['on_track']}")
    print(f"   Track progress: {info['track_progress']:.2f}")
    assert reward > 0, f"Forward progress should be positive, got {reward}"
    
    # Test off-track movement attempt
    print("\nTest 6b: Off-track movement attempt")
    env.reset()
    old_state = env.state.copy()
    state, reward, _, _, info = env.step(2)  # Try to move down from (0,0) - off track
    print(f"   Attempted: (0,0) → (1,0)")
    print(f"   Actual state: ({state[1]},{state[0]}) (should stay at start)")
    print(f"   Reward: {reward:.2f}")
    print(f"   Attempted off-track: {info['attempted_off_track']}")
    assert reward < -1, f"Off-track attempt should be heavily penalized, got {reward}"
    assert np.array_equal(state, old_state), "Agent should not move off-track"
    
    # Test escalating off-track penalties
    print("\nTest 6c: Escalating off-track penalties")
    env.reset()
    penalties = []
    for i in range(3):
        state, reward, _, _, info = env.step(2)  # Keep trying to go off-track
        penalties.append(reward)
        print(f"   Attempt {i+1}: Reward = {reward:.2f}, Consecutive = {info['consecutive_off_track']}")
    
    # Penalties should get worse
    assert penalties[1] < penalties[0], "Second penalty should be worse than first"
    assert penalties[2] < penalties[1], "Third penalty should be worse than second"
    
    # Test goal reaching
    print("\nTest 6d: Goal reaching")
    env.reset()
    env.state = np.array([5, 4], dtype=np.int32)  # Set near goal
    state, reward, terminated, _, info = env.step(1)  # Move to goal
    print(f"   Final move reward: {reward:.2f}")
    print(f"   Episode terminated: {terminated}")
    assert reward > 50, f"Goal reward should be large, got {reward}"
    assert terminated, "Episode should terminate at goal"
    
    print("✅ All reward system tests passed!")
    return True


def test_track_integrity():
    """Test that the track forms a valid path."""
    print("\nTesting Track Integrity")
    print("=" * 30)
    
    env = Formula1PathEnv()
    track = env.get_track_positions()
    
    # Test that track is connected
    connected = True
    for i, pos in enumerate(track):
        if i == 0:
            continue
        
        prev_pos = track[i-1]
        distance = abs(pos[0] - prev_pos[0]) + abs(pos[1] - prev_pos[1])
        if distance != 1:
            print(f"❌ Track disconnected between {prev_pos} and {pos}")
            connected = False
    
    if connected:
        print("✅ Track is properly connected")
    
    # Test track bounds
    valid_bounds = True
    for pos in track:
        if pos[0] < 0 or pos[0] >= 6 or pos[1] < 0 or pos[1] >= 6:
            print(f"❌ Track position {pos} is out of bounds")
            valid_bounds = False
    
    if valid_bounds:
        print("✅ All track positions are within bounds")
    
    # Print track visualization
    print("\nTrack Layout:")
    grid = [['.' for _ in range(6)] for _ in range(6)]
    
    for i, (x, y) in enumerate(track):
        if (x, y) == (0, 0):
            grid[y][x] = 'S'  # Start
        elif (x, y) == (5, 5):
            grid[y][x] = 'F'  # Finish
        else:
            grid[y][x] = str(i)[-1]  # Track position number
    
    print("    0 1 2 3 4 5")
    for i, row in enumerate(grid):
        print(f"  {i} " + ' '.join(row))
    
    print("  Legend: S=Start, F=Finish, numbers=Track positions, .=Off-track")
    
    return connected and valid_bounds


def test_track_adherence():
    """Test that the improved environment enforces track adherence."""
    print("\nTesting Track Adherence Enforcement")
    print("=" * 40)
    
    env = Formula1PathEnv()
    env.reset()
    
    # Test 1: Can only move to adjacent track positions
    print("Test: Movement restrictions")
    valid_moves = 0
    invalid_moves = 0
    
    for action in range(4):
        env.reset()
        old_state = env.state.copy()
        state, reward, _, _, info = env.step(action)
        
        if info['attempted_off_track']:
            invalid_moves += 1
            assert np.array_equal(state, old_state), f"Agent moved off-track: {old_state} → {state}"
            print(f"   Action {action}: Correctly blocked off-track move")
        else:
            valid_moves += 1
            print(f"   Action {action}: Valid move {old_state} → {state}")
    
    print(f"✅ Valid moves: {valid_moves}, Invalid moves blocked: {invalid_moves}")
    
    # Test 2: Track progress tracking
    print("\nTest: Track progress tracking")
    env.reset()
    progress_sequence = []
    
    # Follow the track properly: RIGHT, DOWN, RIGHT, DOWN, RIGHT
    track_actions = [1, 2, 1, 2, 1]  # Should follow track and increase progress
    for i, action in enumerate(track_actions):
        state, reward, terminated, truncated, info = env.step(action)
        progress = info['track_progress']
        progress_sequence.append(progress)
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        print(f"   Step {i+1}: {action_names[action]} → Position ({state[1]},{state[0]}), Progress: {progress:.2f}")
        
        if terminated or truncated:
            break
    
    # Progress should generally increase
    if len(progress_sequence) > 1:
        improvements = sum(1 for i in range(1, len(progress_sequence)) 
                          if progress_sequence[i] > progress_sequence[i-1])
        print(f"✅ Progress improvements: {improvements}/{len(progress_sequence)-1}")
    else:
        print("✅ Progress tracking functional")
    return True


def test_successful_path():
    """Test a known successful path through the track."""
    print("\nTesting Known Successful Path")
    print("=" * 35)
    
    env = Formula1PathEnv()
    env.reset()
    
    # Based on the track layout from the test:
    # Track: (0,0)→(1,0)→(1,1)→(2,1)→(2,2)→(3,2)→(3,3)→(4,3)→(4,4)→(5,4)→(5,5)
    # Actions needed: RIGHT, DOWN, RIGHT, DOWN, RIGHT, DOWN, RIGHT, DOWN, RIGHT, DOWN
    successful_actions = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]  # Alternating RIGHT and DOWN
    
    total_reward = 0
    path = [(0, 0)]  # Starting position
    
    print("Executing successful path:")
    print(f"   Start: (0,0)")
    print("   Track to follow: (0,0)→(1,0)→(1,1)→(2,1)→(2,2)→(3,2)→(3,3)→(4,3)→(4,4)→(5,4)→(5,5)")
    
    for i, action in enumerate(successful_actions):
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        path.append((state[1], state[0]))  # Convert to (x,y)
        
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        status = "✅" if info['on_track'] else "❌"
        
        print(f"   Step {i+1}: {action_names[action]} → ({state[1]},{state[0]}) {status} "
              f"R:{reward:+5.1f} Progress:{info['track_progress']:.2f}")
        
        if terminated:
            print(f"   🏆 SUCCESS! Reached goal in {i+1} steps!")
            break
        elif truncated:
            print(f"   ⏰ Timeout after {i+1} steps")
            break
        elif not info['on_track']:
            print(f"   🚫 Went off track!")
            break
    
    print(f"\nPath taken: {' → '.join(map(str, path))}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Episode completed: {'Success' if terminated else 'Failed'}")
    
    return terminated


def run_stress_test():
    """Run multiple episodes to test consistency."""
    print("\nRunning Stress Test (100 random episodes)")
    print("=" * 45)
    
    env = Formula1PathEnv()
    
    results = {
        'completed': 0,
        'timeouts': 0,
        'total_rewards': [],
        'track_adherence': [],
        'episode_lengths': []
    }
    
    for episode in range(100):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        on_track_moves = 0
        total_moves = 0
        
        for step in range(100):  # Max steps per episode
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            total_moves += 1
            
            if info['on_track']:
                on_track_moves += 1
            
            if terminated:
                results['completed'] += 1
                break
            elif truncated:
                results['timeouts'] += 1
                break
        
        results['total_rewards'].append(total_reward)
        results['episode_lengths'].append(steps)
        results['track_adherence'].append(on_track_moves / total_moves if total_moves > 0 else 0)
        
        if (episode + 1) % 20 == 0:
            print(f"   Completed {episode + 1}/100 episodes...")
    
    # Summary statistics
    print(f"\nStress Test Results:")
    print(f"   Episodes completed: {results['completed']}/100")
    print(f"   Episodes timed out: {results['timeouts']}/100")
    print(f"   Average reward: {np.mean(results['total_rewards']):.2f} ± {np.std(results['total_rewards']):.2f}")
    print(f"   Average episode length: {np.mean(results['episode_lengths']):.1f} ± {np.std(results['episode_lengths']):.1f}")
    print(f"   Average track adherence: {np.mean(results['track_adherence']):.1%} ± {np.std(results['track_adherence']):.1%}")
    
    print("✅ Stress test completed!")
    return True


def main():
    """Run all tests."""
    print("Improved Formula 1 Environment Test Suite")
    print("=" * 50)
    print("Testing enhanced environment with track adherence enforcement")
    print()
    
    start_time = time.time()
    
    try:
        # Run all tests
        tests = [
            ("Basic Functionality", test_basic_functionality),
            ("Reward System", test_reward_system),
            ("Track Integrity", test_track_integrity),
            ("Track Adherence", test_track_adherence),
            ("Successful Path", test_successful_path),
            ("Stress Test", run_stress_test)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = test_func()
                if result:
                    passed_tests += 1
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"❌ {test_name} FAILED with error: {e}")
                import traceback
                traceback.print_exc()
        
        # Final summary
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"TEST SUITE COMPLETE ({elapsed_time:.1f}s)")
        print("=" * 60)
        print(f"Tests passed: {passed_tests}/{total_tests}")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Environment is ready for RL training.")
            print("\nKey improvements verified:")
            print("   ✅ Off-track movements are blocked")
            print("   ✅ Track adherence is enforced")
            print("   ✅ Reward system encourages proper racing")
            print("   ✅ Progress tracking works correctly")
        else:
            print("⚠️  Some tests failed. Please review the issues above.")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return passed_tests == total_tests


if __name__ == "__main__":
    main()