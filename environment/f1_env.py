import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Formula1PathEnv(gym.Env):
    """
    Custom Formula 1 racing environment for reinforcement learning.
    
    The agent (F1 car) must navigate from start to finish while staying
    on the racing line to minimize lap time (steps taken).
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Environment parameters
        self.grid_size = 6
        self.render_mode = render_mode
        
        # Define action and observation spaces
        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        self.action_space = spaces.Discrete(4)
        
        # State: Agent's (x, y) position on the grid
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32
        )
        
        # Define the F1 racing track (optimal racing line)
        # This represents the fastest path through the circuit
        self.track = [
            (0, 0),  # Start/Finish line
            (0, 1),  # Turn 1 entry
            (1, 1),  # Turn 1 apex
            (1, 2),  # Turn 1 exit
            (2, 2),  # Straight section
            (2, 3),  # Turn 2 entry  
            (3, 3),  # Turn 2 apex
            (3, 4),  # Turn 2 exit
            (4, 4),  # Final sector
            (4, 5),  # Turn 3 entry
            (5, 5)   # Finish line
        ]
        
        # Convert track to set for faster lookup
        self.track_set = set(self.track)
        
        # Episode parameters
        self.max_steps = 100  # Maximum steps before timeout
        self.current_step = 0
        self.state = None
        
        # Action mappings for readability
        self.action_to_direction = {
            0: (-1, 0),  # Up (decrease y)
            1: (0, 1),   # Right (increase x) 
            2: (1, 0),   # Down (increase y)
            3: (0, -1)   # Left (decrease x)
        }
        
        # Performance tracking
        self.episode_rewards = []
        self.successful_episodes = 0
        self.total_episodes = 0
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial state (0, 0)
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Reset agent to starting position
        self.state = np.array([0, 0], dtype=np.int32)
        self.current_step = 0
        
        # Track episode statistics
        self.total_episodes += 1
        
        return self.state.copy(), {}
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Integer representing the action to take
            
        Returns:
            observation: New state after action
            reward: Reward for this step
            terminated: Whether episode ended successfully  
            truncated: Whether episode ended due to time limit
            info: Additional information dictionary
        """
        self.current_step += 1
        
        # Calculate new position based on action
        dy, dx = self.action_to_direction[action]
        new_y = self.state[0] + dy
        new_x = self.state[1] + dx
        
        # Ensure new position is within grid bounds
        new_y = np.clip(new_y, 0, self.grid_size - 1)
        new_x = np.clip(new_x, 0, self.grid_size - 1)
        
        new_state = np.array([new_y, new_x], dtype=np.int32)
        
        # Calculate reward based on new position
        reward = self._calculate_reward(new_state)
        
        # Check termination conditions
        terminated = tuple(new_state) == (5, 5)  # Reached finish line
        truncated = self.current_step >= self.max_steps  # Timeout
        
        # Update state only if move is valid (on track)
        if tuple(new_state) in self.track_set:
            self.state = new_state
        # If off-track, agent stays in current position (penalty applied via reward)
        
        # Track successful episodes
        if terminated:
            self.successful_episodes += 1
        
        # Create info dictionary with useful metrics
        info = {
            'on_track': tuple(self.state) in self.track_set,
            'distance_to_goal': self._manhattan_distance_to_goal(),
            'steps_taken': self.current_step,
            'success_rate': self.successful_episodes / self.total_episodes if self.total_episodes > 0 else 0
        }
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def _calculate_reward(self, new_state):
        """
        Calculate reward for the given state transition.
        
        Reward structure designed to encourage:
        - Reaching the goal quickly (+10)
        - Staying on track (-0.1 per step vs -1 for off-track)
        - Efficient racing lines
        
        Args:
            new_state: The proposed new state
            
        Returns:
            float: Reward value
        """
        # Check if new position is the goal
        if tuple(new_state) == (5, 5):
            return 10.0  # Large reward for completing the lap
            
        # Penalty for going off the racing line
        elif tuple(new_state) not in self.track_set:
            return -1.0  # Significant penalty for off-track excursion
            
        else:
            # Base penalty for each step on track (encourages speed)
            reward = -0.1
            
            # Optional: Small bonus for making progress toward goal
            current_distance = self._manhattan_distance_to_goal()
            new_distance = abs(new_state[0] - 5) + abs(new_state[1] - 5)
            
            if new_distance < current_distance:
                reward += 0.05  # Small bonus for getting closer to goal
                
            return reward
    
    def _manhattan_distance_to_goal(self):
        """Calculate Manhattan distance from current position to goal."""
        return abs(self.state[0] - 5) + abs(self.state[1] - 5)
    
    def render(self):
        """
        Render the environment (placeholder for visualization).
        Will be implemented in the rendering module.
        """
        if self.render_mode == "human":
            print(f"Agent at position: {tuple(self.state)}, Step: {self.current_step}")
        
    def get_track_positions(self):
        """Return the track positions for visualization."""
        return self.track.copy()
    
    def is_on_track(self, position):
        """Check if a position is on the racing track."""
        return tuple(position) in self.track_set


# Test function to verify environment works correctly
def test_environment():
    """Test the environment with random actions."""
    print("🏁 Testing Formula 1 Environment...")
    
    env = Formula1PathEnv()
    
    # Test reset
    state, info = env.reset()
    print(f"Initial state: {state}")
    assert tuple(state) == (0, 0), "Reset should start at (0,0)"
    
    # Test a few random actions
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step+1}: Action={action}, State={state}, Reward={reward:.2f}, "
              f"On-track={info['on_track']}")
        
        if terminated or truncated:
            print(f"Episode ended! Total reward: {total_reward:.2f}")
            break
    
    print("✅ Environment test completed successfully!")


if __name__ == "__main__":
    test_environment()