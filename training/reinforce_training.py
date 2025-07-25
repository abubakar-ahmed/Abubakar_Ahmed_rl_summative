import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import json
from collections import deque

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.f1_env import Formula1PathEnv


class PolicyNetwork(nn.Module):
    """
    Neural network for the REINFORCE policy.
    
    Takes state as input and outputs action probabilities.
    """
    
    def __init__(self, state_size=2, action_size=4, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state)
    
    def get_action(self, state):
        """
        Get action from the policy.
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
            log_prob: Log probability of the action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state_tensor)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob


class REINFORCEAgent:
    """
    REINFORCE agent for Formula 1 racing.
    
    Implements the classic policy gradient algorithm with Monte Carlo returns.
    """
    
    def __init__(self, state_size=2, action_size=4, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Neural network
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Training data storage
        self.reset_episode()
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rates = deque(maxlen=100)
        
    def reset_episode(self):
        """Reset episode data storage."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
    
    def get_action(self, state):
        """Get action from the current policy."""
        action, log_prob = self.policy.get_action(state)
        
        # Store for training
        self.states.append(state)
        self.log_probs.append(log_prob)
        
        return action
    
    def store_transition(self, action, reward):
        """Store action and reward for training."""
        self.actions.append(action)
        self.rewards.append(reward)
    
    def compute_returns(self):
        """Compute discounted returns for the episode."""
        returns = []
        G = 0
        
        # Compute returns backwards
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def update_policy(self):
        """Update the policy using REINFORCE."""
        if not self.rewards:
            return 0.0
        
        # Compute returns
        returns = self.compute_returns()
        returns = torch.FloatTensor(returns)
        
        # Normalize returns (helps with training stability)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return policy_loss.item()
    
    def train_episode(self, env):
        """Train on a single episode."""
        self.reset_episode()
        
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 100:  # Max episode length
            # Get action from policy
            action = self.get_action(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store transition
            self.store_transition(action, reward)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if terminated or truncated:
                break
        
        # Update policy
        loss = self.update_policy()
        
        # Track performance
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.success_rates.append(1.0 if terminated else 0.0)
        
        return total_reward, steps, terminated, loss
    
    def evaluate(self, env, n_episodes=10):
        """Evaluate the current policy."""
        total_rewards = []
        success_count = 0
        episode_lengths = []
        
        # Temporary disable training mode
        self.policy.eval()
        
        with torch.no_grad():
            for _ in range(n_episodes):
                state, _ = env.reset()
                total_reward = 0
                steps = 0
                
                while steps < 100:
                    # Get action (deterministic)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs = self.policy(state_tensor)
                    action = torch.argmax(action_probs, dim=1).item()
                    
                    state, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    if terminated:
                        success_count += 1
                        break
                    elif truncated:
                        break
                
                total_rewards.append(total_reward)
                episode_lengths.append(steps)
        
        # Re-enable training mode
        self.policy.train()
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'success_rate': success_count / n_episodes,
            'mean_length': np.mean(episode_lengths)
        }
    
    def save_model(self, filepath):
        """Save the trained policy."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma
            }
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained policy."""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train_reinforce_agent(episodes=2000, learning_rate=0.001, gamma=0.99, 
                         eval_freq=200, save_path="models/pg/"):
    """
    Train a REINFORCE agent on the F1 environment.
    
    Args:
        episodes: Number of training episodes
        learning_rate: Learning rate for policy updates
        gamma: Discount factor
        eval_freq: Frequency of evaluation
        save_path: Directory to save the trained model
        
    Returns:
        Trained REINFORCE agent and training metrics
    """
    print("Training REINFORCE Agent for Formula 1 Path Optimization")
    print("=" * 65)
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Hyperparameters:")
    print(f"   Episodes: {episodes}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Gamma (Discount): {gamma}")
    print(f"   Evaluation Frequency: {eval_freq}")
    
    # Create environments
    train_env = Formula1PathEnv()
    eval_env = Formula1PathEnv()
    
    # Create agent
    agent = REINFORCEAgent(
        state_size=2,
        action_size=4,
        learning_rate=learning_rate,
        gamma=gamma
    )
    
    print(f"\nPolicy Network:")
    print(agent.policy)
    
    # Training metrics
    training_rewards = []
    training_lengths = []
    training_success_rates = []
    training_losses = []
    evaluation_scores = []
    evaluation_episodes = []
    
    print(f"\nStarting Training...")
    start_time = datetime.now()
    
    try:
        for episode in range(episodes):
            # Train one episode
            reward, length, success, loss = agent.train_episode(train_env)
            
            # Store metrics
            training_rewards.append(reward)
            training_lengths.append(length)
            training_success_rates.append(1.0 if success else 0.0)
            training_losses.append(loss)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                recent_reward = np.mean(agent.episode_rewards)
                recent_success = np.mean(agent.success_rates)
                recent_length = np.mean(agent.episode_lengths)
                
                print(f"Episode {episode + 1:4d}: "
                      f"Reward={recent_reward:6.2f}, "
                      f"Success={recent_success:5.1%}, "
                      f"Length={recent_length:5.1f}, "
                      f"Loss={loss:8.3f}")
            
            # Periodic evaluation
            if (episode + 1) % eval_freq == 0:
                eval_results = agent.evaluate(eval_env, n_episodes=20)
                evaluation_scores.append(eval_results)
                evaluation_episodes.append(episode + 1)
                
                print(f"Evaluation at episode {episode + 1}:")
                print(f"   Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
                print(f"   Success Rate: {eval_results['success_rate']:.1%}")
                print(f"   Mean Length: {eval_results['mean_length']:.1f}")
        
        training_time = datetime.now() - start_time
        print(f"\nTraining completed in {training_time}")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        training_time = datetime.now() - start_time
        print(f"   Trained for {training_time}")
    
    # Save the trained model
    model_path = os.path.join(save_path, "reinforce_f1_model.pth")
    agent.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    # Final evaluation
    print(f"\nFinal Evaluation:")
    final_eval = agent.evaluate(eval_env, n_episodes=50)
    
    print(f"   Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"   Success Rate: {final_eval['success_rate']:.1%}")
    print(f"   Mean Episode Length: {final_eval['mean_length']:.1f}")
    
    # Save training metrics
    training_data = {
        'episodes': list(range(1, len(training_rewards) + 1)),
        'training_rewards': training_rewards,
        'training_lengths': training_lengths,
        'training_success_rates': training_success_rates,
        'training_losses': training_losses,
        'evaluation_episodes': evaluation_episodes,
        'evaluation_scores': evaluation_scores,
        'final_evaluation': final_eval,
        'hyperparameters': {
            'learning_rate': learning_rate,
            'gamma': gamma,
            'episodes': episodes
        }
    }
    
    metrics_path = os.path.join(save_path, "reinforce_training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_data, f, indent=2, default=str)
    print(f"Training metrics saved to: {metrics_path}")
    
    # Create performance plots
    create_reinforce_plots(training_data, save_path)
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    return agent, training_data


def create_reinforce_plots(training_data, save_path):
    """Create and save REINFORCE training performance plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('REINFORCE Training Performance - Formula 1 Path Optimization', fontsize=16)
    
    episodes = training_data['episodes']
    
    # Plot 1: Training Rewards (smoothed)
    window = 50
    if len(training_data['training_rewards']) >= window:
        smoothed_rewards = np.convolve(training_data['training_rewards'], 
                                     np.ones(window)/window, mode='valid')
        smoothed_episodes = episodes[window-1:]
        ax1.plot(smoothed_episodes, smoothed_rewards, 'b-', linewidth=2, label=f'Smoothed ({window} episodes)')
    
    ax1.plot(episodes, training_data['training_rewards'], 'lightblue', alpha=0.3, label='Raw')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Learning Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Success Rate (smoothed)
    if len(training_data['training_success_rates']) >= window:
        smoothed_success = np.convolve(training_data['training_success_rates'], 
                                     np.ones(window)/window, mode='valid')
        ax2.plot(smoothed_episodes, [s * 100 for s in smoothed_success], 'g-', linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate Improvement')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # Plot 3: Episode Length (smoothed)
    if len(training_data['training_lengths']) >= window:
        smoothed_lengths = np.convolve(training_data['training_lengths'], 
                                     np.ones(window)/window, mode='valid')
        ax3.plot(smoothed_episodes, smoothed_lengths, 'r-', linewidth=2)
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Episode Length')
    ax3.set_title('Episode Efficiency')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Policy Loss
    if training_data['training_losses']:
        if len(training_data['training_losses']) >= window:
            smoothed_loss = np.convolve(training_data['training_losses'], 
                                      np.ones(window)/window, mode='valid')
            ax4.plot(smoothed_episodes, smoothed_loss, 'purple', linewidth=2)
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Policy Loss')
        ax4.set_title('Training Loss')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.axis('off')
        ax4.text(0.5, 0.5, 'No loss data available', ha='center', va='center')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_path, "reinforce_training_performance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training plots saved to: {plot_path}")
    
    plt.show()


def hyperparameter_tuning():
    """
    Run hyperparameter tuning for REINFORCE.
    
    Tests different learning rates and discount factors.
    """
    print("Starting REINFORCE Hyperparameter Tuning")
    print("=" * 50)
    
    # Hyperparameter combinations to test
    param_combinations = [
        {'learning_rate': 0.001, 'gamma': 0.99, 'name': 'Standard'},
        {'learning_rate': 0.0005, 'gamma': 0.95, 'name': 'Conservative'},
        {'learning_rate': 0.002, 'gamma': 0.99, 'name': 'Aggressive'},
    ]
    
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\nExperiment {i+1}/3: {params['name']}")
        print(f"   Learning Rate: {params['learning_rate']}")
        print(f"   Gamma: {params['gamma']}")
        
        # Train model with these parameters
        agent, training_data = train_reinforce_agent(
            episodes=1500,  # Shorter for tuning
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            eval_freq=300,
            save_path=f"models/pg/reinforce_tune_{params['name'].lower()}/"
        )
        
        # Store results
        final_eval = training_data['final_evaluation']
        
        results.append({
            'name': params['name'],
            'params': params,
            'final_reward': final_eval['mean_reward'],
            'final_success': final_eval['success_rate']
        })
        
        print(f"   Result: {final_eval['mean_reward']:.2f} reward, {final_eval['success_rate']:.1%} success")
    
    # Print tuning summary
    print(f"\nREINFORCE Hyperparameter Tuning Results:")
    print("=" * 55)
    
    best_result = max(results, key=lambda x: x['final_reward'])
    
    for result in results:
        marker = "Best" if result == best_result else "   "
        print(f"{marker} {result['name']:>12}: "
              f"{result['final_reward']:6.2f} reward, "
              f"{result['final_success']:5.1%} success")
    
    print(f"\nBest Configuration: {best_result['name']}")
    return best_result


def test_trained_model(model_path="models/pg/reinforce_f1_model.pth", n_episodes=10):
    """Test a trained REINFORCE model."""
    print(f"Testing Trained REINFORCE Model")
    print("=" * 40)
    
    # Create environment and agent
    env = Formula1PathEnv()
    agent = REINFORCEAgent()
    
    try:
        # Load trained model
        agent.load_model(model_path)
        print(f"Model loaded from: {model_path}")
        
        # Evaluate model
        results = agent.evaluate(env, n_episodes=n_episodes)
        
        print(f"\nTest Results ({n_episodes} episodes):")
        print(f"   Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"   Success Rate: {results['success_rate']:.1%}")
        print(f"   Mean Episode Length: {results['mean_length']:.1f}")
        
        # Show some sample episodes
        print(f"\nSample Episodes:")
        agent.policy.eval()
        
        with torch.no_grad():
            for episode in range(min(3, n_episodes)):
                print(f"\n  Episode {episode + 1}:")
                state, _ = env.reset()
                total_reward = 0
                steps = 0
                path = [tuple(state)]
                
                while steps < 100:
                    # Get action (deterministic)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs = agent.policy(state_tensor)
                    action = torch.argmax(action_probs, dim=1).item()
                    
                    state, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    path.append(tuple(state))
                    
                    if terminated:
                        print(f"    SUCCESS in {steps} steps, reward: {total_reward:.2f}")
                        break
                    elif truncated:
                        print(f"    TIMEOUT after {steps} steps, reward: {total_reward:.2f}")
                        break
                
                # Show path taken
                print(f"    Path: {' → '.join([f'({p[1]},{p[0]})' for p in path[:min(8, len(path))]])}{'...' if len(path) > 8 else ''}")
        
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print(f"   Please train a model first using train_reinforce_agent()")
    except Exception as e:
        print(f"Error testing model: {e}")
    finally:
        env.close()


def main():
    """Main training function for REINFORCE."""
    print("Formula 1 REINFORCE Training Suite")
    print("Choose training mode:")
    print("1. Standard Training (recommended)")
    print("2. Hyperparameter Tuning")
    print("3. Quick Training (testing)")
    print("4. Test Existing Model")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            print("\nStarting Standard REINFORCE Training...")
            agent, training_data = train_reinforce_agent(
                episodes=2000,
                learning_rate=0.001,
                gamma=0.99,
                eval_freq=200,
                save_path="models/pg/"
            )
            
        elif choice == "2":
            print("\nStarting Hyperparameter Tuning...")
            best_config = hyperparameter_tuning()
            
            print(f"\nTraining final model with best configuration...")
            agent, training_data = train_reinforce_agent(
                episodes=3000,  # Longer training with best params
                learning_rate=best_config['params']['learning_rate'],
                gamma=best_config['params']['gamma'],
                eval_freq=250,
                save_path="models/pg/reinforce_best/"
            )
            
        elif choice == "3":
            print("\nStarting Quick Training...")
            agent, training_data = train_reinforce_agent(
                episodes=500,
                learning_rate=0.002,
                gamma=0.99,
                eval_freq=100,
                save_path="models/pg/reinforce_quick/"
            )
            
        elif choice == "4":
            print("\nTesting Existing Model...")
            model_path = input("Enter model path (or press Enter for default): ").strip()
            if not model_path:
                model_path = "models/pg/reinforce_f1_model.pth"
            test_trained_model(model_path, n_episodes=20)
            return
            
        else:
            print("Invalid choice. Running standard training...")
            agent, training_data = train_reinforce_agent()
        
        print(f"\nREINFORCE Training Complete!")
        print(f"Next steps:")
        print(f"1. Run 'python training/pg_training.py' for PPO and A2C")
        print(f"2. Compare all models with analysis script")
        print(f"3. Create final performance video")
        
    except KeyboardInterrupt:
        print(f"\nTraining cancelled by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()