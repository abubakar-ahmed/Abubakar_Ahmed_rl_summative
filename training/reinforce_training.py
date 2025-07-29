import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random

# Environment import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.f1_env import ImprovedF1Env

class PolicyNetwork(nn.Module):
    """Improved policy network with better architecture"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class REINFORCEAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=100, factor=0.5)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        
    def get_action(self, state, epsilon=0.1):
        state = torch.FloatTensor(state).to(self.device)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            action = torch.randint(0, probs.size(-1), (1,)).item()
        else:
            action = dist.sample().item()
            
        self.log_probs.append(dist.log_prob(torch.tensor(action).to(self.device)))
        return action

    def update_policy(self):
        if not self.rewards:
            return 0.0

        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Update network
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        # Clear buffers
        self.log_probs = []
        self.rewards = []
        
        return loss.item()

    def train_episode(self, env):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 200:  # Increased max steps
            action = self.get_action(state, epsilon=max(0.01, 0.1 * (1 - len(self.episode_rewards)/2000)))
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Reward shaping
            shaped_reward = reward
            if terminated:
                shaped_reward += 10.0  # Bonus for success
            elif truncated:
                shaped_reward -= 1.0   # Penalty for timeout
            
            self.rewards.append(shaped_reward)
            episode_reward += reward
            state = next_state
            steps += 1
            done = terminated or truncated
        
        # Update policy
        loss = self.update_policy()
        
        # Track metrics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(steps)
        self.success_rates.append(1 if terminated else 0)
        
        return episode_reward, steps, terminated, loss

    def evaluate(self, env, n_episodes=10):
        self.policy.eval()
        total_rewards = []
        successes = 0
        
        with torch.no_grad():
            for _ in range(n_episodes):
                state, _ = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    action_probs = self.policy(state_tensor)
                    action = torch.argmax(action_probs).item()
                    state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                
                total_rewards.append(episode_reward)
                if terminated:
                    successes += 1
        
        self.policy.train()
        return {
            'mean_reward': np.mean(total_rewards),
            'success_rate': successes / n_episodes
        }

    def save_model(self, path):
        torch.save({
            'policy_state': self.policy.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

def train_reinforce(episodes=2000):
    env = ImprovedF1Env()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = REINFORCEAgent(state_size, action_size)
    
    best_success = 0
    rewards_history = []
    
    for episode in range(1, episodes+1):
        reward, steps, success, loss = agent.train_episode(env)
        rewards_history.append(reward)
        
        # Logging
        if episode % 100 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            success_rate = np.mean(agent.success_rates[-100:])
            print(f"Episode {episode}: Reward={avg_reward:.2f}, Success={success_rate:.1%}")
            
            # Evaluation
            eval_results = agent.evaluate(env)
            if eval_results['success_rate'] > best_success:
                best_success = eval_results['success_rate']
                agent.save_model("models/reinforce/reinforce.zip")
                print(f"New best model! Success rate: {best_success:.1%}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title("Training Rewards")
    
    plt.subplot(1, 2, 2)
    window = 50
    smooth = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    plt.plot(smooth)
    plt.title(f"Smoothed Rewards (window={window})")
    
    plt.tight_layout()
    plt.show()
    
    return agent

if __name__ == "__main__":
    agent = train_reinforce()