# Formula 1 Path Optimization RL Simulator

A 6x6 grid-based Formula 1 track simulation where an AI agent learns to navigate from start (0,0) to finish (5,5) while staying on the racing line.

## Features
- Custom F1-themed racing environment
- Multiple RL algorithms (DQN, REINFORCE, PPO, A2C)
- Real-time visualization with Pygame
- Performance analysis and comparison

## Setup
```bash
pip install -r requirements.txt
```

## Usage
- Random agent demo: `python main.py`
- Train DQN: `python training/dqn_training.py`
- Train policy gradient methods: `python training/pg_training.py`

## Environment Details
- **Grid Size**: 6x6
- **Start Position**: (0,0) 
- **Goal Position**: (5,5)
- **Actions**: Up, Right, Down, Left
- **Rewards**: +10 for goal, -0.1 per step, -1 for off-track

## Models Trained
1. **DQN** - Deep Q-Network with experience replay
2. **REINFORCE** - Policy gradient with Monte Carlo returns  
3. **PPO** - Proximal Policy Optimization
4. **A2C** - Advantage Actor-Critic
