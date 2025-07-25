import pygame
import numpy as np
from stable_baselines3 import DQN
from environment.f1_env import Formula1PathEnv
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class F1Visualizer:
    """Enhanced visualizer with real-time rendering"""
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.cell_size = 100
        self.width = env.grid_size * self.cell_size
        self.height = env.grid_size * self.cell_size
        self.screen = pygame.display.set_mode((self.width, self.height + 80))
        pygame.display.set_caption("F1 RL Agent")
        
        # Colors
        self.colors = {
            'track': (50, 50, 50),
            'off_track': (34, 139, 34),
            'car': (255, 0, 0),
            'hud': (0, 0, 0)
        }
        
    def render(self, state, reward, steps, done):
        """Render current state"""
        self.screen.fill(self.colors['off_track'])
        
        # Draw track
        for x in range(self.env.grid_size):
            for y in range(self.env.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                  self.cell_size, self.cell_size)
                color = self.colors['track'] if (x,y) in self.env.track_set else self.colors['off_track']
                pygame.draw.rect(self.screen, color, rect)
        
        # Draw car
        car_x = state[1] * self.cell_size + self.cell_size//2
        car_y = state[0] * self.cell_size + self.cell_size//2
        pygame.draw.circle(self.screen, self.colors['car'], (car_x, car_y), 15)
        
        # Draw HUD
        hud_rect = pygame.Rect(0, self.height, self.width, 80)
        pygame.draw.rect(self.screen, self.colors['hud'], hud_rect)
        font = pygame.font.SysFont('Arial', 20)
        texts = [
            f"Steps: {steps}",
            f"Reward: {reward:.2f}",
            f"Status: {'Racing' if not done else 'Finished'}"
        ]
        
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, (255,255,255))
            self.screen.blit(text_surface, (10, self.height + 10 + i*25))
        
        pygame.display.flip()

def train_f1_agent():
    """Enhanced training with visualization"""
    env = Formula1PathEnv()
    visualizer = F1Visualizer(env)
    
    # Improved hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        buffer_size=100000,
        batch_size=128,
        gamma=0.99,
        exploration_final_eps=0.02,
        exploration_fraction=0.3,
        target_update_interval=1000,
        verbose=1
    )
    
    # Train for longer
    model.learn(total_timesteps=200000)
    model.save("f1_racing_agent")
    
    return model

def test_agent(model_path="f1_racing_agent"):
    """Interactive testing with visualization"""
    env = Formula1PathEnv()
    visualizer = F1Visualizer(env)
    model = DQN.load(model_path)
    
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(int(action))
            
            total_reward += reward
            steps += 1
            
            visualizer.render(obs, total_reward, steps, done)
            pygame.time.delay(100)  # Control speed
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        
        print(f"Episode {episode+1}: Reward={total_reward:.1f}, Steps={steps}")

if __name__ == "__main__":
    # Train and test
    trained_model = train_f1_agent()
    test_agent()