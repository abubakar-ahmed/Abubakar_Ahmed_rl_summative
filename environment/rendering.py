import pygame
import numpy as np
from typing import Dict, Optional, List, Tuple
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.f1_env import ImprovedF1Env


class ImprovedF1Renderer:
    """
    Renderer for the improved 8x8 F1 racing environment.
    Features better visuals, progress tracking, and training metrics display.
    """
    
    def __init__(self, grid_size: int = 8, cell_size: int = 80):
        """Initialize the enhanced renderer"""
        pygame.init()
        
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.width = grid_size * cell_size
        self.height = grid_size * cell_size + 120  # More space for enhanced HUD
        
        # Create display surface
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Improved F1 Track Environment - DQN Training")
        
        # Fonts
        self.font_large = pygame.font.SysFont('arial', 20, bold=True)
        self.font_medium = pygame.font.SysFont('arial', 16)
        self.font_small = pygame.font.SysFont('arial', 12)
        
        # Define the improved track (matches ImprovedF1Env)
        self.track = [
            (1, 1),  # Start
            (2, 1),  # Down
            (3, 1),  # Down
            (4, 1),  # Down
            (5, 1),  # Turn preparation
            (5, 2),  # Right turn
            (5, 3),  # Right turn
            (5, 4),  # Right turn
            (4, 4),  # Up turn
            (3, 4),  # Up
            (2, 4),  # Up
            (1, 4),  # Up
            (1, 3),  # Left turn
            (1, 2),  # Finish line
        ]
        self.track_set = set(self.track)
        self.track_positions = {pos: idx for idx, pos in enumerate(self.track)}
        
        # Enhanced colors
        self.colors = {
            'track': (45, 45, 45),           # Dark asphalt
            'track_border': (255, 255, 255), # White track borders
            'off_track': (34, 139, 34),      # Forest green grass
            'car': (220, 20, 60),            # Crimson car
            'car_shadow': (139, 0, 0),       # Dark red shadow
            'start': (0, 255, 0),            # Bright green start
            'finish': (255, 215, 0),         # Gold finish
            'checkpoint': (70, 130, 180),    # Steel blue checkpoints
            'next_target': (255, 165, 0),    # Orange next target
            'progress_bar': (0, 191, 255),   # Deep sky blue
            'progress_bg': (169, 169, 169),  # Dark gray
            'hud_bg': (25, 25, 25),          # Very dark HUD
            'hud_text': (255, 255, 255),     # White text
            'grid': (100, 100, 100),         # Gray grid lines
            'reward_positive': (50, 205, 50), # Lime green for positive rewards
            'reward_negative': (255, 69, 0),  # Red orange for negative rewards
        }
        
        # Animation tracking
        self.car_rotation = 0
        self.reward_display_timer = 0
        self.last_reward = 0
        self.reward_flash_color = self.colors['hud_text']
        
        # Performance tracking
        self.frame_count = 0
        self.total_episodes = 0

    def render(self, state: np.ndarray, step: int, reward: float, 
               total_reward: float = 0, info: Optional[Dict] = None,
               training_info: Optional[Dict] = None) -> np.ndarray:
        """Render the current environment state with enhanced visuals"""
        
        # Handle reward display timing
        if reward != 0:
            self.last_reward = reward
            self.reward_display_timer = 30  # Display for 30 frames
            self.reward_flash_color = (self.colors['reward_positive'] if reward > 0 
                                     else self.colors['reward_negative'])
        
        if self.reward_display_timer > 0:
            self.reward_display_timer -= 1
        
        # Clear screen with grass color
        self.screen.fill(self.colors['off_track'])
        
        # Draw track with enhanced visuals
        self._draw_enhanced_track()
        
        # Draw grid lines
        self._draw_grid()
        
        # Draw track progress indicators
        self._draw_progress_indicators(info)
        
        # Draw start/finish with animations
        self._draw_start_finish()
        
        # Draw car with rotation and shadow
        self._draw_enhanced_car(state, info)
        
        # Draw enhanced HUD
        self._draw_enhanced_hud(state, step, reward, total_reward, info, training_info)
        
        pygame.display.flip()
        self.frame_count += 1
        return self._get_rgb_array()

    def _draw_enhanced_track(self):
        """Draw the racing track with borders and styling"""
        for i, (x, y) in enumerate(self.track):
            rect = pygame.Rect(
                x * self.cell_size,
                y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            
            # Draw track surface
            pygame.draw.rect(self.screen, self.colors['track'], rect)
            
            # Draw track borders
            pygame.draw.rect(self.screen, self.colors['track_border'], rect, 2)
            

    def _draw_grid(self):
        """Draw subtle grid lines"""
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                self.screen, 
                self.colors['grid'],
                (x * self.cell_size, 0),
                (x * self.cell_size, self.grid_size * self.cell_size),
                1
            )
        for y in range(self.grid_size + 1):
            pygame.draw.line(
                self.screen, 
                self.colors['grid'],
                (0, y * self.cell_size),
                (self.grid_size * self.cell_size, y * self.cell_size),
                1
            )

    def _draw_progress_indicators(self, info: Optional[Dict]):
        """Draw track progress and next target indicator"""
        if not info:
            return
            
        # Highlight completed checkpoints
        current_index = info.get('track_index', 0)
        for i, (x, y) in enumerate(self.track):
            center = (
                x * self.cell_size + self.cell_size // 2,
                y * self.cell_size + self.cell_size // 2
            )
            
            if i <= current_index and i not in [0, len(self.track)-1]:
                # Completed checkpoint
                pygame.draw.circle(
                    self.screen,
                    self.colors['checkpoint'],
                    center,
                    8
                )
                pygame.draw.circle(
                    self.screen,
                    (255, 255, 255),
                    center,
                    8,
                    2
                )
        
        # Highlight next target
        if current_index < len(self.track) - 1:
            next_target = self.track[current_index + 1]
            target_center = (
                next_target[0] * self.cell_size + self.cell_size // 2,
                next_target[1] * self.cell_size + self.cell_size // 2
            )
            
            # Animated pulsing target
            pulse = int(15 * (1 + math.sin(self.frame_count * 0.2)))
            pygame.draw.circle(
                self.screen,
                self.colors['next_target'],
                target_center,
                15 + pulse,
                3
            )

    def _draw_start_finish(self):
        """Draw enhanced start and finish markers"""
        # Start (animated green)
        # Start (animated green)
        start_rect = pygame.Rect(
            1 * self.cell_size, 1 * self.cell_size, 
            self.cell_size, self.cell_size
        )

        # Pulsing green effect
        pulse_intensity = int(50 * (1 + math.sin(self.frame_count * 0.1)))
        start_color = (0, min(255, 200 + pulse_intensity), 0)

        pygame.draw.rect(self.screen, start_color, start_rect, 4)

        start_text = self.font_medium.render("START", True, (255, 255, 255))
        text_rect = start_text.get_rect(center=start_rect.center)
        self.screen.blit(start_text, text_rect)

        
        # Finish (animated checkered pattern)
        finish_rect = pygame.Rect(
            1 * self.cell_size, 2 * self.cell_size,
            self.cell_size, self.cell_size
        )
        
        # Animated checkered pattern
        checker_size = self.cell_size // 6
        offset = (self.frame_count // 10) % (checker_size * 2)
        
        for i in range(8):
            for j in range(8):
                x_pos = finish_rect.left + (i * checker_size) - offset
                y_pos = finish_rect.top + j * checker_size
                
                if 0 <= x_pos < finish_rect.right and 0 <= y_pos < finish_rect.bottom:
                    color = (255, 255, 255) if (i + j) % 2 == 0 else (0, 0, 0)
                    checker_rect = pygame.Rect(x_pos, y_pos, checker_size, checker_size)
                    checker_rect = checker_rect.clip(finish_rect)
                    pygame.draw.rect(self.screen, color, checker_rect)
        
        finish_text = self.font_medium.render("FINISH", True, (255, 255, 0))
        text_rect = finish_text.get_rect(center=finish_rect.center)
        self.screen.blit(finish_text, text_rect)

    def _draw_enhanced_car(self, state: np.ndarray, info: Optional[Dict]):
        """Draw F1 car with enhanced visuals and rotation"""
        y, x = state
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        
        # Determine car rotation based on movement direction
        if info and 'next_target' in info:
            target_x, target_y = info['next_target']
            dx = target_x - x
            dy = target_y - y
            
            if dx > 0:
                self.car_rotation = 0  # Right
            elif dx < 0:
                self.car_rotation = 180  # Left
            elif dy > 0:
                self.car_rotation = 90  # Down
            elif dy < 0:
                self.car_rotation = 270  # Up
        
        # Draw car shadow
        shadow_offset = 3
        shadow_points = [
            (center_x - 18 + shadow_offset, center_y - 10 + shadow_offset),
            (center_x + 18 + shadow_offset, center_y - 10 + shadow_offset),
            (center_x + 15 + shadow_offset, center_y + 10 + shadow_offset),
            (center_x - 15 + shadow_offset, center_y + 10 + shadow_offset)
        ]
        pygame.draw.polygon(self.screen, self.colors['car_shadow'], shadow_points)
        
        # Draw main car body
        car_points = [
            (center_x - 18, center_y - 10),
            (center_x + 18, center_y - 10),
            (center_x + 15, center_y + 10),
            (center_x - 15, center_y + 10)
        ]
        pygame.draw.polygon(self.screen, self.colors['car'], car_points)
        
        # Draw car details
        # Cockpit
        cockpit_rect = pygame.Rect(center_x - 8, center_y - 6, 16, 12)
        pygame.draw.rect(self.screen, (40, 40, 40), cockpit_rect, border_radius=3)
        
        # Front wing
        pygame.draw.rect(
            self.screen, 
            (100, 100, 100), 
            (center_x - 12, center_y - 12, 24, 4)
        )
        
        # Wheels
        wheel_positions = [
            (center_x - 12, center_y - 8),
            (center_x + 12, center_y - 8),
            (center_x - 12, center_y + 8),
            (center_x + 12, center_y + 8)
        ]
        
        for wheel_pos in wheel_positions:
            pygame.draw.circle(self.screen, (20, 20, 20), wheel_pos, 6)
            pygame.draw.circle(self.screen, (60, 60, 60), wheel_pos, 4)
        
        # Speed lines if moving fast (based on recent rewards)
        if hasattr(self, 'last_reward') and self.last_reward > 5:
            for i in range(3):
                line_start = (center_x - 30 - i*5, center_y - 2 + i*2)
                line_end = (center_x - 20 - i*5, center_y - 2 + i*2)
                pygame.draw.line(
                    self.screen, 
                    (255, 255, 255, 100), 
                    line_start, 
                    line_end, 
                    2
                )

    def _draw_enhanced_hud(self, state: np.ndarray, step: int, reward: float, 
                          total_reward: float, info: Optional[Dict], 
                          training_info: Optional[Dict]):
        """Draw comprehensive HUD with training metrics"""
        hud_y_start = self.grid_size * self.cell_size
        hud_rect = pygame.Rect(0, hud_y_start, self.width, 120)
        pygame.draw.rect(self.screen, self.colors['hud_bg'], hud_rect)
        
        # Title
        title_text = self.font_large.render("F1 DQN Training Environment", True, self.colors['hud_text'])
        self.screen.blit(title_text, (10, hud_y_start + 5))
        
        # Position and basic info
        y_pos = hud_y_start + 30
        pos_text = f"Position: ({state[1]}, {state[0]}) | Step: {step}"
        pos_surface = self.font_medium.render(pos_text, True, self.colors['hud_text'])
        self.screen.blit(pos_surface, (10, y_pos))
        
        # Reward display with color coding
        reward_color = self.reward_flash_color if self.reward_display_timer > 0 else self.colors['hud_text']
        reward_text = f"Reward: {reward:.1f} | Total: {total_reward:.1f}"
        reward_surface = self.font_medium.render(reward_text, True, reward_color)
        self.screen.blit(reward_surface, (200, y_pos))
        
        # Progress information
        y_pos += 20
        if info:
            progress_text = f"Track Progress: {info.get('track_index', 0)}/{len(self.track)-1} ({info.get('progress', 0):.1%})"
            progress_surface = self.font_medium.render(progress_text, True, self.colors['hud_text'])
            self.screen.blit(progress_surface, (10, y_pos))
            
            # Progress bar
            bar_x = 250
            bar_width = 200
            bar_height = 15
            progress_ratio = info.get('progress', 0)
            
            # Background
            pygame.draw.rect(
                self.screen, 
                self.colors['progress_bg'], 
                (bar_x, y_pos, bar_width, bar_height)
            )
            
            # Progress fill
            if progress_ratio > 0:
                pygame.draw.rect(
                    self.screen, 
                    self.colors['progress_bar'], 
                    (bar_x, y_pos, int(bar_width * progress_ratio), bar_height)
                )
            
            # Border
            pygame.draw.rect(
                self.screen, 
                self.colors['hud_text'], 
                (bar_x, y_pos, bar_width, bar_height), 
                2
            )
        
        # Training metrics
        y_pos += 25
        if training_info:
            completion_rate = training_info.get('completion_rate', 0)
            episode_count = training_info.get('episode_count', 0)
            
            training_text = f"Episodes: {episode_count} | Completion Rate: {completion_rate:.1%}"
            training_surface = self.font_medium.render(training_text, True, self.colors['hud_text'])
            self.screen.blit(training_surface, (10, y_pos))
        
        # Status indicators
        if info:
            status_x = self.width - 150
            on_track = info.get('on_track', False)
            status_color = self.colors['reward_positive'] if on_track else self.colors['reward_negative']
            status_text = "ON TRACK" if on_track else "OFF TRACK"
            status_surface = self.font_medium.render(status_text, True, status_color)
            self.screen.blit(status_surface, (status_x, y_pos))

    def _get_rgb_array(self) -> np.ndarray:
        """Convert screen to RGB array for gym compatibility"""
        return np.transpose(
            pygame.surfarray.array3d(self.screen),
            (1, 0, 2)
        )

    def close(self):
        """Clean up resources"""
        pygame.quit()

    def set_training_info(self, episode_count: int, completion_rate: float):
        """Update training information for display"""
        self.total_episodes = episode_count


def test_enhanced_renderer():
    """Test the enhanced renderer with the improved environment"""
    from improved_f1_env import ImprovedF1Env
    
    env = ImprovedF1Env()
    renderer = ImprovedF1Renderer()
    
    print("Testing Enhanced F1 Renderer")
    print("Watch the car navigate through the track with enhanced visuals!")
    
    # Test with a sequence of moves
    obs, _ = env.reset()
    total_reward = 0
    
    # Optimal path for the improved track
    optimal_actions = [2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 3, 3]
    
    for i, action in enumerate(optimal_actions + [0] * 5):  # Add some extra moves
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Simulate training info
        training_info = {
            'completion_rate': min(1.0, i / len(optimal_actions)),
            'episode_count': i + 1
        }
        
        # Render with enhanced visuals
        renderer.render(
            state=obs[:2].astype(int),  # Extract position from observation
            step=i + 1,
            reward=reward,
            total_reward=total_reward,
            info=info,
            training_info=training_info
        )
        
        pygame.time.wait(800)  # Pause to see the animation
        
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close()
                return
        
        if terminated:
            print("Track completed!")
            break
    
    # Keep final frame visible
    pygame.time.wait(3000)
    renderer.close()


if __name__ == "__main__":
    test_enhanced_renderer()