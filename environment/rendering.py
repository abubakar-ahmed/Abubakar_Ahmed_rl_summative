import pygame
import numpy as np
import math
from typing import Tuple, List, Optional


class RaceTrackRenderer:
    """
    High-quality renderer for the Formula 1 racing environment.
    
    Features:
    - Realistic F1 track visualization
    - Animated racing car
    - Professional HUD with telemetry
    - Checkered flag finish line
    - Racing-themed color scheme
    """
    
    def __init__(self, grid_size: int = 6, cell_size: int = 100):
        """
        Initialize the racing track renderer.
        
        Args:
            grid_size: Size of the grid (6x6)
            cell_size: Size of each grid cell in pixels
        """
        pygame.init()
        
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.width = grid_size * cell_size
        self.height = grid_size * cell_size
        
        # Create display surface
        self.screen = pygame.display.set_mode((self.width, self.height + 80))  # Extra space for HUD
        pygame.display.set_caption("Formula 1 Path Optimization Simulator")
        
        # Fonts for HUD
        self.font_large = pygame.font.SysFont('arial', 24, bold=True)
        self.font_medium = pygame.font.SysFont('arial', 18)
        self.font_small = pygame.font.SysFont('arial', 14)
        
        # Define the racing track (from environment)
        self.track = [(0,0), (0,1), (1,1), (1,2), (2,2), (2,3), 
                      (3,3), (3,4), (4,4), (4,5), (5,5)]
        self.track_set = set(self.track)
        
        # Colors (F1 themed)
        self.colors = {
            'track': (50, 50, 50),          # Dark asphalt
            'racing_line': (100, 100, 100), # Racing line
            'off_track': (34, 139, 34),     # Grass green
            'barriers': (220, 20, 60),      # Barrier red
            'car': (255, 0, 0),             # Ferrari red
            'car_detail': (255, 255, 255),  # White details
            'start_finish': (255, 255, 255), # White for start/finish
            'hud_bg': (0, 0, 0),            # Black HUD background
            'hud_text': (255, 255, 255),    # White HUD text
            'success': (0, 255, 0),         # Green for success
            'warning': (255, 165, 0),       # Orange for warnings
            'danger': (255, 0, 0)           # Red for danger
        }
        
        # Animation state
        self.car_rotation = 0
        self.frame_count = 0
        
    def render(self, state: np.ndarray, step: int, reward: float, 
               total_reward: float = 0, info: dict = None) -> np.ndarray:
        """
        Render the current state of the F1 environment.
        
        Args:
            state: Current agent position [y, x]
            step: Current step number
            reward: Reward for current step
            total_reward: Cumulative reward
            info: Additional information from environment
            
        Returns:
            RGB array of the rendered frame
        """
        self.frame_count += 1
        
        # Clear screen with track background
        self.screen.fill(self.colors['off_track'])
        
        # Draw the racing circuit
        self._draw_track()
        
        # Draw grid lines (subtle)
        self._draw_grid_lines()
        
        # Draw start and finish areas
        self._draw_start_finish()
        
        # Draw the F1 car
        self._draw_f1_car(state)
        
        # Draw HUD with telemetry
        self._draw_hud(state, step, reward, total_reward, info)
        
        # Update display
        pygame.display.flip()
        
        # Return RGB array for GIF creation
        return self._get_rgb_array()
    
    def _draw_track(self):
        """Draw the racing track with realistic appearance."""
        # Draw track surface
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                
                if (x, y) in self.track_set:
                    # Track surface (asphalt)
                    pygame.draw.rect(self.screen, self.colors['track'], rect)
                    
                    # Add track texture (subtle lines)
                    for i in range(0, self.cell_size, 20):
                        pygame.draw.line(self.screen, (60, 60, 60), 
                                       (rect.left + i, rect.top), 
                                       (rect.left + i, rect.bottom), 1)
                else:
                    # Off-track area (grass)
                    pygame.draw.rect(self.screen, self.colors['off_track'], rect)
                    
                    # Add grass texture
                    for i in range(0, self.cell_size, 10):
                        for j in range(0, self.cell_size, 10):
                            if (i + j) % 20 == 0:
                                pygame.draw.circle(self.screen, (20, 120, 20),
                                                 (rect.left + i + 5, rect.top + j + 5), 2)
    
    def _draw_grid_lines(self):
        """Draw subtle grid lines for reference."""
        grid_color = (70, 70, 70)
        
        # Vertical lines
        for x in range(self.grid_size + 1):
            pygame.draw.line(self.screen, grid_color,
                           (x * self.cell_size, 0),
                           (x * self.cell_size, self.height), 1)
        
        # Horizontal lines  
        for y in range(self.grid_size + 1):
            pygame.draw.line(self.screen, grid_color,
                           (0, y * self.cell_size),
                           (self.width, y * self.cell_size), 1)
    
    def _draw_start_finish(self):
        """Draw start/finish line with checkered pattern."""
        # Start position (0,0) - Green light
        start_rect = pygame.Rect(0, 0, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 200, 0), start_rect, 5)
        
        # Add "START" text
        start_text = self.font_small.render("START", True, (255, 255, 255))
        self.screen.blit(start_text, (5, 5))
        
        # Finish position (5,5) - Checkered flag pattern
        finish_rect = pygame.Rect(5 * self.cell_size, 5 * self.cell_size, 
                                self.cell_size, self.cell_size)
        
        # Checkered pattern
        checker_size = self.cell_size // 8
        for i in range(8):
            for j in range(8):
                color = (255, 255, 255) if (i + j) % 2 == 0 else (0, 0, 0)
                checker_rect = pygame.Rect(
                    finish_rect.left + i * checker_size,
                    finish_rect.top + j * checker_size,
                    checker_size, checker_size
                )
                pygame.draw.rect(self.screen, color, checker_rect)
        
        # Add "FINISH" text
        finish_text = self.font_small.render("FINISH", True, (255, 255, 0))
        self.screen.blit(finish_text, (finish_rect.left + 5, finish_rect.top + 5))
    
    def _draw_f1_car(self, state: np.ndarray):
        """Draw an animated F1 racing car."""
        agent_y, agent_x = state
        
        # Calculate car center position
        car_center_x = agent_x * self.cell_size + self.cell_size // 2
        car_center_y = agent_y * self.cell_size + self.cell_size // 2
        
        # Car dimensions
        car_length = 40
        car_width = 20
        
        # Create car body rectangle
        car_rect = pygame.Rect(
            car_center_x - car_length // 2,
            car_center_y - car_width // 2,
            car_length, car_width
        )
        
        # Draw car body (Ferrari red)
        pygame.draw.rect(self.screen, self.colors['car'], car_rect, border_radius=5)
        
        # Draw car details
        # Front wing
        wing_rect = pygame.Rect(car_rect.right - 5, car_rect.top - 2, 8, car_width + 4)
        pygame.draw.rect(self.screen, self.colors['car_detail'], wing_rect)
        
        # Cockpit
        cockpit_rect = pygame.Rect(car_rect.left + 10, car_rect.top + 3, 15, car_width - 6)
        pygame.draw.rect(self.screen, (50, 50, 50), cockpit_rect, border_radius=3)
        
        # Wheels
        wheel_radius = 6
        # Front wheels
        pygame.draw.circle(self.screen, (30, 30, 30), 
                         (car_rect.right - 8, car_rect.top), wheel_radius)
        pygame.draw.circle(self.screen, (30, 30, 30), 
                         (car_rect.right - 8, car_rect.bottom), wheel_radius)
        # Rear wheels
        pygame.draw.circle(self.screen, (30, 30, 30), 
                         (car_rect.left + 8, car_rect.top), wheel_radius)
        pygame.draw.circle(self.screen, (30, 30, 30), 
                         (car_rect.left + 8, car_rect.bottom), wheel_radius)
        
        # Racing number
        number_text = self.font_small.render("44", True, (255, 255, 255))
        number_rect = number_text.get_rect(center=(car_center_x, car_center_y))
        self.screen.blit(number_text, number_rect)
        
        # Speed effect (if moving)
        if self.frame_count > 1:  # Car has moved
            for i in range(3):
                trail_x = car_center_x - car_length//2 - 10 - i*5
                trail_y = car_center_y + (i-1)*3
                pygame.draw.circle(self.screen, (100, 100, 100, 50), 
                                 (trail_x, trail_y), 2)
    
    def _draw_hud(self, state: np.ndarray, step: int, reward: float, 
                  total_reward: float, info: dict):
        """Draw professional racing HUD with telemetry."""
        hud_y = self.height
        hud_height = 80
        
        # HUD background
        hud_rect = pygame.Rect(0, hud_y, self.width, hud_height)
        pygame.draw.rect(self.screen, self.colors['hud_bg'], hud_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), hud_rect, 2)
        
        # Position information
        pos_text = f"POSITION: ({state[1]}, {state[0]})"
        pos_surface = self.font_medium.render(pos_text, True, self.colors['hud_text'])
        self.screen.blit(pos_surface, (10, hud_y + 10))
        
        # Lap information
        lap_text = f"LAP TIME: {step:03d} steps"
        lap_surface = self.font_medium.render(lap_text, True, self.colors['hud_text'])
        self.screen.blit(lap_surface, (10, hud_y + 35))
        
        # Reward information
        reward_color = self.colors['success'] if reward > 0 else \
                      self.colors['warning'] if reward > -0.5 else self.colors['danger']
        reward_text = f"REWARD: {reward:+.2f}"
        reward_surface = self.font_medium.render(reward_text, True, reward_color)
        self.screen.blit(reward_surface, (200, hud_y + 10))
        
        # Total reward
        total_text = f"TOTAL: {total_reward:+.1f}"
        total_surface = self.font_medium.render(total_text, True, self.colors['hud_text'])
        self.screen.blit(total_surface, (200, hud_y + 35))
        
        # Track status
        if info:
            status_color = self.colors['success'] if info.get('on_track', True) else self.colors['danger']
            status_text = "ON TRACK" if info.get('on_track', True) else "OFF TRACK"
            status_surface = self.font_medium.render(status_text, True, status_color)
            self.screen.blit(status_surface, (350, hud_y + 10))
            
            # Distance to goal
            distance = info.get('distance_to_goal', 0)
            dist_text = f"TO FINISH: {distance}"
            dist_surface = self.font_medium.render(dist_text, True, self.colors['hud_text'])
            self.screen.blit(dist_surface, (350, hud_y + 35))
        
        # Performance indicator
        if step > 0:
            efficiency = max(0, min(100, (1000 - step * 10)))  # Simple efficiency metric
            perf_text = f"EFFICIENCY: {efficiency:.0f}%"
            perf_color = self.colors['success'] if efficiency > 70 else \
                        self.colors['warning'] if efficiency > 40 else self.colors['danger']
            perf_surface = self.font_medium.render(perf_text, True, perf_color)
            self.screen.blit(perf_surface, (480, hud_y + 25))
    
    def _get_rgb_array(self) -> np.ndarray:
        """Convert the pygame surface to RGB array for GIF creation."""
        # Get the surface as a 3D array
        surface_array = pygame.surfarray.array3d(self.screen)
        
        # Pygame uses (width, height, channels), we need (height, width, channels)
        # Also, pygame uses RGB but we need to ensure correct format
        rgb_array = np.transpose(surface_array, (1, 0, 2))
        
        return rgb_array
    
    def close(self):
        """Clean up pygame resources."""
        pygame.quit()


# Test function for the renderer
def test_renderer():
    """Test the rendering system with a simple animation."""
    print("Testing F1 Track Renderer...")
    
    renderer = RaceTrackRenderer()
    
    # Simulate a short race
    test_positions = [(0,0), (0,1), (1,1), (1,2), (2,2), (2,3), (3,3)]
    
    for i, pos in enumerate(test_positions):
        state = np.array([pos[1], pos[0]], dtype=np.int32)  # Note: y, x format
        reward = -0.1 if i < len(test_positions)-1 else 10.0
        total_reward = sum([-0.1] * i) + (10.0 if i == len(test_positions)-1 else 0)
        
        info = {
            'on_track': True,
            'distance_to_goal': abs(pos[0] - 5) + abs(pos[1] - 5),
            'steps_taken': i + 1
        }
        
        renderer.render(state, i + 1, reward, total_reward, info)
        
        # Keep window open briefly
        pygame.time.wait(1000)
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close()
                return
    
    print("Renderer test completed!")
    pygame.time.wait(2000)  # Keep window open for 2 seconds
    renderer.close()


if __name__ == "__main__":
    test_renderer()