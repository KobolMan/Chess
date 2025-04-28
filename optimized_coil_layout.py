# optimized_coil_layout.py

import numpy as np
import math
import pygame

# Import for typing purposes
from typing import List, Tuple, Set, Dict, Optional, Union, Any

class OptimizedCoilLayout:
    """
    Implements the optimized electromagnet layout with:
    - Permanent magnets at the center of each square
    - Shared boundary electromagnets at the edges between squares
    - Selective corner electromagnets at intersections
    
    This class interfaces with the existing CoilGrid to maintain compatibility.
    """
    
    def __init__(self, board_squares=8, coil_grid_size=20):
        """
        Initialize the optimized coil layout.
        
        Args:
            board_squares: Number of squares on the chessboard (typically 8)
            coil_grid_size: Size of the original coil grid (for compatibility)
        """
        self.board_squares = board_squares
        self.coil_grid_size = coil_grid_size
        
        # Initialize coil positions
        self.permanent_magnets = []  # Center of each square (64 total)
        self.edge_coils = []         # Shared boundaries (112 total)
        self.corner_coils = []       # Strategic intersections (25-30 total)
        
        # Create coil mapping between optimized layout and original grid
        self.coil_mapping = {}  # Maps optimized coil positions to grid indices
        
        # Generate the optimized layout
        self._generate_layout()
        
        # Power and current arrays for the optimized coils
        self.permanent_magnet_strength = np.ones(len(self.permanent_magnets))
        self.edge_coil_power = np.zeros(len(self.edge_coils))
        self.edge_coil_current = np.zeros(len(self.edge_coils))
        self.corner_coil_power = np.zeros(len(self.corner_coils))
        self.corner_coil_current = np.zeros(len(self.corner_coils))
    
    def _generate_layout(self):
        """Generate the optimized coil layout positions."""
        # Generate permanent magnets at square centers
        for row in range(self.board_squares):
            for col in range(self.board_squares):
                # Convert to coil grid coordinates
                center_x = (col + 0.5) * self.coil_grid_size / self.board_squares
                center_y = (row + 0.5) * self.coil_grid_size / self.board_squares
                self.permanent_magnets.append((center_x, center_y))
        
        # Generate edge coils (shared boundaries)
        # Horizontal edges
        for row in range(self.board_squares + 1):
            for col in range(self.board_squares):
                edge_x = (col + 0.5) * self.coil_grid_size / self.board_squares
                edge_y = row * self.coil_grid_size / self.board_squares
                self.edge_coils.append((edge_x, edge_y, "horizontal"))
        
        # Vertical edges
        for row in range(self.board_squares):
            for col in range(self.board_squares + 1):
                edge_x = col * self.coil_grid_size / self.board_squares
                edge_y = (row + 0.5) * self.coil_grid_size / self.board_squares
                self.edge_coils.append((edge_x, edge_y, "vertical"))
        
        # Generate corner coils (intersections)
        # We'll use a selective approach - place at key intersections
        for row in range(0, self.board_squares + 1, 2):
            for col in range(0, self.board_squares + 1, 2):
                corner_x = col * self.coil_grid_size / self.board_squares
                corner_y = row * self.coil_grid_size / self.board_squares
                self.corner_coils.append((corner_x, corner_y))
        
        # Add extra corners for diagonal control
        for row in range(1, self.board_squares, 2):
            for col in range(1, self.board_squares, 2):
                if (row + col) % 4 == 0:  # Selective pattern for diagonals
                    corner_x = col * self.coil_grid_size / self.board_squares
                    corner_y = row * self.coil_grid_size / self.board_squares
                    self.corner_coils.append((corner_x, corner_y))
    
    def map_to_grid(self, coil_grid):
        """
        Maps the optimized coil layout to the original grid indices.
        This maintains compatibility with the existing visualization.
        
        Args:
            coil_grid: The existing CoilGrid object to map to
        """
        # Reset the grid
        coil_grid.reset()
        
        # Map each optimized coil to the nearest grid position
        grid_size = coil_grid.size
        
        # Function to find nearest grid index
        def find_nearest_grid_index(x, y):
            grid_x = min(max(0, round(x)), grid_size - 1)
            grid_y = min(max(0, round(y)), grid_size - 1)
            return grid_y, grid_x  # Row, Col format for grid indexing
        
        # Map permanent magnets (static, just for visualization)
        for i, (x, y) in enumerate(self.permanent_magnets):
            grid_idx = find_nearest_grid_index(x, y)
            # Store mapping for reference
            self.coil_mapping[('permanent', i)] = grid_idx
            # Permanent magnets are represented with a special marker
            coil_grid.update_coil(grid_idx[0], grid_idx[1], 10, 0)  # Low power, neutral current
        
        # Map edge coils
        for i, (x, y, orientation) in enumerate(self.edge_coils):
            grid_idx = find_nearest_grid_index(x, y)
            self.coil_mapping[('edge', i)] = grid_idx
            power = self.edge_coil_power[i]
            current = self.edge_coil_current[i]
            if power > 0:
                coil_grid.update_coil(grid_idx[0], grid_idx[1], power, current)
        
        # Map corner coils
        for i, (x, y) in enumerate(self.corner_coils):
            grid_idx = find_nearest_grid_index(x, y)
            self.coil_mapping[('corner', i)] = grid_idx
            power = self.corner_coil_power[i]
            current = self.corner_coil_current[i]
            if power > 0:
                coil_grid.update_coil(grid_idx[0], grid_idx[1], power, current)
    
    def activate_pattern(self, pattern_type, piece_position, target=None, intensity=100, blocked_coils=None):
        """
        Activates a coil pattern based on the optimized layout.
        
        Args:
            pattern_type: Type of movement pattern ("directed", "knight", etc.)
            piece_position: Current piece position (col, row) in board coordinates
            target: Target position (col, row) in board coordinates
            intensity: Desired intensity (0-100)
            blocked_coils: Set of coil positions to avoid
        """
        # Reset coil states
        self.edge_coil_power.fill(0)
        self.edge_coil_current.fill(0)
        self.corner_coil_power.fill(0)
        self.corner_coil_current.fill(0)
        
        # Convert board positions to coil grid coordinates
        piece_x = piece_position[0] * self.coil_grid_size / self.board_squares
        piece_y = piece_position[1] * self.coil_grid_size / self.board_squares
        
        if target is not None:
            target_x = target[0] * self.coil_grid_size / self.board_squares
            target_y = target[1] * self.coil_grid_size / self.board_squares
            
            # Calculate direction and distance
            dx = target_x - piece_x
            dy = target_y - piece_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Skip if already at target
            if distance < 0.1:
                return
            
            # Normalize direction
            if distance > 0:
                dir_x = dx / distance
                dir_y = dy / distance
            else:
                dir_x, dir_y = 0, 0
        else:
            target_x, target_y = None, None
            dir_x, dir_y, distance = 0, 0, 0
        
        # Identify nearest coils to activate based on pattern and direction
        if pattern_type == "directed":
            self._activate_directed_pattern(piece_x, piece_y, target_x, target_y, dir_x, dir_y, distance, intensity, blocked_coils)
        elif pattern_type == "knight":
            self._activate_knight_pattern(piece_x, piece_y, target_x, target_y, intensity, blocked_coils)
        elif pattern_type == "straight_horizontal":
            self._activate_straight_pattern(piece_x, piece_y, target_x, target_y, True, intensity, blocked_coils)
        elif pattern_type == "straight_vertical":
            self._activate_straight_pattern(piece_x, piece_y, target_x, target_y, False, intensity, blocked_coils)
        elif pattern_type == "radial":
            self._activate_radial_pattern(piece_x, piece_y, intensity, blocked_coils)
    
    def _activate_directed_pattern(self, piece_x, piece_y, target_x, target_y, dir_x, dir_y, distance, intensity, blocked_coils):
        """Activate coils to move a piece in the specified direction."""
        # Determine which edge coils to activate
        for i, (x, y, orientation) in enumerate(self.edge_coils):
            # Calculate vector from piece to this coil
            vec_x = x - piece_x
            vec_y = y - piece_y
            coil_dist = np.sqrt(vec_x**2 + vec_y**2)
            
            # Skip if too far
            if coil_dist > 2.5:
                continue
            
            # Calculate dot product with direction
            proj = vec_x * dir_x + vec_y * dir_y
            
            # Power based on distance and alignment with direction
            power = intensity * max(0, 1 - coil_dist/2.5)**2
            
            # Project magnitude perpendicular to direction
            perp_dist_sq = max(0, coil_dist**2 - proj**2)
            perp_dist = np.sqrt(perp_dist_sq)
            
            # Apply direction focus
            direction_focus = max(0, 1 - perp_dist/1.5)**2
            power *= direction_focus
            
            # Set current direction based on projection
            current = -1 if proj > 0.1 else 1 if proj < -0.1 else 0
            
            # Apply if not blocked
            if not self._is_coil_blocked(x, y, blocked_coils):
                self.edge_coil_power[i] = power
                self.edge_coil_current[i] = current
        
        # Activate corner coils for diagonal movement
        if abs(dir_x) > 0.3 and abs(dir_y) > 0.3:  # Moving diagonally
            for i, (x, y) in enumerate(self.corner_coils):
                vec_x = x - piece_x
                vec_y = y - piece_y
                coil_dist = np.sqrt(vec_x**2 + vec_y**2)
                
                if coil_dist > 2.5:
                    continue
                
                # Calculate dot product with direction
                proj = vec_x * dir_x + vec_y * dir_y
                
                # Power calculation
                power = intensity * max(0, 1 - coil_dist/2.5)**2
                
                # Apply direction focus
                perp_dist_sq = max(0, coil_dist**2 - proj**2)
                perp_dist = np.sqrt(perp_dist_sq)
                direction_focus = max(0, 1 - perp_dist/1.0)**2 
                power *= direction_focus
                
                # Set current direction based on projection
                current = -1 if proj > 0.1 else 1 if proj < -0.1 else 0
                
                # Apply if not blocked
                if not self._is_coil_blocked(x, y, blocked_coils):
                    self.corner_coil_power[i] = power
                    self.corner_coil_current[i] = current
    
    def _activate_knight_pattern(self, piece_x, piece_y, target_x, target_y, intensity, blocked_coils):
        """Special pattern for knight moves."""
        # Knight moves use a combination of edge and corner coils
        # First use directed movement
        dx = target_x - piece_x
        dy = target_y - piece_y
        distance = np.sqrt(dx**2 + dy**2)
        if distance > 0:
            dir_x = dx / distance
            dir_y = dy / distance
        else:
            dir_x, dir_y = 0, 0
            
        self._activate_directed_pattern(piece_x, piece_y, target_x, target_y, dir_x, dir_y, distance, intensity, blocked_coils)
        
        # Then add specific L-shape enhancement
        # For an L-shape, we need to activate coils along both segments
        # Let's break it into two parts: the long segment and the short segment
        if abs(dx) > abs(dy):
            # Horizontal longer segment
            seg1_dir_x = np.sign(dx)
            seg1_dir_y = 0
            seg2_dir_x = 0
            seg2_dir_y = np.sign(dy)
        else:
            # Vertical longer segment
            seg1_dir_x = 0
            seg1_dir_y = np.sign(dy)
            seg2_dir_x = np.sign(dx)
            seg2_dir_y = 0
            
        # Enhance activation for key coils along these segments
        for i, (x, y, orientation) in enumerate(self.edge_coils):
            vec_x = x - piece_x
            vec_y = y - piece_y
            
            # Check alignment with first segment
            proj_seg1 = vec_x * seg1_dir_x + vec_y * seg1_dir_y
            
            # Check alignment with second segment
            proj_seg2 = vec_x * seg2_dir_x + vec_y * seg2_dir_y
            
            if proj_seg1 > 0.5 or proj_seg2 > 0.5:
                # Enhance this coil if it's aligned with either segment
                if self.edge_coil_power[i] > 0:
                    self.edge_coil_power[i] *= 1.5  # Increase power
                    self.edge_coil_power[i] = min(100, self.edge_coil_power[i])  # Cap at 100
    
    def _activate_straight_pattern(self, piece_x, piece_y, target_x, target_y, horizontal, intensity, blocked_coils):
        """Activate coils for straight line movement (horizontal or vertical)."""
        if horizontal:
            # Activate horizontal edge coils
            dir_x = np.sign(target_x - piece_x) if target_x != piece_x else 0
            
            for i, (x, y, orientation) in enumerate(self.edge_coils):
                if orientation != "horizontal":
                    continue
                    
                # Check if this coil is in the path
                if abs(y - piece_y) > 0.5:  # Not in the same row
                    continue
                    
                # Calculate distance and direction
                dx = x - piece_x
                coil_dist = abs(dx)
                
                # Skip if too far
                if coil_dist > 3.0:
                    continue
                
                # Set power based on distance
                power = intensity * max(0, 1 - coil_dist/3.0)**2
                
                # Current direction based on target
                current = -1 if dx * dir_x > 0 else 1 if dx * dir_x < 0 else 0
                
                # Apply if not blocked
                if not self._is_coil_blocked(x, y, blocked_coils):
                    self.edge_coil_power[i] = power
                    self.edge_coil_current[i] = current
        else:
            # Activate vertical edge coils
            dir_y = np.sign(target_y - piece_y) if target_y != piece_y else 0
            
            for i, (x, y, orientation) in enumerate(self.edge_coils):
                if orientation != "vertical":
                    continue
                    
                # Check if this coil is in the path
                if abs(x - piece_x) > 0.5:  # Not in the same column
                    continue
                    
                # Calculate distance and direction
                dy = y - piece_y
                coil_dist = abs(dy)
                
                # Skip if too far
                if coil_dist > 3.0:
                    continue
                
                # Set power based on distance
                power = intensity * max(0, 1 - coil_dist/3.0)**2
                
                # Current direction based on target
                current = -1 if dy * dir_y > 0 else 1 if dy * dir_y < 0 else 0
                
                # Apply if not blocked
                if not self._is_coil_blocked(x, y, blocked_coils):
                    self.edge_coil_power[i] = power
                    self.edge_coil_current[i] = current
    
    def _activate_radial_pattern(self, piece_x, piece_y, intensity, blocked_coils):
        """Activate coils in a radial pattern around the piece."""
        # Activate nearby edge coils
        for i, (x, y, _) in enumerate(self.edge_coils):
            # Calculate distance
            dx = x - piece_x
            dy = y - piece_y
            coil_dist = np.sqrt(dx**2 + dy**2)
            
            # Skip if too far
            if coil_dist > 2.0:
                continue
            
            # Power decreases with distance
            power = intensity * max(0, 1 - coil_dist/2.0)**2
            
            # Current direction (attract toward piece)
            current = -1  # Attract
            
            # Apply if not blocked
            if not self._is_coil_blocked(x, y, blocked_coils):
                self.edge_coil_power[i] = power
                self.edge_coil_current[i] = current
        
        # Activate nearby corner coils
        for i, (x, y) in enumerate(self.corner_coils):
            # Calculate distance
            dx = x - piece_x
            dy = y - piece_y
            coil_dist = np.sqrt(dx**2 + dy**2)
            
            # Skip if too far
            if coil_dist > 2.0:
                continue
            
            # Power decreases with distance
            power = intensity * max(0, 1 - coil_dist/2.0)**2
            
            # Current direction (attract toward piece)
            current = -1  # Attract
            
            # Apply if not blocked
            if not self._is_coil_blocked(x, y, blocked_coils):
                self.corner_coil_power[i] = power
                self.corner_coil_current[i] = current
    
    def _is_coil_blocked(self, x, y, blocked_coils):
        """Check if a coil position is in the blocked set."""
        if blocked_coils is None:
            return False
            
        # Convert to grid indices for comparison with blocked_coils
        grid_y = min(max(0, round(y)), self.coil_grid_size - 1)
        grid_x = min(max(0, round(x)), self.coil_grid_size - 1)
        
        return (grid_y, grid_x) in blocked_coils
    
    def visualize(self, surface, board_pixel_size, x_offset=0, y_offset=0):
        """
        Draw a custom visualization of the optimized coil layout.
        
        Args:
            surface: Pygame surface to draw on
            board_pixel_size: Size of the board in pixels
            x_offset: Horizontal offset for drawing
            y_offset: Vertical offset for drawing
        """
        # Create a transparent surface
        coil_surface = pygame.Surface((board_pixel_size, board_pixel_size), pygame.SRCALPHA)
        
        # Calculate pixel size of a square
        square_size = board_pixel_size / self.board_squares
        
        # Function to convert coil grid coords to pixels
        def coil_to_pixel(x, y):
            px = x * board_pixel_size / self.coil_grid_size
            py = y * board_pixel_size / self.coil_grid_size
            return int(px), int(py)
        
        # Draw permanent magnets (square centers)
        for i, (x, y) in enumerate(self.permanent_magnets):
            px, py = coil_to_pixel(x, y)
            # Draw a circle with a distinctive pattern
            pygame.draw.circle(coil_surface, (0, 100, 0, 150), (px, py), int(square_size * 0.15))
            pygame.draw.circle(coil_surface, (0, 150, 0, 100), (px, py), int(square_size * 0.1))
        
        # Draw edge coils
        for i, (x, y, orientation) in enumerate(self.edge_coils):
            px, py = coil_to_pixel(x, y)
            power = self.edge_coil_power[i]
            current = self.edge_coil_current[i]
            
            # Size based on power
            radius = int(square_size * 0.1 * (0.6 + 0.4 * power / 100))
            min_radius = int(square_size * 0.08)
            
            # Color based on current direction (red=repel, blue=attract)
            if power > 0:
                alpha = int(np.clip(power * 2.0, 50, 200))
                color = (255, 100, 100, alpha) if current > 0 else (100, 100, 255, alpha)
                pygame.draw.circle(coil_surface, color, (px, py), radius)
            else:
                # Inactive coil outline
                edge_color = (200, 200, 200, 100) if orientation == "horizontal" else (180, 180, 180, 100)
                pygame.draw.circle(coil_surface, edge_color, (px, py), min_radius, 1)
        
        # Draw corner coils
        for i, (x, y) in enumerate(self.corner_coils):
            px, py = coil_to_pixel(x, y)
            power = self.corner_coil_power[i]
            current = self.corner_coil_current[i]
            
            # Size based on power
            radius = int(square_size * 0.12 * (0.6 + 0.4 * power / 100))
            min_radius = int(square_size * 0.1)
            
            # Color based on current direction
            if power > 0:
                alpha = int(np.clip(power * 2.0, 50, 200))
                color = (255, 100, 100, alpha) if current > 0 else (100, 100, 255, alpha)
                pygame.draw.circle(coil_surface, color, (px, py), radius)
            else:
                # Inactive coil outline - diamond shape for corners
                corner_color = (150, 150, 150, 100)
                size = min_radius
                points = [
                    (px, py - size),
                    (px + size, py),
                    (px, py + size),
                    (px - size, py)
                ]
                pygame.draw.polygon(coil_surface, corner_color, points, 1)
        
        # Draw the coil surface onto the main surface
        surface.blit(coil_surface, (x_offset, y_offset))
    
    def update_magnetic_field(self, coil_grid):
        """
        Update the magnetic field in the original coil grid based on the optimized layout.
        This maintains compatibility with the existing field visualization.
        
        Args:
            coil_grid: The existing CoilGrid object to update
        """
        # First map our layout to the grid
        self.map_to_grid(coil_grid)
        
        # Let the original grid calculate its field
        coil_grid.update_magnetic_field()

# Example usage:
# optimized_layout = OptimizedCoilLayout()
# coil_grid = CoilGrid(size=20, board_squares=8)
# 
# # To visualize and control:
# optimized_layout.activate_pattern("directed", piece_position, target_position, 100)
# optimized_layout.visualize(surface, board_pixel_size)
# 
# # To use with existing field visualization:
# optimized_layout.update_magnetic_field(coil_grid)
# coil_grid.draw_field_overlay(surface, board_pixel_size)