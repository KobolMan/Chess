#!/usr/bin/env python3
# Electromagnetic Chess Coil Simulation
# Simulates a 20x20 grid of electromagnetic coils controlling chess pieces

import pygame
import numpy as np
import math
import sys
import time
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# Initialize pygame
pygame.init()

# Constants
BOARD_SIZE = 800  # Pixels
BOARD_SQUARES = 8
SQUARE_SIZE = BOARD_SIZE // BOARD_SQUARES
COIL_GRID_SIZE = 20  # 20x20 coil grid
COIL_SIZE = BOARD_SIZE // COIL_GRID_SIZE
WINDOW_WIDTH = BOARD_SIZE + 400  # Extra space for controls and info
WINDOW_HEIGHT = BOARD_SIZE + 200  # Extra space for visualization
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
HIGHLIGHT = (124, 252, 0)
TRANSPARENT = (0, 0, 0, 100)

# Piece specifications
class PieceType(Enum):
    KING = 'k'
    QUEEN = 'q'
    BISHOP = 'b'
    KNIGHT = 'n'
    ROOK = 'r'
    PAWN = 'p'

class PieceColor(Enum):
    WHITE = 'w'
    BLACK = 'b'

# Piece size and magnetic properties
PIECE_PROPERTIES = {
    PieceType.KING: {'diameter': 40, 'height': 60, 'magnet_strength': 1.0},
    PieceType.QUEEN: {'diameter': 38, 'height': 55, 'magnet_strength': 1.0},
    PieceType.BISHOP: {'diameter': 35, 'height': 50, 'magnet_strength': 0.9},
    PieceType.KNIGHT: {'diameter': 35, 'height': 50, 'magnet_strength': 0.9},
    PieceType.ROOK: {'diameter': 33, 'height': 45, 'magnet_strength': 0.8},
    PieceType.PAWN: {'diameter': 29, 'height': 40, 'magnet_strength': 0.7}
}

# Move types
class MoveType(Enum):
    STRAIGHT = 'straight'
    DIAGONAL = 'diagonal'
    KNIGHT = 'knight'
    CUSTOM = 'custom'

# Define the chess pieces with Unicode symbols
PIECE_SYMBOLS = {
    (PieceColor.WHITE, PieceType.KING): '♔',
    (PieceColor.WHITE, PieceType.QUEEN): '♕',
    (PieceColor.WHITE, PieceType.ROOK): '♖',
    (PieceColor.WHITE, PieceType.BISHOP): '♗',
    (PieceColor.WHITE, PieceType.KNIGHT): '♘',
    (PieceColor.WHITE, PieceType.PAWN): '♙',
    (PieceColor.BLACK, PieceType.KING): '♚',
    (PieceColor.BLACK, PieceType.QUEEN): '♛',
    (PieceColor.BLACK, PieceType.ROOK): '♜',
    (PieceColor.BLACK, PieceType.BISHOP): '♝',
    (PieceColor.BLACK, PieceType.KNIGHT): '♞',
    (PieceColor.BLACK, PieceType.PAWN): '♟'
}

class ChessPiece:
    """Represents a chess piece with physical and magnetic properties"""
    def __init__(self, color, piece_type, position):
        self.color = color
        self.piece_type = piece_type
        self.position = np.array(position, dtype=float)  # (row, col) in chess coordinates
        self.velocity = np.array([0.0, 0.0])
        self.properties = PIECE_PROPERTIES[piece_type]
        self.diameter = self.properties['diameter']
        self.height = self.properties['height']
        self.magnet_strength = self.properties['magnet_strength']
        self.symbol = PIECE_SYMBOLS[(color, piece_type)]
        self.path = []  # For storing path history
        self.active = True  # Whether the piece is active (not captured)
        self.capture_path = []  # Path to follow when captured
        self.original_position = np.array(position, dtype=float)  # Store initial position
        
    def get_board_position(self):
        """Get position in board coordinates (0-7, 0-7)"""
        return self.position
    
    def get_pixel_position(self):
        """Get position in pixel coordinates for rendering"""
        col, row = self.position
        return (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                row * SQUARE_SIZE + SQUARE_SIZE // 2)
    
    def get_coil_position(self):
        """Get position in coil grid coordinates (0-19, 0-19)"""
        col, row = self.position
        return (col * (COIL_GRID_SIZE / BOARD_SQUARES),
                row * (COIL_GRID_SIZE / BOARD_SQUARES))
    
    def apply_force(self, force, dt, damping=0.7):
        """Apply force to the piece and update its position"""
        # Mass is proportional to diameter^2 * height
        mass = (self.diameter ** 2) * self.height * 0.0001
        
        # Calculate acceleration (F = ma)
        acceleration = force / mass
        
        # Limit maximum acceleration
        max_accel = 50.0
        accel_mag = np.linalg.norm(acceleration)
        if accel_mag > max_accel:
            acceleration = acceleration * (max_accel / accel_mag)
            
        # Update velocity with damping
        self.velocity = self.velocity * (1 - damping * dt) + acceleration * dt
        
        # Limit maximum velocity
        max_velocity = 5.0
        vel_mag = np.linalg.norm(self.velocity)
        if vel_mag > max_velocity:
            self.velocity = self.velocity * (max_velocity / vel_mag)
            
        # Update position
        new_position = self.position + self.velocity * dt
        
        # Keep on board (add some margin)
        margin = 0.1
        new_position[0] = np.clip(new_position[0], -margin, BOARD_SQUARES - 1 + margin)
        new_position[1] = np.clip(new_position[1], -margin, BOARD_SQUARES - 1 + margin)
        
        self.position = new_position
        
        # Record position in path history
        self.path.append(self.position.copy())
        if len(self.path) > 100:  # Limit path length
            self.path.pop(0)
    
    def update_capture_path(self, target_position, num_steps=50):
        """Set a path to follow when captured"""
        start = self.position.copy()
        end = np.array(target_position)
        
        # Create a smooth path
        self.capture_path = []
        for i in range(num_steps):
            t = i / (num_steps - 1)
            # Use an ease-out curve
            t = 1 - (1 - t) ** 2
            pos = start + t * (end - start)
            self.capture_path.append(pos)
    
    def follow_capture_path(self, step):
        """Follow the capture path"""
        if step < len(self.capture_path):
            self.position = self.capture_path[step]
            self.path.append(self.position.copy())
            return True
        return False
    
    def draw(self, surface, font, selected=False):
        """Draw the chess piece on the given surface"""
        # Skip if not active
        if not self.active:
            return
            
        # Get position in pixels
        x, y = self.get_pixel_position()
        
        # Draw the piece with shadow
        text_color = WHITE if self.color == PieceColor.WHITE else BLACK
        bg_color = (0, 0, 0, 100) if not selected else (0, 255, 0, 100)
        
        # Calculate size based on piece type (kings are largest)
        size = int(72 * (self.diameter / 40))
        piece_font = pygame.font.SysFont('segoeuisymbol', size)
        
        # Create the piece text and shadow
        piece_text = piece_font.render(self.symbol, True, text_color)
        text_rect = piece_text.get_rect(center=(x, y))
        
        # Draw selection highlight if selected
        if selected:
            pygame.draw.circle(surface, HIGHLIGHT, (int(x), int(y)), int(self.diameter * 0.7))
        
        # Draw the piece
        surface.blit(piece_text, text_rect)
        
        # Draw the path if active
        if len(self.path) > 1:
            points = [self.get_pixel_position_from_board_pos(pos) for pos in self.path]
            pygame.draw.lines(surface, GREEN if selected else BLUE, False, points, 2)
            
    def get_pixel_position_from_board_pos(self, board_pos):
        """Convert board position to pixel position"""
        col, row = board_pos
        return (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                row * SQUARE_SIZE + SQUARE_SIZE // 2)
    
    def reset_to_original_position(self):
        """Reset piece to its original position"""
        self.position = self.original_position.copy()
        self.velocity = np.array([0.0, 0.0])
        self.path = [self.position.copy()]

class CoilGrid:
    """Represents the 20x20 grid of electromagnetic coils"""
    def __init__(self, size=COIL_GRID_SIZE):
        self.size = size
        self.coil_power = np.zeros((size, size))  # Power level of each coil (0-100%)
        self.coil_current = np.zeros((size, size))  # Current direction (+/-)
        self.magnetic_field = np.zeros((size, size, 2))  # Magnetic field vector at each point
        
    def reset(self):
        """Reset all coils to zero power"""
        self.coil_power.fill(0)
        self.coil_current.fill(0)
        self.magnetic_field.fill(0)
        
    def update_coil(self, row, col, power, current_direction=1):
        """Update a single coil's power and current direction"""
        if 0 <= row < self.size and 0 <= col < self.size:
            self.coil_power[row, col] = np.clip(power, 0, 100)
            self.coil_current[row, col] = np.sign(current_direction)
            
    def activate_coil_pattern(self, pattern_type, position, target=None, intensity=100, radius=3):
        """
        Activate a pattern of coils around the given position
        
        Args:
            pattern_type: The type of pattern to activate
            position: Central position in coil coordinates (0-19, 0-19)
            target: Target position for directed patterns
            intensity: Maximum intensity (0-100)
            radius: Radius of influence in coil units
        """
        center_x, center_y = position
        
        # Convert to int indices for array access
        center_x_int = int(center_x)
        center_y_int = int(center_y)
        
        if pattern_type == "radial":
            # Radial pattern with decreasing intensity from center
            for r in range(max(0, center_y_int - radius), min(self.size, center_y_int + radius + 1)):
                for c in range(max(0, center_x_int - radius), min(self.size, center_x_int + radius + 1)):
                    # Calculate distance from center
                    distance = np.sqrt((r - center_y)**2 + (c - center_x)**2)
                    
                    if distance <= radius:
                        # Set power based on distance (decreasing outward)
                        power = intensity * (1 - distance / radius)
                        self.update_coil(r, c, power)
                        
        elif pattern_type == "directed" and target is not None:
            # Directed pattern toward target
            target_x, target_y = target
            
            # Calculate direction vector
            direction = np.array([target_x - center_x, target_y - center_y])
            distance_to_target = np.linalg.norm(direction)
            
            if distance_to_target > 0:
                direction = direction / distance_to_target
            
            for r in range(max(0, center_y_int - radius), min(self.size, center_y_int + radius + 1)):
                for c in range(max(0, center_x_int - radius), min(self.size, center_x_int + radius + 1)):
                    # Calculate position relative to center
                    rel_pos = np.array([c - center_x, r - center_y])
                    distance = np.linalg.norm(rel_pos)
                    
                    if distance <= radius:
                        # Project relative position onto direction vector
                        proj = np.dot(rel_pos, direction)
                        
                        # Calculate perpendicular distance to the line of movement
                        perp_distance = np.sqrt(max(0, distance**2 - proj**2))
                        
                        # Set power based on factors:
                        # 1. Distance from center (decreasing with distance)
                        # 2. Projection onto direction (positive values are ahead)
                        # 3. Perpendicular distance (closer to line is stronger)
                        
                        # Normalize projection to be 0-1 within the radius
                        norm_proj = (proj + radius) / (2 * radius)
                        norm_proj = np.clip(norm_proj, 0, 1)
                        
                        # Calculate power based on position relative to movement direction
                        if proj > 0:  # Ahead of the piece (pulling)
                            power = intensity * (1 - distance / radius) * (1 - perp_distance / radius) * norm_proj
                            # Negative current for attraction
                            self.update_coil(r, c, power, -1)
                        else:  # Behind the piece (pushing)
                            power = intensity * (1 - distance / radius) * (1 - perp_distance / radius) * (1 - norm_proj)
                            # Positive current for repulsion
                            self.update_coil(r, c, power, 1)
                            
        elif pattern_type == "knight":
            # Special pattern for knight's L-shaped moves
            target_x, target_y = target
            
            # Calculate direction vector to target
            direction = np.array([target_x - center_x, target_y - center_y])
            distance_to_target = np.linalg.norm(direction)
            
            if distance_to_target > 0:
                direction = direction / distance_to_target
            
            # Determine the L shape of the knight's move
            # Normalize the components to determine which direction is primary
            abs_dx = abs(target_x - center_x)
            abs_dy = abs(target_y - center_y)
            
            # Knight moves are typically 2 squares in one direction, 1 in another
            # Determine which leg of the L to prioritize
            if abs_dx > abs_dy:
                # Horizontal movement first (longer leg of L)
                mid1_x = center_x + direction[0] * min(abs_dx, radius * 1.5)
                mid1_y = center_y
                # Then vertical movement (shorter leg of L)
                mid2_x = mid1_x
                mid2_y = mid1_y + direction[1] * min(abs_dy, radius)
            else:
                # Vertical movement first (longer leg of L)
                mid1_x = center_x
                mid1_y = center_y + direction[1] * min(abs_dy, radius * 1.5)
                # Then horizontal movement (shorter leg of L)
                mid2_x = mid1_x + direction[0] * min(abs_dx, radius)
                mid2_y = mid1_y
            
            # Intermediate point for guidance
            mid_x = (mid1_x + mid2_x) / 2
            mid_y = (mid1_y + mid2_y) / 2
            
            # Apply coil activations along the projected path, focused on midpoints
            for r in range(max(0, center_y_int - radius*2), min(self.size, center_y_int + radius*2 + 1)):
                for c in range(max(0, center_x_int - radius*2), min(self.size, center_x_int + radius*2 + 1)):
                    # Calculate distances to different points along the path
                    dist_to_mid1 = np.sqrt((r - mid1_y)**2 + (c - mid1_x)**2)
                    dist_to_mid2 = np.sqrt((r - mid2_y)**2 + (c - mid2_x)**2)
                    dist_to_mid = np.sqrt((r - mid_y)**2 + (c - mid_x)**2)
                    dist_to_target = np.sqrt((r - target_y)**2 + (c - target_x)**2)
                    
                    # Find the minimum distance to any point on the path
                    min_dist = min(dist_to_mid1, dist_to_mid2, dist_to_mid, dist_to_target)
                    
                    # Activate coils that are close to any part of the path
                    if min_dist <= radius:
                        # Stronger attraction toward the path
                        power_factor = 1.0 - min_dist / radius
                        
                        # Determine if we're ahead or behind the piece in the knight's move path
                        # For simplicity, use a combination of distances to determine
                        # if we're in the "pull ahead" or "push from behind" region
                        if (c - center_x) * direction[0] + (r - center_y) * direction[1] > 0:
                            # We're ahead in the general movement direction - attract
                            power = intensity * power_factor * 1.2  # Boost attraction slightly
                            self.update_coil(r, c, power, -1)  # Negative for attraction
                        else:
                            # We're behind - repulse
                            power = intensity * power_factor * 0.8  # Slightly weaker repulsion
                            self.update_coil(r, c, power, 1)  # Positive for repulsion
            
            # Add extra attraction at key points
            key_points = [(mid1_x, mid1_y), (mid2_x, mid2_y), (target_x, target_y)]
            for point_x, point_y in key_points:
                point_x_int, point_y_int = int(point_x), int(point_y)
                for r in range(max(0, point_y_int - 1), min(self.size, point_y_int + 2)):
                    for c in range(max(0, point_x_int - 1), min(self.size, point_x_int + 2)):
                        # Distance to the key point
                        dist = np.sqrt((r - point_y)**2 + (c - point_x)**2)
                        if dist <= 1.5:  # Small radius around key points
                            # Strong attraction at key path points
                            power = intensity * (1 - dist/1.5) * 1.5  # Extra strong
                            self.update_coil(r, c, power, -1)  # Negative for attraction
                        
        else:
            # Default to simple radial pattern if unknown type
            for r in range(max(0, center_y_int - radius), min(self.size, center_y_int + radius + 1)):
                for c in range(max(0, center_x_int - radius), min(self.size, center_x_int + radius + 1)):
                    # Calculate distance from center
                    distance = np.sqrt((r - center_y)**2 + (c - center_x)**2)
                    
                    if distance <= radius:
                        # Set power based on distance (decreasing outward)
                        power = intensity * (1 - distance / radius)
                        self.update_coil(r, c, power)
                        
    def update_magnetic_field(self):
        """Calculate the magnetic field vector at each point based on coil activation"""
        # Reset the field
        self.magnetic_field.fill(0)
        
        # For each coil, calculate its contribution to the field
        for r in range(self.size):
            for c in range(self.size):
                power = self.coil_power[r, c]
                if power > 0:
                    # Direction determined by current
                    direction = self.coil_current[r, c]
                    
                    # Influence other points in the grid within a radius
                    radius = 5  # Field influence radius
                    
                    for r2 in range(max(0, r - radius), min(self.size, r + radius + 1)):
                        for c2 in range(max(0, c - radius), min(self.size, c + radius + 1)):
                            # Calculate distance
                            distance = np.sqrt((r2 - r)**2 + (c2 - c)**2)
                            
                            if distance > 0 and distance <= radius:
                                # Vector from coil to point
                                field_vector = np.array([c2 - c, r2 - r])
                                field_vector = field_vector / distance  # Normalize
                                
                                # Field strength decreases with distance squared
                                strength = power * (1 / (1 + distance**2))
                                
                                # Update field vector at point
                                # For attraction (negative current), field points toward the coil
                                # For repulsion (positive current), field points away from the coil
                                self.magnetic_field[r2, c2] += field_vector * strength * -direction
    
    def calculate_force(self, piece_position, piece_magnet_strength):
        """Calculate the force on a piece at the given position"""
        # Convert piece position to coil grid coordinates
        col, row = piece_position
        col_grid = col * (self.size / BOARD_SQUARES)
        row_grid = row * (self.size / BOARD_SQUARES)
        
        # Interpolate field at the piece position
        col_idx = int(col_grid)
        row_idx = int(row_grid)
        
        # Ensure indices are within bounds
        if not (0 <= col_idx < self.size - 1 and 0 <= row_idx < self.size - 1):
            return np.array([0.0, 0.0])
        
        # Get surrounding field vectors
        field_00 = self.magnetic_field[row_idx, col_idx]
        field_01 = self.magnetic_field[row_idx, col_idx + 1]
        field_10 = self.magnetic_field[row_idx + 1, col_idx]
        field_11 = self.magnetic_field[row_idx + 1, col_idx + 1]
        
        # Bilinear interpolation
        dx = col_grid - col_idx
        dy = row_grid - row_idx
        
        field_x = (1 - dy) * ((1 - dx) * field_00[0] + dx * field_01[0]) + \
                 dy * ((1 - dx) * field_10[0] + dx * field_11[0])
        field_y = (1 - dy) * ((1 - dx) * field_00[1] + dx * field_01[1]) + \
                 dy * ((1 - dx) * field_10[1] + dx * field_11[1])
                 
        field = np.array([field_x, field_y])
        
        # Force is proportional to field strength and piece magnet strength
        force = field * piece_magnet_strength * 10.0  # Scale factor for simulation
        
        return force
    
    def draw(self, surface):
        """Draw the coil grid on the given surface"""
        coil_size = BOARD_SIZE / self.size
        
        # Create a surface for coils with transparency
        coil_surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
        
        # Draw each coil
        for r in range(self.size):
            for c in range(self.size):
                # Calculate position
                x = c * coil_size + coil_size / 2
                y = r * coil_size + coil_size / 2
                
                # Get coil power and current
                power = self.coil_power[r, c]
                current = self.coil_current[r, c]
                
                if power > 0:
                    # Determine color based on current direction (red for repulsion, blue for attraction)
                    if current > 0:  # Repulsion
                        color = (255, 100, 100, int(power * 2.55))  # Red with alpha based on power
                    else:  # Attraction
                        color = (100, 100, 255, int(power * 2.55))  # Blue with alpha based on power
                    
                    # Draw the active coil
                    radius = int(coil_size/2 * 0.8 * (0.5 + 0.5 * power/100))  # Size varies with power
                    pygame.draw.circle(coil_surface, color, (int(x), int(y)), radius)
                
                # Draw the coil outline
                pygame.draw.circle(coil_surface, DARK_GRAY, (int(x), int(y)), int(coil_size/2 * 0.8), 1)
        
        # Blit the coil surface onto the main surface
        surface.blit(coil_surface, (0, 0))
        
    def draw_field_overlay(self, surface, resolution=24):
        """Draw the magnetic field vectors as an overlay"""
        step_size = BOARD_SIZE / resolution
        
        # Create a surface for field vectors with transparency
        field_surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
        
        # Draw field vectors
        for r in range(resolution):
            for c in range(resolution):
                # Calculate position in pixels
                x = c * step_size + step_size / 2
                y = r * step_size + step_size / 2
                
                # Calculate position in grid coordinates
                grid_c = c * (self.size / resolution)
                grid_r = r * (self.size / resolution)
                
                # Interpolate field at this position
                grid_c_idx = int(grid_c)
                grid_r_idx = int(grid_r)
                
                # Ensure indices are within bounds
                if 0 <= grid_c_idx < self.size - 1 and 0 <= grid_r_idx < self.size - 1:
                    # Get surrounding field vectors
                    field_00 = self.magnetic_field[grid_r_idx, grid_c_idx]
                    field_01 = self.magnetic_field[grid_r_idx, grid_c_idx + 1]
                    field_10 = self.magnetic_field[grid_r_idx + 1, grid_c_idx]
                    field_11 = self.magnetic_field[grid_r_idx + 1, grid_c_idx + 1]
                    
                    # Bilinear interpolation
                    dx = grid_c - grid_c_idx
                    dy = grid_r - grid_r_idx
                    
                    field_x = (1 - dy) * ((1 - dx) * field_00[0] + dx * field_01[0]) + \
                             dy * ((1 - dx) * field_10[0] + dx * field_11[0])
                    field_y = (1 - dy) * ((1 - dx) * field_00[1] + dx * field_01[1]) + \
                             dy * ((1 - dx) * field_10[1] + dx * field_11[1])
                    
                    field = np.array([field_x, field_y])
                    field_strength = np.linalg.norm(field)
                    
                    # Draw arrow if field is strong enough
                    if field_strength > 0.05:
                        # Normalize and scale
                        max_arrow_length = step_size * 0.8
                        arrow_length = min(field_strength * 5, max_arrow_length)
                        if arrow_length > 0:
                            field_normalized = field / field_strength
                            
                            # Calculate arrow end point
                            end_x = x + field_normalized[0] * arrow_length
                            end_y = y + field_normalized[1] * arrow_length
                            
                            # Draw arrow line
                            alpha = min(255, int(field_strength * 255 * 5))
                            arrow_color = (255, 255, 255, alpha)
                            pygame.draw.line(field_surface, arrow_color, (int(x), int(y)), 
                                            (int(end_x), int(end_y)), 2)
                            
                            # Draw arrow head
                            head_length = min(6, arrow_length / 3)
                            if head_length > 1:
                                angle = math.atan2(field_normalized[1], field_normalized[0])
                                head1_x = end_x - head_length * math.cos(angle + math.pi/6)
                                head1_y = end_y - head_length * math.sin(angle + math.pi/6)
                                head2_x = end_x - head_length * math.cos(angle - math.pi/6)
                                head2_y = end_y - head_length * math.sin(angle - math.pi/6)
                                
                                pygame.draw.line(field_surface, arrow_color, 
                                                (int(end_x), int(end_y)), 
                                                (int(head1_x), int(head1_y)), 2)
                                pygame.draw.line(field_surface, arrow_color, 
                                                (int(end_x), int(end_y)), 
                                                (int(head2_x), int(head2_y)), 2)
        
        # Blit the field surface onto the main surface
        surface.blit(field_surface, (0, 0))
    
    def generate_heatmap(self, resolution=400):
        """Generate a heatmap of the field strength for plotting"""
        # Create a high-resolution grid
        heatmap = np.zeros((resolution, resolution))
        
        # Calculate field strength at each point
        for r in range(resolution):
            for c in range(resolution):
                # Convert to grid coordinates
                grid_c = c * (self.size / resolution)
                grid_r = r * (self.size / resolution)
                
                # Interpolate field at this position
                grid_c_idx = int(grid_c)
                grid_r_idx = int(grid_r)
                
                # Ensure indices are within bounds
                if 0 <= grid_c_idx < self.size - 1 and 0 <= grid_r_idx < self.size - 1:
                    # Get surrounding field vectors
                    field_00 = self.magnetic_field[grid_r_idx, grid_c_idx]
                    field_01 = self.magnetic_field[grid_r_idx, grid_c_idx + 1]
                    field_10 = self.magnetic_field[grid_r_idx + 1, grid_c_idx]
                    field_11 = self.magnetic_field[grid_r_idx + 1, grid_c_idx + 1]
                    
                    # Bilinear interpolation
                    dx = grid_c - grid_c_idx
                    dy = grid_r - grid_r_idx
                    
                    field_x = (1 - dy) * ((1 - dx) * field_00[0] + dx * field_01[0]) + \
                             dy * ((1 - dx) * field_10[0] + dx * field_11[0])
                    field_y = (1 - dy) * ((1 - dx) * field_00[1] + dx * field_01[1]) + \
                             dy * ((1 - dx) * field_10[1] + dx * field_11[1])
                    
                    # Field strength is the magnitude of the field vector
                    field_strength = np.sqrt(field_x**2 + field_y**2)
                    heatmap[r, c] = field_strength
        
        # Smooth the heatmap slightly
        heatmap = gaussian_filter(heatmap, sigma=1.0)
        
        return heatmap
    
    def plot_heatmap(self, figsize=(10, 10)):
        """Plot the magnetic field strength as a heatmap"""
        heatmap = self.generate_heatmap()
        
        # Create a figure
        plt.figure(figsize=figsize)
        
        # Define custom colormap (blue to white to red)
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list("field_cmap", colors, N=256)
        
        # Plot the heatmap
        plt.imshow(heatmap, cmap=cmap, aspect='equal', origin='upper')
        plt.colorbar(label='Field Strength')
        plt.title('Magnetic Field Strength')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig("field_heatmap.png", dpi=300)
        plt.close()
        
        return "field_heatmap.png"

class ChessBoard:
    """Represents the chessboard and game state"""
    def __init__(self):
        # Initialize the pygame window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Electromagnetic Chess Coil Simulation")
        
        # Initialize fonts
        self.font = pygame.font.SysFont('segoeui', 24)
        self.small_font = pygame.font.SysFont('segoeui', 16)
        
        # Initialize chessboard state
        self.pieces = []
        self.initialize_pieces()
        
        # Initialize coil grid
        self.coil_grid = CoilGrid()
        
        # Initialize UI state
        self.selected_piece = None
        self.target_position = None
        self.move_in_progress = False
        self.move_timer = 0
        self.move_complete = False
        self.captured_piece = None
        self.capture_step = 0
        self.capture_complete = False
        self.temporarily_moved_pieces = []  # Track pieces that were moved to clear a path
        
        # Visualization options
        self.show_coils = True
        self.show_field = False
        self.show_forces = False
        self.show_paths = True
        
        # Move pattern options
        self.current_pattern = "directed"
        self.patterns = ["directed", "radial", "knight"]
        
        # Simulation parameters
        self.simulation_speed = 1.0
        self.field_update_timer = 0
        
        # Create an offboard position for captured pieces
        self.capture_area = (9, 4)  # In board coordinates, just off the right edge
        
        # Clock for controlling FPS
        self.clock = pygame.time.Clock()
        
    def initialize_pieces(self):
        """Initialize the chess pieces in their starting positions"""
        # Clear existing pieces
        self.pieces = []
        
        # Add pawns
        for col in range(8):
            self.pieces.append(ChessPiece(PieceColor.WHITE, PieceType.PAWN, (col, 6)))
            self.pieces.append(ChessPiece(PieceColor.BLACK, PieceType.PAWN, (col, 1)))
        
        # Add major pieces
        piece_types = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN, 
                      PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]
        
        for col, piece_type in enumerate(piece_types):
            self.pieces.append(ChessPiece(PieceColor.WHITE, piece_type, (col, 7)))
            self.pieces.append(ChessPiece(PieceColor.BLACK, piece_type, (col, 0)))
            
    def get_piece_at_position(self, position):
        """Get piece at the given board position"""
        row, col = position
        # Find all pieces near the clicked position
        nearby_pieces = []
        for piece in self.pieces:
            if piece.active:  # Only check active pieces
                piece_row, piece_col = piece.get_board_position()
                # Use a larger tolerance to make selection easier
                distance = math.sqrt((piece_row - row)**2 + (piece_col - col)**2)
                if distance < 0.7:  # Increased from 0.5 to make selection easier
                    nearby_pieces.append((piece, distance))
        
        # If we found pieces, return the closest one
        if nearby_pieces:
            # Sort by distance and return the closest
            return min(nearby_pieces, key=lambda x: x[1])[0]
        return None
    
    def is_position_occupied(self, position, ignore_piece=None):
        """Check if the given position is occupied by a piece"""
        row, col = position
        for piece in self.pieces:
            if piece != ignore_piece and piece.active:
                piece_row, piece_col = piece.get_board_position()
                # Use a small tolerance
                distance = math.sqrt((piece_row - row)**2 + (piece_col - col)**2)
                if distance < 0.5:
                    return True
        return False
    
    def handle_click(self, pos):
        """Handle mouse click events"""
        # Convert pixel position to board coordinates
        col = pos[0] // SQUARE_SIZE
        row = pos[1] // SQUARE_SIZE
        
        # Ignore clicks outside the board
        if not (0 <= col < 8 and 0 <= row < 8):
            return
        
        # If no piece is selected, select the piece at the clicked position
        if not self.selected_piece:
            piece = self.get_piece_at_position((row, col))
            if piece:
                self.selected_piece = piece
                print(f"Selected {piece.piece_type.value} at {col}, {row}")
                # Highlight valid move squares
                self.calculate_valid_moves()
                
        # If a piece is already selected, set the target position
        elif not self.move_in_progress and not self.move_complete:
            # Convert to float coords for smoother movement
            target = (float(col), float(row))
            
            # Check if the move is valid (not occupied by same color piece)
            target_piece = self.get_piece_at_position((row, col))
            
            if target_piece and target_piece.color == self.selected_piece.color:
                # Cannot move to a square occupied by same color piece
                print(f"Cannot move to square occupied by {target_piece.piece_type.value} of same color")
                # Select this piece instead
                self.selected_piece = target_piece
                self.calculate_valid_moves()
                return
            
            # Check if path is clear (for non-knight pieces)
            if self.selected_piece.piece_type != PieceType.KNIGHT:
                if not self.is_path_clear(self.selected_piece.get_board_position(), target):
                    print("Path is blocked by other pieces")
                    # Try to clear the path if needed
                    if self.attempt_to_clear_path(self.selected_piece.get_board_position(), target):
                        print("Path cleared, proceeding with move")
                    else:
                        print("Unable to clear path")
                        return
            else:
                # For knights, check if landing square is clear or occupied by opponent
                if target_piece and target_piece.color == self.selected_piece.color:
                    print("Cannot move to square occupied by same color piece")
                    return
            
            # Set the target position
            self.target_position = target
            print(f"Set target to {col}, {row}")
            
            # Handle capture if there's a piece at the target
            if target_piece and target_piece != self.selected_piece:
                # Capture the piece
                self.captured_piece = target_piece
                self.captured_piece.active = False
                self.captured_piece.update_capture_path(self.capture_area)
                print(f"Capturing {target_piece.piece_type.value}")
            
            # Start the move
            self.start_move()
    
    def calculate_valid_moves(self):
        """Calculate and highlight valid moves for the selected piece"""
        # This would be implemented based on chess rules
        # For now, just print a message
        if self.selected_piece:
            print(f"Valid moves for {self.selected_piece.piece_type.value} would be highlighted here")
    
    def is_path_clear(self, start_pos, end_pos):
        """Check if the path between start and end is clear of pieces"""
        start_col, start_row = start_pos
        end_col, end_row = end_pos
        
        # Calculate direction
        delta_col = end_col - start_col
        delta_row = end_row - start_row
        
        # Normalize to get direction vector
        steps = max(abs(delta_col), abs(delta_row))
        if steps == 0:
            return True  # Same position
            
        dir_col = delta_col / steps
        dir_row = delta_row / steps
        
        # Check each position along the path (excluding start and end)
        for i in range(1, int(steps)):
            check_col = start_col + dir_col * i
            check_row = start_row + dir_row * i
            
            # Round to nearest position for checking
            col_int, row_int = round(check_col), round(check_row)
            
            # Check if any piece is at this position
            for piece in self.pieces:
                if not piece.active or piece == self.selected_piece:
                    continue
                    
                piece_col, piece_row = piece.get_board_position()
                piece_col_int, piece_row_int = round(piece_col), round(piece_row)
                
                if piece_col_int == col_int and piece_row_int == row_int:
                    print(f"Path blocked by {piece.piece_type.value} at {col_int}, {row_int}")
                    return False
                    
        return True
        
    def attempt_to_clear_path(self, start_pos, end_pos):
        """Try to move pieces out of the way to clear a path"""
        # This is a simplified version - in a real implementation,
        # you would use more sophisticated path planning
        
        # For now, just identify pieces in the way
        blocking_pieces = []
        
        start_col, start_row = start_pos
        end_col, end_row = end_pos
        
        # Calculate direction
        delta_col = end_col - start_col
        delta_row = end_row - start_row
        
        # Normalize to get direction vector
        steps = max(abs(delta_col), abs(delta_row))
        if steps == 0:
            return True  # Same position
            
        dir_col = delta_col / steps
        dir_row = delta_row / steps
        
        # Find blocking pieces
        for i in range(1, int(steps)):
            check_col = start_col + dir_col * i
            check_row = start_row + dir_row * i
            
            # Round to nearest position for checking
            col_int, row_int = round(check_col), round(check_row)
            
            # Check if any piece is at this position
            for piece in self.pieces:
                if not piece.active or piece == self.selected_piece:
                    continue
                    
                piece_col, piece_row = piece.get_board_position()
                piece_col_int, piece_row_int = round(piece_col), round(piece_row)
                
                if piece_col_int == col_int and piece_row_int == row_int:
                    blocking_pieces.append(piece)
        
        # If there are pawns blocking the way, try to move them aside
        for piece in blocking_pieces:
            if piece.piece_type == PieceType.PAWN:
                # Determine which side to move (perpendicular to the path)
                perpendicular_col = -dir_row  # Perpendicular direction
                perpendicular_row = dir_col
                
                # Check both sides to see if we can move the pawn
                for side_factor in [0.5, -0.5]:  # Try both sides, but less movement
                    new_col = piece.position[0] + perpendicular_col * side_factor
                    new_row = piece.position[1] + perpendicular_row * side_factor
                    
                    # Ensure the new position is on the board
                    if 0 <= new_col < 8 and 0 <= new_row < 8:
                        # Check if the new position is empty
                        if not self.is_position_occupied((new_row, new_col), piece):
                            # Move the pawn to clear the path
                            print(f"Moving {piece.piece_type.value} aside to clear path")
                            piece.position = np.array([new_col, new_row])
                            # Remember this piece was moved temporarily
                            self.temporarily_moved_pieces.append(piece)
                            return True
        
        # If we couldn't clear the path
        return False
    
    def start_move(self):
        """Start moving the selected piece to the target position"""
        if self.selected_piece and self.target_position:
            self.move_in_progress = True
            self.move_timer = 0
            self.move_complete = False
            self.capture_step = 0
            self.capture_complete = False
            
            # Clear the coil grid
            self.coil_grid.reset()
            
            print(f"Starting move from {self.selected_piece.get_board_position()} to {self.target_position}")
    
    def update_move(self, dt):
        """Update the current move in progress"""
        if self.move_in_progress and self.selected_piece and self.target_position:
            # Update move timer
            self.move_timer += dt * self.simulation_speed
            
            # Get current position and target
            current_pos = self.selected_piece.get_board_position()
            target_pos = self.target_position
            
            # Calculate coil positions
            current_coil_pos = self.selected_piece.get_coil_position()
            target_coil_pos = (
                target_pos[0] * (COIL_GRID_SIZE / BOARD_SQUARES),
                target_pos[1] * (COIL_GRID_SIZE / BOARD_SQUARES)
            )
            
            # Activate coils based on current pattern
            intensity = 100  # Maximum intensity
            
            # Reset coils periodically to simulate pulsing
            self.field_update_timer += dt
            if self.field_update_timer >= 0.1:  # Update every 100ms
                self.field_update_timer = 0
                self.coil_grid.reset()
                
                # Determine move type based on piece type and target
                dx = target_pos[0] - current_pos[0]
                dy = target_pos[1] - current_pos[1]
                
                # Calculate Euclidean distance for path progress
                total_distance = math.sqrt(dx**2 + dy**2)
                current_distance = math.sqrt(
                    (current_pos[0] - target_pos[0])**2 + 
                    (current_pos[1] - target_pos[1])**2
                )
                progress = max(0, min(100, 100 * (1 - current_distance / total_distance)))
                
                # Display progress in percentage
                if self.move_timer > 0.5:  # Wait a bit before showing progress
                    print(f"Move progress: {progress:.1f}%", end="\r")
                
                # Determine pattern based on move type
                if self.selected_piece.piece_type == PieceType.KNIGHT and abs(dx) + abs(dy) >= 3:
                    pattern = "knight"
                elif abs(dx) == abs(dy) and abs(dx) > 0:
                    pattern = "directed"  # Diagonal move
                elif abs(dx) == 0 or abs(dy) == 0:
                    pattern = "directed"  # Straight move
                else:
                    pattern = self.current_pattern
                
                # For knight moves, check if we need to handle intermediate obstacles
                if pattern == "knight":
                    # Define the L-shaped path: first move horizontally/vertically, then make the turn
                    # Currently we're using the knight pattern which handles this automatically,
                    # but we could add more specific path planning here if needed
                    
                    # Calculate midpoint for knight's move (2/3 of the way through the L shape)
                    if abs(dx) > abs(dy):  # Move horizontally first
                        midpoint_col = current_pos[0] + 2*dx/3
                        midpoint_row = current_pos[1]
                    else:  # Move vertically first
                        midpoint_col = current_pos[0]
                        midpoint_row = current_pos[1] + 2*dy/3
                    
                    midpoint_coil_pos = (
                        midpoint_col * (COIL_GRID_SIZE / BOARD_SQUARES),
                        midpoint_row * (COIL_GRID_SIZE / BOARD_SQUARES)
                    )
                    
                    # If we're less than halfway through the move, target the midpoint
                    # Otherwise, target the final destination
                    if progress < 50:
                        # Use knight pattern but with intermediate target
                        self.coil_grid.activate_coil_pattern(
                            "knight", current_coil_pos, midpoint_coil_pos, intensity, radius=4)
                    else:
                        # Complete the move to the final target
                        self.coil_grid.activate_coil_pattern(
                            "directed", current_coil_pos, target_coil_pos, intensity, radius=4)
                else:
                    # Use the selected pattern for non-knight moves
                    if pattern == "directed":
                        self.coil_grid.activate_coil_pattern(
                            "directed", current_coil_pos, target_coil_pos, intensity, radius=4)
                    else:
                        self.coil_grid.activate_coil_pattern(
                            pattern, current_coil_pos, target_coil_pos, intensity, radius=4)
                
                # Update the magnetic field
                self.coil_grid.update_magnetic_field()
            
            # Calculate force on the piece
            force = self.coil_grid.calculate_force(
                self.selected_piece.get_board_position(),
                self.selected_piece.magnet_strength)
            
            # Update piece position based on force
            self.selected_piece.apply_force(force, dt * self.simulation_speed)
            
            # Check if move is complete
            current_pos = self.selected_piece.get_board_position()
            distance_to_target = np.linalg.norm(
                np.array(current_pos) - np.array(self.target_position))
            
            if distance_to_target < 0.1:
                # Snap to target position
                self.selected_piece.position = np.array(self.target_position)
                self.move_in_progress = False
                self.move_complete = True
                self.coil_grid.reset()
                print("\nMove complete!") 
                
                # Return any temporarily moved pieces to their original positions
                for piece in self.temporarily_moved_pieces:
                    print(f"Returning {piece.piece_type.value} to original position")
                    piece.reset_to_original_position()
                self.temporarily_moved_pieces = []
            
            # Handle captured piece movement
            if self.captured_piece and not self.capture_complete:
                # Move captured piece along its path
                if self.capture_step < len(self.captured_piece.capture_path):
                    self.captured_piece.follow_capture_path(self.capture_step)
                    self.capture_step += self.simulation_speed
                else:
                    self.capture_complete = True
                    print("Capture complete!")
    
    def draw_board(self):
        """Draw the chessboard"""
        # Draw background
        self.screen.fill(DARK_GRAY)
        
        # Draw squares
        for row in range(8):
            for col in range(8):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, 
                                (col * SQUARE_SIZE, row * SQUARE_SIZE, 
                                 SQUARE_SIZE, SQUARE_SIZE))
                                 
        # Draw rank and file indicators
        for i in range(8):
            # Draw file indicators (a-h) along the bottom
            file_text = self.small_font.render(chr(97 + i), True, WHITE)
            file_rect = file_text.get_rect(center=(i * SQUARE_SIZE + SQUARE_SIZE // 2, 
                                                  BOARD_SIZE + 20))
            self.screen.blit(file_text, file_rect)
            
            # Draw rank indicators (1-8) along the right side
            rank_text = self.small_font.render(str(8 - i), True, WHITE)
            rank_rect = rank_text.get_rect(center=(BOARD_SIZE + 20, 
                                                  i * SQUARE_SIZE + SQUARE_SIZE // 2))
            self.screen.blit(rank_text, rank_rect)
        
    def draw_pieces(self):
        """Draw all chess pieces"""
        # Sort pieces so selected piece is drawn last (on top)
        sorted_pieces = sorted(self.pieces, 
                              key=lambda p: p == self.selected_piece or not p.active)
        
        for piece in sorted_pieces:
            piece.draw(self.screen, self.font, piece == self.selected_piece)
    
    def draw_controls(self):
        """Draw control panel and information"""
        # Draw panel background
        pygame.draw.rect(self.screen, LIGHT_GRAY, 
                        (BOARD_SIZE, 0, WINDOW_WIDTH - BOARD_SIZE, BOARD_SIZE))
        
        # Draw title
        title_text = self.font.render("Electromagnetic Chess Coil Simulation", True, BLACK)
        self.screen.blit(title_text, (BOARD_SIZE + 20, 20))
        
        # Draw control instructions
        help_y = 60
        help_line_height = 25
        
        help_lines = [
            "Click a piece to select it",
            "Click again to set target position",
            "Press R to reset the board",
            "Press C to toggle coil visualization",
            "Press F to toggle field visualization",
            "Press V to toggle force vectors",
            "Press P to toggle path visualization",
            f"Current pattern: {self.current_pattern.upper()}",
            "Press M to cycle movement patterns",
            f"Simulation speed: {self.simulation_speed:.1f}x",
            "Press + or - to adjust speed"
        ]
        
        for i, line in enumerate(help_lines):
            text = self.small_font.render(line, True, BLACK)
            self.screen.blit(text, (BOARD_SIZE + 20, help_y + i * help_line_height))
        
        # Draw status information if a move is in progress
        if self.selected_piece and self.target_position:
            status_y = help_y + len(help_lines) * help_line_height + 20
            
            # Show selected piece info
            piece_info = f"Selected: {self.selected_piece.symbol} ({self.selected_piece.piece_type.value})"
            text = self.small_font.render(piece_info, True, BLUE)
            self.screen.blit(text, (BOARD_SIZE + 20, status_y))
            
            # Show target position
            target_col, target_row = self.target_position
            target_info = f"Target: ({int(target_col)}, {int(target_row)})"
            text = self.small_font.render(target_info, True, GREEN)
            self.screen.blit(text, (BOARD_SIZE + 20, status_y + 25))
            
            # Show move status
            if self.move_in_progress:
                # Calculate progress
                current_pos = self.selected_piece.get_board_position()
                distance_to_target = np.linalg.norm(
                    np.array(current_pos) - np.array(self.target_position))
                total_distance = np.linalg.norm(
                    np.array(self.selected_piece.path[0]) - np.array(self.target_position))
                progress = min(100, int((1 - distance_to_target / total_distance) * 100))
                
                status = f"Move in progress: {progress}%"
                text = self.small_font.render(status, True, ORANGE)
                self.screen.blit(text, (BOARD_SIZE + 20, status_y + 50))
            elif self.move_complete:
                status = "Move complete!"
                text = self.small_font.render(status, True, GREEN)
                self.screen.blit(text, (BOARD_SIZE + 20, status_y + 50))
        
        # Draw visualization toggles
        vis_y = BOARD_SIZE - 150
        vis_lines = [
            f"Coils: {'ON' if self.show_coils else 'OFF'}",
            f"Field: {'ON' if self.show_field else 'OFF'}",
            f"Forces: {'ON' if self.show_forces else 'OFF'}",
            f"Paths: {'ON' if self.show_paths else 'OFF'}"
        ]
        
        for i, line in enumerate(vis_lines):
            text = self.small_font.render(line, True, BLACK)
            self.screen.blit(text, (BOARD_SIZE + 20, vis_y + i * 25))
        
        # Draw current pattern info
        pattern_y = BOARD_SIZE - 40
        pattern_text = f"Pattern: {self.current_pattern.upper()}"
        text = self.small_font.render(pattern_text, True, RED)
        self.screen.blit(text, (BOARD_SIZE + 20, pattern_y))
        
    def draw_field_heatmap(self):
        """Draw the magnetic field heatmap in the bottom area"""
        heatmap_path = self.coil_grid.plot_heatmap()
        
        # Load the heatmap image
        try:
            heatmap_surface = pygame.image.load(heatmap_path)
            heatmap_rect = heatmap_surface.get_rect()
            
            # Scale to fit the available space
            available_width = WINDOW_WIDTH - 40
            available_height = WINDOW_HEIGHT - BOARD_SIZE - 40
            
            scale_factor = min(available_width / heatmap_rect.width, 
                              available_height / heatmap_rect.height)
            
            scaled_width = int(heatmap_rect.width * scale_factor)
            scaled_height = int(heatmap_rect.height * scale_factor)
            
            heatmap_surface = pygame.transform.scale(heatmap_surface, 
                                                   (scaled_width, scaled_height))
            
            # Position at the bottom of the window
            heatmap_x = (WINDOW_WIDTH - scaled_width) // 2
            heatmap_y = BOARD_SIZE + 20
            
            # Draw the heatmap
            self.screen.blit(heatmap_surface, (heatmap_x, heatmap_y))
            
            # Add title
            text = self.font.render("Magnetic Field Strength Heatmap", True, WHITE)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, BOARD_SIZE + 10))
            self.screen.blit(text, text_rect)
            
        except Exception as e:
            print(f"Error loading heatmap: {e}")
    
    def reset(self):
        """Reset the board to initial state"""
        self.initialize_pieces()
        self.selected_piece = None
        self.target_position = None
        self.move_in_progress = False
        self.move_timer = 0
        self.move_complete = False
        self.captured_piece = None
        self.capture_step = 0
        self.capture_complete = False
        self.temporarily_moved_pieces = []
        self.coil_grid.reset()
        print("Board reset")
    
    def cycle_pattern(self):
        """Cycle through available coil activation patterns"""
        current_index = self.patterns.index(self.current_pattern)
        next_index = (current_index + 1) % len(self.patterns)
        self.current_pattern = self.patterns[next_index]
        print(f"Switched to pattern: {self.current_pattern}")
    
    def run(self):
        """Main game loop"""
        running = True
        last_time = time.time()
        
        while running:
            # Calculate delta time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_c:
                        self.show_coils = not self.show_coils
                    elif event.key == pygame.K_f:
                        self.show_field = not self.show_field
                    elif event.key == pygame.K_v:
                        self.show_forces = not self.show_forces
                    elif event.key == pygame.K_p:
                        self.show_paths = not self.show_paths
                    elif event.key == pygame.K_m:
                        self.cycle_pattern()
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.simulation_speed = min(5.0, self.simulation_speed + 0.1)
                    elif event.key == pygame.K_MINUS:
                        self.simulation_speed = max(0.1, self.simulation_speed - 0.1)
            
            # Update game state
            if self.move_in_progress:
                self.update_move(dt)
            
            # Draw everything
            self.draw_board()
            
            # Draw coils if enabled
            if self.show_coils:
                self.coil_grid.draw(self.screen)
            
            # Draw field vectors if enabled
            if self.show_field:
                self.coil_grid.draw_field_overlay(self.screen)
            
            # Draw pieces
            self.draw_pieces()
            
            # Draw controls
            self.draw_controls()
            
            # Update display
            pygame.display.flip()
            
            # Cap the frame rate
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

# Main function
def main():
    """Main function to run the simulation"""
    # Initialize pygame and start the simulation
    board = ChessBoard()
    board.run()

if __name__ == "__main__":
    main()