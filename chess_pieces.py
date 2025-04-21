# chess_pieces.py

import numpy as np
import pygame # Needed for color constants potentially shared
import math
from enum import Enum

# --- Constants ---
# Colors (can be defined here or imported from visualization if preferred)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
HIGHLIGHT = (124, 252, 0)

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
    """Represents a chess piece with physical and magnetic properties. Does NOT handle drawing."""
    def __init__(self, color: PieceColor, piece_type: PieceType, position,
                 board_squares=8, square_size=100, coil_grid_size=20): # Added size args
        self.color = color
        self.piece_type = piece_type
        self.position = np.array(position, dtype=float)  # (col, row) in board coordinates
        self.velocity = np.array([0.0, 0.0])

        # Store size info needed for conversions
        self.board_squares = board_squares
        self.square_size = square_size
        self.coil_grid_size = coil_grid_size

        # Load properties based on piece type
        self.properties = PIECE_PROPERTIES[piece_type]
        self.diameter = self.properties['diameter']
        self.height = self.properties['height']
        self.magnet_strength = self.properties['magnet_strength']
        self.symbol = PIECE_SYMBOLS[(color, piece_type)]

        self.path = [self.position.copy()]  # Store path history, start with initial position
        self.active = True  # Whether the piece is active (not captured)
        self.capture_path = []  # Path to follow when captured (list of board coords tuples)
        self.original_position = np.array(position, dtype=float)  # Store initial position
        self.position_before_temp_move = None # Store position before being moved aside

    def get_board_position(self):
        """Get position in board coordinates (col, row float)"""
        return tuple(self.position)

    # In chess_pieces.py - replace the get_pixel_position method with this:
    
    def get_pixel_position(self):
        """
        Get position in pixel coordinates for rendering - FIXED VERSION
        """
        col, row = self.position
        return (int(col * self.square_size), int(row * self.square_size))

    def get_coil_position(self):
        """Get position in coil grid coordinates (col, row float)"""
        col, row = self.position
        return (col * (self.coil_grid_size / self.board_squares),
                row * (self.coil_grid_size / self.board_squares))

    def apply_force(self, force, dt):
        """Apply external force to the piece and update its position"""
        mass = (self.diameter ** 2) * self.height * 0.0001
        if mass <= 0: mass = 0.01
        acceleration = force / mass
        max_accel = 500.0
        accel_mag = np.linalg.norm(acceleration)
        if accel_mag > max_accel: acceleration = acceleration * (max_accel / accel_mag)
        self.velocity += acceleration * dt # Direct velocity update
        max_velocity = 8.0
        vel_mag = np.linalg.norm(self.velocity)
        if vel_mag > max_velocity: self.velocity = self.velocity * (max_velocity / vel_mag)
        new_position = self.position + self.velocity * dt
        margin = 0.1
        new_position[0] = np.clip(new_position[0], -margin, self.board_squares-1+margin)
        new_position[1] = np.clip(new_position[1], -margin, self.board_squares-1+margin)
        self.position = new_position
        self.path.append(self.position.copy())
        if len(self.path) > 100: self.path.pop(0)

    def follow_capture_path(self, step_index):
        """Follow the capture path by moving towards the step_index node"""
        if self.capture_path and step_index < len(self.capture_path):
            # Target is the center of the square represented by the node
            target_pos = np.array(self.capture_path[step_index], dtype=float)
            move_vector = target_pos - self.position
            dist = np.linalg.norm(move_vector)

            # Move fractionally towards the target node each update
            move_fraction = 0.2 # Adjust for desired smoothness/speed
            if dist > 0.05: # Move only if not already very close
                self.position += move_vector * move_fraction
            else:
                # Snap to target node if close enough, ready for next step
                self.position = target_pos
                # Return True indicating node reached, potentially ready for next index
                return True # Let caller increment step_index

            self.path.append(self.position.copy())
            if len(self.path) > 100: self.path.pop(0)
            return False # Still moving towards current node
        return True # Path finished or empty

    def reset_to_original_position(self):
        """Reset piece to its original starting position"""
        self.position = self.original_position.copy()
        self.velocity = np.array([0.0, 0.0])
        self.path = [self.position.copy()]
        self.active = True
        self.capture_path = []
        self.position_before_temp_move = None

    def return_from_temporary_move(self):
        """Return piece from a temporary move aside position"""
        if self.position_before_temp_move is not None:
            self.position = self.position_before_temp_move.copy()
            self.velocity = np.array([0.0, 0.0])
            self.path = [self.position.copy()]
            self.position_before_temp_move = None
            print(f"Returned {self.symbol} from temporary move.")
        else:
            print(f"Warning: No temporary move position recorded for {self.symbol}.")