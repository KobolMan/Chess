import pygame
import sys
import time
import math
import heapq  # For priority queue in A* algorithm
from hardware_controller import ElectromagnetGridController  # Import the hardware controller

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 440  # mm, matches specification
DISPLAY_SCALE = 1.8  # Scale factor to make board larger on screen
SCALED_BOARD_SIZE = int(BOARD_SIZE * DISPLAY_SCALE)
SQUARE_SIZE = SCALED_BOARD_SIZE // 8  # Scaled square size
FPS = 60

# Piece size specifications (mm)
PIECE_DIAMETERS = {
    'k': 40,  # King
    'q': 38,  # Queen
    'b': 35,  # Bishop
    'n': 35,  # Knight
    'r': 33,  # Rook
    'p': 29   # Pawn
}

# Convert to scaled display size
SCALED_PIECE_DIAMETERS = {piece: int(diameter * DISPLAY_SCALE * 0.8) for piece, diameter in PIECE_DIAMETERS.items()}

# Coil grid settings
COIL_GRID_SIZE = 20  # 20x20 grid of coils
COIL_SPACING = SCALED_BOARD_SIZE / COIL_GRID_SIZE  # Distance between coil centers
COIL_DIAMETER = int(COIL_SPACING * 0.9)  # Slightly smaller than spacing to allow gaps
COIL_RADIUS = COIL_DIAMETER // 2

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (124, 252, 0)  # Highlight color for selected pieces
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
COIL_INACTIVE = (100, 100, 100, 50)  # Transparent gray
COIL_ACTIVE = (255, 100, 0, 150)  # Transparent orange
COIL_BORDER = (50, 50, 50)

# Piece names and mappings
PIECE_NAMES = {
    'wp': 'Pawn_W', 'wr': 'Rook_W', 'wn': 'Knight_W', 'wb': 'Bishop_W', 'wq': 'Queen_W', 'wk': 'King_W',
    'bp': 'Pawn_B', 'br': 'Rook_B', 'bn': 'Knight_B', 'bb': 'Bishop_B', 'bq': 'Queen_B', 'bk': 'King_B'
}

# Load chess piece images
def load_pieces():
    pieces = {}
    
    # In a real implementation, you would load actual PNG images
    # This simulates that with simple text representation
    font = pygame.font.SysFont('Arial', 64)
    small_font = pygame.font.SysFont('Arial', 14)
    
    # Unicode representations of chess pieces
    unicode_pieces = {
        'wp': '♙', 'wr': '♖', 'wn': '♘', 'wb': '♗', 'wq': '♕', 'wk': '♔',
        'bp': '♟', 'br': '♜', 'bn': '♞', 'bb': '♝', 'bq': '♛', 'bk': '♚'
    }
    
    for key, symbol in unicode_pieces.items():
        text_color = WHITE if key.startswith('w') else BLACK
        # Create the piece symbol
        piece_text = font.render(symbol, True, text_color)
        
        # Create the piece label
        label_text = small_font.render(PIECE_NAMES[key], True, RED)
        
        # Store both the piece and its label
        pieces[key] = {
            'symbol': piece_text,
            'label': label_text
        }
    
    return pieces

# Initialize board
def init_board():
    # Standard chess starting position
    # Empty squares are represented by ''
    board = [
        ['br', 'bn', 'bb', 'bq', 'bk', 'bb', 'bn', 'br'],
        ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp'],
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp'],
        ['wr', 'wn', 'wb', 'wq', 'wk', 'wb', 'wn', 'wr']
    ]
    return board

class SmartChessboard:
    def __init__(self):
        # Set up display
        self.screen = pygame.display.set_mode((SCALED_BOARD_SIZE + 200, SCALED_BOARD_SIZE))  # Extra width for captured pieces
        pygame.display.set_caption("Smart Chessboard Simulation")
        self.clock = pygame.time.Clock()
        
        # Initialize board state
        self.board = init_board()
        self.pieces = load_pieces()
        
        # Piece movement state
        self.selected_piece = None
        self.dragging = False
        self.drag_pos = (0, 0)
        
        # Hardware control simulation variables
        self.moving_hardware = False
        self.hardware_move_start = None
        self.hardware_move_end = None
        self.hardware_move_progress = 0.0
        self.captured_piece = None
        
        # Lists to track captured pieces
        self.captured_white = []
        self.captured_black = []
        
        # Electromagnet coil grid (20x20)
        self.coil_grid = [[0 for _ in range(COIL_GRID_SIZE)] for _ in range(COIL_GRID_SIZE)]
        self.show_coils = True  # Toggle to show/hide coil visualization
        
        # Create a separate surface for coils with transparency
        self.coil_surface = pygame.Surface((SCALED_BOARD_SIZE, SCALED_BOARD_SIZE), pygame.SRCALPHA)
        
        # Path planning variables
        self.capture_path = []  # Store calculated path for captured pieces
        self.capture_path_progress = 0  # Current position along the path
        self.show_paths = True  # Toggle to show/hide path visualization
        
        # Hardware controller (simulation mode by default)
        self.hardware_mode = False
        self.hardware_controller = None
        try:
            # Try to initialize hardware controller
            self.hardware_controller = ElectromagnetGridController(COIL_GRID_SIZE, simulation_mode=True)
            print("Hardware controller initialized in simulation mode")
        except Exception as e:
            print(f"Warning: Could not initialize hardware controller: {e}")
    
    def draw_board(self):
        # Draw the chess board
        for row in range(8):
            for col in range(8):
                # Determine square color
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                
                # Highlight selected piece
                if (self.selected_piece and 
                    self.selected_piece[0] == row and 
                    self.selected_piece[1] == col and
                    not self.dragging):
                    color = HIGHLIGHT
                
                # Draw square
                pygame.draw.rect(
                    self.screen, 
                    color, 
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                )
        
        # Draw the coil grid if enabled
        if self.show_coils:
            self._draw_coil_grid()
        
        # Draw pieces on the board
        for row in range(8):
            for col in range(8):
                # Draw piece if present and not being dragged
                piece = self.board[row][col]
                if piece and not (self.dragging and self.selected_piece and 
                                 self.selected_piece[0] == row and self.selected_piece[1] == col):
                    piece_img = self.pieces[piece]['symbol']
                    piece_rect = piece_img.get_rect(center=(
                        col * SQUARE_SIZE + SQUARE_SIZE // 2,
                        row * SQUARE_SIZE + SQUARE_SIZE // 2
                    ))
                    self.screen.blit(piece_img, piece_rect)
                    
                    # Draw piece label below the piece
                    label_img = self.pieces[piece]['label']
                    label_rect = label_img.get_rect(center=(
                        col * SQUARE_SIZE + SQUARE_SIZE // 2,
                        row * SQUARE_SIZE + SQUARE_SIZE // 2 + 30
                    ))
                    self.screen.blit(label_img, label_rect)
        
        # Draw dragged piece
        if self.dragging and self.selected_piece:
            row, col = self.selected_piece
            piece = self.board[row][col]
            if piece:
                piece_img = self.pieces[piece]['symbol']
                piece_rect = piece_img.get_rect(center=self.drag_pos)
                self.screen.blit(piece_img, piece_rect)
                
                # Draw piece label below the dragged piece
                label_img = self.pieces[piece]['label']
                label_rect = label_img.get_rect(center=(
                    self.drag_pos[0],
                    self.drag_pos[1] + 30
                ))
                self.screen.blit(label_img, label_rect)
        
        # Draw captured pieces area
        self._draw_captured_pieces()
        
        # Draw hardware movement simulation
        if self.moving_hardware:
            self._simulate_hardware_movement()
            
    def _draw_coil_grid(self):
        """Draw the 20x20 grid of electromagnet coils"""
        self.coil_surface.fill((0, 0, 0, 0))  # Clear with transparent background
        
        # Draw all coils
        for row in range(COIL_GRID_SIZE):
            for col in range(COIL_GRID_SIZE):
                # Calculate coil center position
                center_x = col * COIL_SPACING + COIL_SPACING / 2
                center_y = row * COIL_SPACING + COIL_SPACING / 2
                
                # Determine coil color based on activation level (0-100%)
                activation = self.coil_grid[row][col]
                if activation > 0:
                    # Mix inactive and active colors based on activation level
                    r = COIL_INACTIVE[0] + (COIL_ACTIVE[0] - COIL_INACTIVE[0]) * activation / 100
                    g = COIL_INACTIVE[1] + (COIL_ACTIVE[1] - COIL_INACTIVE[1]) * activation / 100
                    b = COIL_INACTIVE[2] + (COIL_ACTIVE[2] - COIL_INACTIVE[2]) * activation / 100
                    a = COIL_INACTIVE[3] + (COIL_ACTIVE[3] - COIL_INACTIVE[3]) * activation / 100
                    color = (int(r), int(g), int(b), int(a))
                else:
                    color = COIL_INACTIVE
                
                # Draw the coil
                pygame.draw.circle(self.coil_surface, color, (int(center_x), int(center_y)), COIL_RADIUS)
                pygame.draw.circle(self.coil_surface, COIL_BORDER, (int(center_x), int(center_y)), COIL_RADIUS, 1)
        
        # Blit the coil surface onto the screen
        self.screen.blit(self.coil_surface, (0, 0))
    
    def _draw_captured_pieces(self):
        # Draw the captured pieces area
        pygame.draw.rect(
            self.screen, 
            (50, 50, 50),  # Dark gray background
            (SCALED_BOARD_SIZE, 0, 200, SCALED_BOARD_SIZE)
        )
        
        # Draw title for captured pieces
        font = pygame.font.SysFont('Arial', 20)
        title = font.render("Captured Pieces", True, WHITE)
        self.screen.blit(title, (SCALED_BOARD_SIZE + 20, 20))
        
        # Draw separator line
        pygame.draw.line(
            self.screen,
            WHITE,
            (SCALED_BOARD_SIZE, SCALED_BOARD_SIZE // 2),
            (SCALED_BOARD_SIZE + 200, SCALED_BOARD_SIZE // 2),
            2
        )
        
        # Draw white captured pieces
        white_title = font.render("Black captured:", True, WHITE)
        self.screen.blit(white_title, (SCALED_BOARD_SIZE + 20, 60))
        
        for i, piece in enumerate(self.captured_black):
            piece_img = self.pieces[piece]['symbol']
            row = i // 4
            col = i % 4
            x = SCALED_BOARD_SIZE + 25 + col * 40
            y = 100 + row * 40
            
            # Scale down the piece for captured area
            scaled_img = pygame.transform.scale(piece_img, (30, 30))
            self.screen.blit(scaled_img, (x, y))
        
        # Draw black captured pieces
        black_title = font.render("White captured:", True, WHITE)
        self.screen.blit(black_title, (SCALED_BOARD_SIZE + 20, SCALED_BOARD_SIZE // 2 + 30))
        
        for i, piece in enumerate(self.captured_white):
            piece_img = self.pieces[piece]['symbol']
            row = i // 4
            col = i % 4
            x = SCALED_BOARD_SIZE + 25 + col * 40
            y = SCALED_BOARD_SIZE // 2 + 70 + row * 40
            
            # Scale down the piece for captured area
            scaled_img = pygame.transform.scale(piece_img, (30, 30))
            self.screen.blit(scaled_img, (x, y))
    
    def _simulate_hardware_movement(self):
        # This method simulates the hardware movement of chess pieces
        if self.hardware_move_progress < 1.0:
            start_row, start_col = self.hardware_move_start
            end_row, end_col = self.hardware_move_end
            
            # Calculate current position based on progress
            current_col = start_col + (end_col - start_col) * self.hardware_move_progress
            current_row = start_row + (end_row - start_row) * self.hardware_move_progress
            
            # Draw a path to show the movement
            pygame.draw.line(
                self.screen,
                (255, 0, 0),  # Red line
                (start_col * SQUARE_SIZE + SQUARE_SIZE // 2, start_row * SQUARE_SIZE + SQUARE_SIZE // 2),
                (end_col * SQUARE_SIZE + SQUARE_SIZE // 2, end_row * SQUARE_SIZE + SQUARE_SIZE // 2),
                3
            )
            
            # Draw a circle at the current position to represent the moving piece
            pygame.draw.circle(
                self.screen,
                (255, 0, 0),  # Red circle
                (int(current_col * SQUARE_SIZE + SQUARE_SIZE // 2), 
                 int(current_row * SQUARE_SIZE + SQUARE_SIZE // 2)),
                10
            )
            
            # Reset coils before activating for this frame
            self._reset_coils()
            
            # Activate coils near the current position
            self._activate_coils_for_position(current_row, current_col)
            
            # If there's a captured piece, show it moving to the captured area
            if self.captured_piece:
                # If we haven't calculated a path yet, do it now
                if not self.capture_path:
                    # Calculate path for captured piece avoiding obstacles
                    self.capture_path = self._find_capture_path(end_row, end_col, self.captured_piece[0] == 'w')
                    self.capture_path_progress = 0
                
                # Calculate current position along the capture path
                if self.capture_path and self.capture_path_progress < len(self.capture_path):
                    # Get current point in the path
                    path_index = min(int(self.capture_path_progress), len(self.capture_path) - 1)
                    current_path_row, current_path_col = self.capture_path[path_index]
                    
                    # Convert to screen coordinates
                    captured_x = current_path_col * SQUARE_SIZE + SQUARE_SIZE // 2
                    captured_y = current_path_row * SQUARE_SIZE + SQUARE_SIZE // 2
                    
                    # Draw the entire capture path if enabled
                    if self.show_paths:
                        for i in range(len(self.capture_path) - 1):
                            r1, c1 = self.capture_path[i]
                            r2, c2 = self.capture_path[i + 1]
                            pygame.draw.line(
                                self.screen,
                                GREEN,  # Green line for planned path
                                (c1 * SQUARE_SIZE + SQUARE_SIZE // 2, r1 * SQUARE_SIZE + SQUARE_SIZE // 2),
                                (c2 * SQUARE_SIZE + SQUARE_SIZE // 2, r2 * SQUARE_SIZE + SQUARE_SIZE // 2),
                                2
                            )
                    
                    # Draw the captured piece at its current position
                    pygame.draw.circle(
                        self.screen,
                        BLUE,  # Blue circle
                        (int(captured_x), int(captured_y)),
                        10
                    )
                    
                    # Advance the captured piece along the path
                    self.capture_path_progress += 0.15  # Speed of captured piece movement
                    
                    # Activate coils for the captured piece (with lower intensity)
                    self._activate_coils_for_position(current_path_row, current_path_col, intensity=50)
            
            # Increment progress for capturing piece
            self.hardware_move_progress += 0.01  # Slower for better visualization
            
            # If movement is complete, update the board
            if self.hardware_move_progress >= 1.0:
                piece = self.board[self.hardware_move_start[0]][self.hardware_move_start[1]]
                self.board[self.hardware_move_start[0]][self.hardware_move_start[1]] = ''
                self.board[self.hardware_move_end[0]][self.hardware_move_end[1]] = piece
                self.moving_hardware = False
                self.captured_piece = None
                self.capture_path = []
                self.capture_path_progress = 0
                
                # Reset all coils
                self._reset_coils()
    
    def _activate_coils_for_position(self, row, col, intensity=100):
        """Activate coils around the current piece position"""
        # Convert chess board position to coil grid position
        grid_col = col * (COIL_GRID_SIZE / 8)
        grid_row = row * (COIL_GRID_SIZE / 8)
        
        # Determine which coils to activate based on the magnetic attraction pattern
        # We'll use a radial pattern with decreasing strength as we move away from the center
        max_radius = 3  # Coils within this radius will be activated
        
        for r in range(COIL_GRID_SIZE):
            for c in range(COIL_GRID_SIZE):
                # Calculate distance from current position
                distance = math.sqrt((r - grid_row)**2 + (c - grid_col)**2)
                
                if distance < max_radius:
                    # Calculate activation level (100% at center, decreasing outward)
                    # Scale by the passed intensity parameter (0-100%)
                    activation = max(0, intensity * (1 - distance / max_radius) / 100)
                    # Only set if the new activation is higher than the current value
                    self.coil_grid[r][c] = max(self.coil_grid[r][c], activation * 100)
                    
                    # If hardware controller is active, set the real coil power
                    if self.hardware_mode and self.hardware_controller:
                        self.hardware_controller.set_coil_power(r, c, activation * 100)
    
    def _reset_coils(self):
        """Reset all coils to inactive state"""
        for r in range(COIL_GRID_SIZE):
            for c in range(COIL_GRID_SIZE):
                self.coil_grid[r][c] = 0
                
        # If hardware controller is active, reset the real coils
        if self.hardware_mode and self.hardware_controller:
            self.hardware_controller.reset_all_coils()
    
    def _find_capture_path(self, start_row, start_col, is_white_piece):
        """
        A* pathfinding algorithm to find a path for the captured piece to exit the board,
        avoiding collisions with other pieces.
        """
        print(f"Finding capture path for {'white' if is_white_piece else 'black'} piece at ({start_row}, {start_col})")
        
        # Determine destination (outside the board)
        if is_white_piece:
            # White piece captured by black goes to black's capture area
            dest_row = 2
            dest_col = 10  # Off the board to the right
        else:
            # Black piece captured by white goes to white's capture area
            dest_row = 6
            dest_col = 10  # Off the board to the right
        
        # Create an obstacle grid
        obstacle_grid = self._create_obstacle_grid()
        
        # A* algorithm
        start = (start_row, start_col)
        goal = (dest_row, dest_col)
        
        # The open set (priority queue)
        open_set = []
        heapq.heappush(open_set, (0, start))  # (priority, position)
        
        # The set of visited nodes
        closed_set = set()
        
        # For each node, which node it came from
        came_from = {}
        
        # For each node, the cost to reach it from the start
        g_score = {start: 0}
        
        # For each node, the total cost to reach the goal through it
        f_score = {start: self._heuristic(start, goal)}
        
        while open_set:
            # Get the node with the lowest f_score
            current_f, current = heapq.heappop(open_set)
            
            # If we've reached the goal, reconstruct and return the path
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                print(f"Found path with {len(path)} steps")
                return path
            
            # Mark as visited
            closed_set.add(current)
            
            # Check all neighbors
            for neighbor in self._get_neighbors(current, obstacle_grid):
                # Skip if already evaluated
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g_score = g_score.get(current, float('inf')) + 1
                
                # If this path to neighbor is better than any previous one
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    # Record this path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                    
                    # Add to open set if not already there
                    if any(pos == neighbor for _, pos in open_set):
                        continue
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # If we get here, no path was found
        print("No path found, using direct path")
        # Return a simple direct path as fallback
        # Move to the right edge then to the destination
        path = [(start_row, col) for col in range(start_col, 9)]  # Move to edge
        path.extend([(row, 8) for row in range(start_row, dest_row, 1 if dest_row > start_row else -1)])  # Move to right height
        path.append(goal)  # Move to destination
        return path
    
    def _create_obstacle_grid(self):
        """Create a grid marking where pieces are on the board"""
        # Initialize grid with False (no obstacles)
        grid = [[False for _ in range(12)] for _ in range(8)]  # Larger than board to allow off-board paths
        
        # Mark pieces as obstacles
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece:
                    # Mark the piece itself as an obstacle
                    grid[row][col] = True
                    
                    # Get the piece type (last character of piece code)
                    piece_type = piece[1].lower()
                    
                    # Add a safety margin based on piece size
                    if piece_type in SCALED_PIECE_DIAMETERS:
                        radius = SCALED_PIECE_DIAMETERS[piece_type] / SQUARE_SIZE / 2
                        
                        # Mark surrounding cells as obstacles based on piece radius
                        for r in range(max(0, int(row - radius)), min(8, int(row + radius + 1))):
                            for c in range(max(0, int(col - radius)), min(8, int(col + radius + 1))):
                                # Check if point is within the radius of the piece
                                distance = math.sqrt((r - row)**2 + (c - col)**2)
                                if distance <= radius:
                                    grid[r][c] = True
        
        return grid
    
    def _get_neighbors(self, position, obstacle_grid):
        """Get valid neighboring positions"""
        row, col = position
        neighbors = []
        
        # Check all 8 directions (including diagonals)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip current position
                
                new_row, new_col = row + dr, col + dc
                
                # Check if within board boundaries or valid off-board area
                if 0 <= new_row < 8 and 0 <= new_col < 12:
                    # Skip if there's an obstacle
                    if new_col < 8 and obstacle_grid[new_row][new_col]:
                        continue
                    
                    neighbors.append((new_row, new_col))
        
        return neighbors
    
    def _heuristic(self, a, b):
        """Calculate heuristic (Manhattan distance)"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _reconstruct_path(self, came_from, current):
        """Reconstruct the path from the came_from dict"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        # Reverse to get start to goal
        path.reverse()
        return path
    
    def handle_click(self, pos):
        col = pos[0] // SQUARE_SIZE
        row = pos[1] // SQUARE_SIZE
        
        # Make sure click is within board boundaries
        if not (0 <= row < 8 and 0 <= col < 8):
            return
        
        # If no piece is selected, select the piece at the clicked position
        if not self.selected_piece:
            if self.board[row][col]:  # If there's a piece at the clicked position
                self.selected_piece = (row, col)
                self.dragging = True
                self.drag_pos = (col * SQUARE_SIZE + SQUARE_SIZE // 2, 
                                row * SQUARE_SIZE + SQUARE_SIZE // 2)
                print(f"Selected {PIECE_NAMES[self.board[row][col]]} at position ({row}, {col})")
        else:
            # If a piece is already selected, move it to the clicked position
            start_row, start_col = self.selected_piece
            
            # Move the piece
            if not self.moving_hardware:
                move_successful = self._move_piece(start_row, start_col, row, col)
                if not move_successful:
                    print(f"Move failed: Cannot place {PIECE_NAMES[self.board[start_row][start_col]]} on occupied square")
            
            # Reset selection
            self.selected_piece = None
            self.dragging = False
    
    def handle_drag(self, pos):
        if self.dragging:
            self.drag_pos = pos
    
    def handle_release(self, pos):
        if self.dragging and self.selected_piece:
            col = pos[0] // SQUARE_SIZE
            row = pos[1] // SQUARE_SIZE
            
            # Make sure release is within board boundaries
            if 0 <= row < 8 and 0 <= col < 8:
                start_row, start_col = self.selected_piece
                
                # Move the piece
                if not self.moving_hardware:
                    move_successful = self._move_piece(start_row, start_col, row, col)
                    if not move_successful:
                        print(f"Move failed: Cannot place {PIECE_NAMES[self.board[start_row][start_col]]} on occupied square")
            
            # Reset selection
            self.selected_piece = None
            self.dragging = False
    
    def _move_piece(self, start_row, start_col, end_row, end_col):
        """Move a piece from start position to end position with hardware simulation"""
        # Only proceed if start and end positions are different
        if (start_row, start_col) != (end_row, end_col):
            moving_piece = self.board[start_row][start_col]
            target_square = self.board[end_row][end_col]
            
            # Check if the target square is occupied
            if target_square != '':
                # If occupied by same color, prevent movement
                if moving_piece[0] == target_square[0]:  # First character indicates color ('w' or 'b')
                    print(f"Cannot move to ({end_row}, {end_col}) - square is already occupied by a piece of the same color: {PIECE_NAMES[target_square]}")
                    return False
                # If occupied by enemy, capture
                else:
                    print(f"{PIECE_NAMES[moving_piece]} captures {PIECE_NAMES[target_square]}!")
                    # Store the captured piece
                    self.captured_piece = target_square
                    # Add to appropriate captured list
                    if target_square[0] == 'w':
                        self.captured_white.append(target_square)
                    else:
                        self.captured_black.append(target_square)
            
            # Start hardware movement simulation
            self.moving_hardware = True
            self.hardware_move_start = (start_row, start_col)
            self.hardware_move_end = (end_row, end_col)
            self.hardware_move_progress = 0.0
            
            # In a real implementation, this is where you'd send commands to the hardware
            self._send_hardware_command(start_row, start_col, end_row, end_col)
            return True
        return False
    
    def _send_hardware_command(self, start_row, start_col, end_row, end_col):
        """
        Placeholder for sending commands to the actual hardware.
        In a real implementation, this would communicate with your electromagnets,
        motors, or other physical components.
        """
        # Check if this is a capture move
        target_piece = self.board[end_row][end_col]
        moving_piece = self.board[start_row][start_col]
        
        if target_piece == '':
            print(f"Hardware Command: Move {PIECE_NAMES[moving_piece]} from ({start_row}, {start_col}) to ({end_row}, {end_col})")
        else:
            print(f"Hardware Command: Move {PIECE_NAMES[moving_piece]} from ({start_row}, {start_col}) to capture {PIECE_NAMES[target_piece]} at ({end_row}, {end_col})")
            
        print(f"Activating electromagnet coil grid (20x20 = 400 coils)")
        
        # If hardware mode is active, use the hardware controller
        if self.hardware_mode and self.hardware_controller:
            # In hardware mode, we'd send real commands to the electromagnet controller
            self.hardware_controller.move_piece(start_row, start_col, end_row, end_col)
            
            # If capturing, handle the captured piece
            if target_piece != '':
                # Calculate path for captured piece
                capture_path = self._find_capture_path(end_row, end_col, target_piece[0] == 'w')
                # Move the captured piece along the path
                self.hardware_controller.move_captured_piece(capture_path)
    
    def reset_board(self):
        """Reset the board to the initial state"""
        self.board = init_board()
        self.selected_piece = None
        self.dragging = False
        self.moving_hardware = False
        self.captured_piece = None
        self.captured_white = []
        self.captured_black = []
        self.capture_path = []
        self.capture_path_progress = 0
        self._reset_coils()
        
        # Reset hardware controller if active
        if self.hardware_mode and self.hardware_controller:
            self.hardware_controller.reset()
    
    def toggle_hardware_mode(self):
        """Toggle between simulation and hardware control modes"""
        self.hardware_mode = not self.hardware_mode
        print(f"Hardware control mode: {'ON' if self.hardware_mode else 'OFF'}")
        
        if self.hardware_mode and not self.hardware_controller:
            try:
                # Try to initialize hardware controller in real mode
                self.hardware_controller = ElectromagnetGridController(COIL_GRID_SIZE, simulation_mode=False)
                print("Hardware controller initialized in real mode")
            except Exception as e:
                print(f"Error: Could not initialize hardware controller: {e}")
                self.hardware_mode = False
    
    def run(self):
        """Main game loop"""
        running = True
        
        # Display instructions
        print("\n===== SMART CHESSBOARD SIMULATOR =====")
        print("- Drag and drop pieces to move them")
        print("- Press 'r' to reset the board")
        print("- Press 'c' to toggle electromagnet coil visualization")
        print("- Press 'p' to toggle path visualization")
        print("- Press 'h' to toggle hardware control mode")
        print("- Collision detection prevents placing pieces on occupied squares")
        print("- Each piece is labeled with its type and color")
        print("- You can capture enemy pieces by moving onto their square")
        print("- Captured pieces use A* pathfinding to avoid obstacles")
        print("- The board uses a 20x20 grid of electromagnet coils")
        print("======================================\n")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        self.handle_click(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_drag(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left mouse button
                        self.handle_release(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset board with 'r' key
                        self.reset_board()
                        print("Board reset to initial position")
                    elif event.key == pygame.K_c:  # Toggle coil visualization with 'c' key
                        self.show_coils = not self.show_coils
                        print(f"Electromagnet coil visualization: {'ON' if self.show_coils else 'OFF'}")
                    elif event.key == pygame.K_p:  # Toggle path visualization with 'p' key
                        self.show_paths = not self.show_paths
                        print(f"Path planning visualization: {'ON' if self.show_paths else 'OFF'}")
                    elif event.key == pygame.K_h:  # Toggle hardware mode with 'h' key
                        self.toggle_hardware_mode()
            
            # Draw everything
            self.screen.fill(BLACK)
            self.draw_board()
            
            # Draw status information at the top of the screen
            font = pygame.font.SysFont('Arial', 18)
            status_text = font.render("Smart Chessboard - 'r':reset, 'c':coils, 'p':paths, 'h':hardware", True, WHITE)
            self.screen.blit(status_text, (10, 10))
            
            # Display hardware mode status
            hardware_text = font.render(f"Hardware mode: {'ON' if self.hardware_mode else 'OFF'}", True, GREEN if self.hardware_mode else WHITE)
            self.screen.blit(hardware_text, (10, SCALED_BOARD_SIZE - 50))
            
            # Display coil information
            if self.show_coils:
                coil_info = font.render(f"20x20 Coil Grid ({COIL_GRID_SIZE}x{COIL_GRID_SIZE} = {COIL_GRID_SIZE*COIL_GRID_SIZE} coils)", True, WHITE)
                self.screen.blit(coil_info, (10, SCALED_BOARD_SIZE - 30))
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Clean up hardware if active
        if self.hardware_controller:
            self.hardware_controller.shutdown()
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    chessboard = SmartChessboard()
    chessboard.run()