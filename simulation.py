import pygame
import sys
import time

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 800
SQUARE_SIZE = BOARD_SIZE // 8
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (124, 252, 0)  # Highlight color for selected pieces
RED = (255, 0, 0)

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
        self.screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
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
        self.hardware_move_progress = 0
        
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
        
        # Draw hardware movement simulation
        if self.moving_hardware:
            self._simulate_hardware_movement()
            
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
            
            # Increment progress
            self.hardware_move_progress += 0.02
            
            # If movement is complete, update the board
            if self.hardware_move_progress >= 1.0:
                piece = self.board[self.hardware_move_start[0]][self.hardware_move_start[1]]
                self.board[self.hardware_move_start[0]][self.hardware_move_start[1]] = ''
                self.board[self.hardware_move_end[0]][self.hardware_move_end[1]] = piece
                self.moving_hardware = False
    
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
            # Check if the destination square is already occupied
            if self.board[end_row][end_col] != '':
                print(f"Cannot move to ({end_row}, {end_col}) - square is already occupied by {PIECE_NAMES[self.board[end_row][end_col]]}")
                return False
                
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
        print(f"Hardware Command: Move piece from ({start_row}, {start_col}) to ({end_row}, {end_col})")
        
        # For a real implementation, this might look like:
        # 
        # # Activate the electromagnet at the starting position
        # self._activate_electromagnet(start_row, start_col)
        # 
        # # Calculate path for the piece to follow
        # path = self._calculate_path(start_row, start_col, end_row, end_col)
        # 
        # # Move the electromagnet along the path
        # for point in path:
        #     self._move_electromagnet_to(point[0], point[1])
        #     time.sleep(0.01)  # Small delay for smooth movement
        # 
        # # Deactivate the electromagnet at the destination
        # self._deactivate_electromagnet()
    
    def _activate_electromagnet(self, row, col):
        """
        Activate the electromagnet at the specified position.
        This would interface with your hardware control system.
        """
        print(f"Activating electromagnet at ({row}, {col})")
        # In a real implementation, this might use GPIO pins on a Raspberry Pi
        # or communicate with a microcontroller via serial or I2C:
        #
        # Example with RPi.GPIO:
        # gpio_pin = self._get_electromagnet_pin(row, col)
        # GPIO.output(gpio_pin, GPIO.HIGH)
    
    def _deactivate_electromagnet(self):
        """Deactivate the currently active electromagnet."""
        print("Deactivating electromagnet")
        # In a real implementation:
        # GPIO.output(self.active_electromagnet_pin, GPIO.LOW)
    
    def _move_electromagnet_to(self, row, col):
        """
        Move the electromagnet to the specified position.
        This would control stepper motors or other positioning mechanism.
        """
        print(f"Moving electromagnet to ({row}, {col})")
        # In a real implementation, this might use stepper motor control:
        #
        # x_steps = self._calculate_x_steps(col)
        # y_steps = self._calculate_y_steps(row)
        # self.motor_controller.move_to(x_steps, y_steps)
    
    def _calculate_path(self, start_row, start_col, end_row, end_col):
        """
        Calculate a path for the piece to follow.
        Could implement various algorithms like A* for path finding if needed.
        """
        # Simple linear path for demonstration
        points = []
        steps = max(abs(end_row - start_row), abs(end_col - start_col)) * 10
        for i in range(steps + 1):
            t = i / steps
            row = start_row + (end_row - start_row) * t
            col = start_col + (end_col - start_col) * t
            points.append((row, col))
        return points
    
    def reset_board(self):
        """Reset the board to the initial state"""
        self.board = init_board()
        self.selected_piece = None
        self.dragging = False
        self.moving_hardware = False
    
    def run(self):
        """Main game loop"""
        running = True
        
        # Display instructions
        print("\n===== SMART CHESSBOARD SIMULATOR =====")
        print("- Drag and drop pieces to move them")
        print("- Press 'r' to reset the board")
        print("- Collision detection prevents placing pieces on occupied squares")
        print("- Each piece is labeled with its type and color")
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
            
            # Draw everything
            self.screen.fill(BLACK)
            self.draw_board()
            
            # Draw status information at the top of the screen
            font = pygame.font.SysFont('Arial', 18)
            status_text = font.render("Smart Chessboard - Press 'r' to reset, drag pieces to move", True, WHITE)
            self.screen.blit(status_text, (10, 10))
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    chessboard = SmartChessboard()
    chessboard.run()