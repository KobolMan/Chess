# chess_simulation.py

import pygame
import numpy as np
import math
import sys
import time
import heapq
from enum import Enum
import os # Needed for path joining potentially, and by coil_controller fix

# Import from local modules
from chess_pieces import ChessPiece, PieceColor, PieceType # Enums too
try:
    # Assuming MAX_COIL_AMPS is defined in coil_controller (add it if not)
    from coil_controller import CoilGrid, MAX_COIL_AMPS
except ImportError:
    print("Warning: MAX_COIL_AMPS not found in coil_controller. Using default.")
    from coil_controller import CoilGrid
    MAX_COIL_AMPS = 0.5 # Default value if not imported

from pathfinding import PathFinder
from visualization import ChessRenderer
from hardware_interface import ElectromagnetController # Type hint

# Attempt to import pygame_widgets
try:
    import pygame_widgets
    from pygame_widgets.slider import Slider
    from pygame_widgets.textbox import TextBox
    WIDGETS_AVAILABLE = True
except ImportError:
    print("WARNING: pygame_widgets not found. PID sliders will be unavailable.")
    print("Install using: pip install pygame_widgets")
    WIDGETS_AVAILABLE = False
    # Define dummy classes if import fails to avoid NameErrors later
    class Slider: pass
    class TextBox: pass

# Import for heatmap fix if needed here (usually needed in coil_controller)
from scipy.ndimage import gaussian_filter, map_coordinates


# --- Constants ---
# Default width/height used if screen detection fails in main.py
DEFAULT_WINDOW_WIDTH = 1600
DEFAULT_WINDOW_HEIGHT = 900
BOARD_SQUARES = 8
COIL_GRID_SIZE = 20
SIDE_PANEL_WIDTH = 400 # Fixed width for control panel + sliders
FPS = 60

class ChessBoard:
    """
    Orchestrates the chess simulation with PID control and interactive tuning.
    """
    def __init__(self, hardware_controller: ElectromagnetController,
                 debug_mode=False, initial_window_size=None):
        self.hardware_controller = hardware_controller
        self.debug_mode = debug_mode # Use constructor arg
    
        # --- Window Setup ---
        if initial_window_size:
            self.window_width, self.window_height = initial_window_size
            print(f"Attempting initial size: {self.window_width}x{self.window_height}")
        else:
            self.window_width, self.window_height = DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT
            print(f"Using default size: {self.window_width}x{self.window_height}")
    
        # Create the display surface with RESIZABLE flag
        # SCALED can sometimes cause issues with widget positioning or rendering, try without first
        try:
             self.screen = pygame.display.set_mode(
                 (self.window_width, self.window_height),
                 pygame.RESIZABLE # Start without SCALED unless needed
             )
        except Exception as e:
            print(f"Error setting Pygame display mode: {e}. Exiting.")
            sys.exit(1)
    
        # Get final actual size after mode set
        self.window_width, self.window_height = self.screen.get_size()
        print(f"Final actual window size: {self.window_width}x{self.window_height}")
    
        pygame.display.set_caption("Electromagnetic Chess Simulation (PID Tuning)")
    
        # Initialize sizes based on actual window size
        self._calculate_layout(self.window_width, self.window_height)
    
        # Initialize components
        self.renderer = ChessRenderer(
            self.board_size_px, BOARD_SQUARES,
            self.window_width, self.window_height,
            board_x_offset=self.board_x_offset,
            heatmap_size_px=self.heatmap_size_px # Pass heatmap size
        )
        self.coil_grid = CoilGrid(size=COIL_GRID_SIZE, board_squares=BOARD_SQUARES)
        self.path_finder = PathFinder(board_size=BOARD_SQUARES)

        # --- PID GAINS (Iteration 3) ---
        self.pid_kp = 80.0  # Updated from 50.0
        self.pid_ki = 0.2   # Updated from 5.0
        self.pid_kd = 44.0  # Updated from 50.0
        self.terminal_damping = 5.0  # Keeping this value
        self.pid_integral_max = 40.0  # Keeping this value
        
        # Also update the temporary values to match
        self.temp_pid_kp = self.pid_kp
        self.temp_pid_ki = self.pid_ki
        self.temp_pid_kd = self.pid_kd

        self.pid_integral = np.array([0.0, 0.0])
        self.pid_previous_error = np.array([0.0, 0.0])

        # Game State & Simulation Params
        self.pieces: list[ChessPiece] = []
        self.captured_white: list[ChessPiece] = []
        self.captured_black: list[ChessPiece] = []
        self.selected_piece: ChessPiece | None = None
        self.target_position: tuple[float, float] | None = None
        self.move_in_progress: bool = False
        self.move_complete: bool = False
        self.last_move_pid_force_mag: float = 0.0 # Store force magnitude
        self.current_total_sim_amps: float = 0.0 # Store sim current

        self.captured_piece: ChessPiece | None = None
        self.capture_step_index: int = 0
        self.capture_path_finished: bool = False
        self.temporarily_moved_pieces: list[ChessPiece] = []

        self.simulation_speed = 1.0
        self.field_update_timer = 0.0
        self.field_update_interval = 0.05

        # Visualization Options
        self.show_coils = False
        self.show_field = False
        self.show_paths = True
        self.show_heatmap = True
        self.show_center_markers = False
        self.renderer.show_position_dots = True

        # Move Pattern Options (for viz/hardware)
        self.current_pattern="directed"
        self.patterns=["directed","knight","radial"]

        # Offboard capture targets
        self.capture_target_white = (BOARD_SQUARES+1.0, BOARD_SQUARES-1.5)
        self.capture_target_black = (BOARD_SQUARES+1.0, 0.5)

        self.clock = pygame.time.Clock()
        self.heatmap_surface = None
        self.heatmap_needs_update = True

        # --- Initialize PID Sliders ---
        self.pid_sliders = {}
        self.pid_textboxes = {}
        if WIDGETS_AVAILABLE:
            self._create_pid_sliders()
        else:
            print("PID Sliders disabled as pygame_widgets is not available.")

        self.initialize_pieces()
        print(f"ChessBoard initialized. HW Sim: {self.hardware_controller.simulation_mode}, Debug: {self.debug_mode}")
        self.print_current_pid_settings("Initial")

    def print_current_pid_settings(self, context="Current"):
        """Prints the currently active PID settings."""
        print(f"--- {context} PID GAINS: Kp={self.pid_kp:.1f}, Ki={self.pid_ki:.1f}, Kd={self.pid_kd:.1f}, TermDamp={self.terminal_damping:.1f}, Imax={self.pid_integral_max:.1f} ---")

    def _center_window(self, width, height):
        """Center the pygame window on the screen."""
        # Get the screen information
        info = pygame.display.Info()
        screen_width = info.current_w
        screen_height = info.current_h

        # Calculate the center position
        pos_x = (screen_width - width) // 2
        pos_y = (screen_height - height) // 2

        # Ensure position is not negative (can happen with multi-monitor setups)
        pos_x = max(0, pos_x)
        pos_y = max(0, pos_y)

        print(f"Centering window at position: ({pos_x}, {pos_y})")

        # Set the window position using SDL environment variable
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{pos_x},{pos_y}"

    def _calculate_layout(self, window_width, window_height):
        """ Recalculates layout based on current window size. """
        self.window_width = window_width
        self.window_height = window_height

        available_width = self.window_width - SIDE_PANEL_WIDTH
        # Keep board and heatmap square and roughly equal, leaving space for panel
        board_size = min(
            available_width // 2,       # Half available width
            self.window_height - 100    # Fit vertically with margin
        )
        # Ensure board size is multiple of squares and has min size
        board_size = max(BOARD_SQUARES * 20, (board_size // BOARD_SQUARES) * BOARD_SQUARES) # Min size 20px/square

        self.board_size_px = board_size
        self.heatmap_size_px = board_size # Keep them same size
        self.square_size_px = board_size // BOARD_SQUARES

        # Heatmap on left, board next to it
        self.board_x_offset = self.heatmap_size_px
        # Panel starts after the board
        self.panel_x = self.board_x_offset + self.board_size_px

        # print(f"Layout Recalculated: Win={window_width}x{window_height}, Board/Heatmap={board_size}px, PanelX={self.panel_x}")

    def _create_pid_sliders(self):
        """Creates the PID slider widgets."""
        if not WIDGETS_AVAILABLE: return

        # Clear previous widgets if they exist (needed for resize)
        self.pid_sliders.clear()
        self.pid_textboxes.clear()
        # If pygame_widgets needs explicit cleanup, do it here (check its docs)

        slider_width = SIDE_PANEL_WIDTH - 100 # Width of sliders, leave space for text
        slider_x = self.panel_x + 30         # X position of sliders
        slider_y_start = 450                 # Starting Y position (Adjust based on panel layout)
        slider_spacing = 70                  # Vertical space between sliders

        # Ensure slider start position is reasonable within window height
        if slider_y_start > self.window_height - 3 * slider_spacing:
            slider_y_start = self.window_height - 3 * slider_spacing - 20 # Adjust if too low

        # --- Kp Slider ---
        self.pid_sliders['kp'] = Slider(self.screen, slider_x, slider_y_start, slider_width, 20, min=0, max=200, step=1, initial=self.temp_pid_kp, colour=(200,200,200), handleColour=(0,150,0))
        self.pid_textboxes['kp'] = TextBox(self.screen, slider_x + slider_width + 10, slider_y_start - 5, 50, 30, fontSize=18, borderThickness=0, colour=self.renderer.LIGHT_GRAY, textColour=self.renderer.BLACK)
        self.pid_textboxes['kp'].disable() # Read-only display

        # --- Ki Slider ---
        self.pid_sliders['ki'] = Slider(self.screen, slider_x, slider_y_start + slider_spacing, slider_width, 20, min=0, max=50, step=0.1, initial=self.temp_pid_ki, colour=(200,200,200), handleColour=(0,0,150))
        self.pid_textboxes['ki'] = TextBox(self.screen, slider_x + slider_width + 10, slider_y_start + slider_spacing - 5, 50, 30, fontSize=18, borderThickness=0, colour=self.renderer.LIGHT_GRAY, textColour=self.renderer.BLACK)
        self.pid_textboxes['ki'].disable()

        # --- Kd Slider ---
        self.pid_sliders['kd'] = Slider(self.screen, slider_x, slider_y_start + 2 * slider_spacing, slider_width, 20, min=0, max=200, step=1, initial=self.temp_pid_kd, colour=(200,200,200), handleColour=(150,0,0))
        self.pid_textboxes['kd'] = TextBox(self.screen, slider_x + slider_width + 10, slider_y_start + 2 * slider_spacing - 5, 50, 30, fontSize=18, borderThickness=0, colour=self.renderer.LIGHT_GRAY, textColour=self.renderer.BLACK)
        self.pid_textboxes['kd'].disable()

        self._update_slider_textboxes() # Set initial text

    def _update_slider_textboxes(self):
        """Updates the text boxes next to sliders with current temp values."""
        if WIDGETS_AVAILABLE and self.pid_sliders: # Check if sliders exist
            try:
                # Check if slider objects actually exist before calling methods
                if 'kp' in self.pid_sliders and self.pid_sliders['kp'] is not None:
                     self.temp_pid_kp = self.pid_sliders['kp'].getValue()
                     self.pid_textboxes['kp'].setText(f"{self.temp_pid_kp:.1f}")
                if 'ki' in self.pid_sliders and self.pid_sliders['ki'] is not None:
                     self.temp_pid_ki = self.pid_sliders['ki'].getValue()
                     self.pid_textboxes['ki'].setText(f"{self.temp_pid_ki:.1f}")
                if 'kd' in self.pid_sliders and self.pid_sliders['kd'] is not None:
                     self.temp_pid_kd = self.pid_sliders['kd'].getValue()
                     self.pid_textboxes['kd'].setText(f"{self.temp_pid_kd:.1f}")
            except Exception as e:
                # Reduced error spam during resize
                # print(f"Warning: Error updating slider textboxes: {e}")
                pass

    # Add window repositioning to the handle_resize method
# Find the handle_resize method and update it to include centering

    def handle_resize(self, new_width, new_height):
        """
        Handle window resize events with robust error handling and
        multi-monitor support. Avoids SDL renderer creation failures.
        """
        # Ignore invalid sizes
        if new_width <= 100 or new_height <= 100:
            return

        # Apply reasonable limits to prevent renderer creation failures
        max_width = 3000  # Practical limit to prevent issues
        max_height = 1600  # Practical limit to prevent issues
        if new_width > max_width or new_height > max_height:
            new_width = min(new_width, max_width)
            new_height = min(new_height, max_height)
            print(f"Limiting resize dimensions to {new_width}x{new_height}")

        # Get current time for debounce calculations
        current_time = time.time()

        # Initialize debounce tracking attributes if needed
        if not hasattr(self, '_last_resize_time'):
            self._last_resize_time = 0
        if not hasattr(self, '_last_resize_dims'):
            self._last_resize_dims = (self.window_width, self.window_height)
        if not hasattr(self, '_resize_settling'):
            self._resize_settling = False
        if not hasattr(self, '_resize_cooldown'):
            self._resize_cooldown = 0.5  # Seconds to wait after major resize

        # Skip if in settling period after major size change
        if self._resize_settling:
            if current_time - self._last_resize_time < self._resize_cooldown:
                return
            else:
                # End of settling period
                self._resize_settling = False

        # Detect if this is a significant change (likely moving between monitors)
        last_width, last_height = self._last_resize_dims
        width_change = abs(new_width - last_width) / max(last_width, 1)
        height_change = abs(new_height - last_height) / max(last_height, 1)

        if width_change > 0.2 or height_change > 0.2:  # 20% change indicates major resize
            # Major resize detected - likely moving between monitors
            # Enter settling period where small adjustments are ignored
            self._resize_settling = True
            self._last_resize_time = current_time
            self._last_resize_dims = (new_width, new_height)
            print(f"Major resize detected: {new_width}x{new_height}")
        else:
            # Normal resize behavior with debouncing

            # Skip if resize happens too quickly after previous resize
            if current_time - self._last_resize_time < 0.3:  # 300ms debounce
                return

            # Skip if dimensions haven't changed much
            if abs(new_width - last_width) < 10 and abs(new_height - last_height) < 10:
                return

        # Update tracking variables
        self._last_resize_time = current_time
        self._last_resize_dims = (new_width, new_height)

        print(f"Applying resize to {new_width}x{new_height}")

        # *** Actual resize logic begins here ***
        # Recalculate layout
        self._calculate_layout(new_width, new_height)

        # For major size changes, set centered environment variable
        if width_change > 0.2 or height_change > 0.2:
            os.environ['SDL_VIDEO_CENTERED'] = '1'

        # Recreate screen surface with error handling
        try:
            # Try without SCALED flag first - more compatible across systems
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height), 
                pygame.RESIZABLE
            )
        except pygame.error as e:
            print(f"Error resizing display: {e}")
            try:
                # Fall back to even more basic window if first attempt fails
                print("Trying fallback resize approach...")
                # Use smaller dimensions if needed
                fallback_width = min(self.window_width, 1200)
                fallback_height = min(self.window_height, 800)

                # Try to center again
                os.environ['SDL_VIDEO_CENTERED'] = '1'

                self.screen = pygame.display.set_mode(
                    (fallback_width, fallback_height),
                    pygame.RESIZABLE
                )
                # Update layout for the fallback size
                self._calculate_layout(fallback_width, fallback_height)
                print(f"Using fallback window size: {fallback_width}x{fallback_height}")
            except pygame.error as e2:
                print(f"Fallback resize also failed: {e2}")
                return

        # Update renderer
        try:
            self.renderer = ChessRenderer(
                self.board_size_px, BOARD_SQUARES,
                self.window_width, self.window_height,
                board_x_offset=self.board_x_offset,
                heatmap_size_px=self.heatmap_size_px
            )

            # Update square size for pieces
            for piece in self.pieces: 
                piece.square_size = self.square_size_px

            # Recreate sliders AFTER renderer (uses renderer colors)
            self._create_pid_sliders()
            self.heatmap_surface = None
            self.heatmap_needs_update = True
        except Exception as e:
            print(f"Error recreating renderer: {e}")
            import traceback
            traceback.print_exc()

    def initialize_pieces(self):
        """Sets up the pieces in their starting positions."""
        self.pieces.clear(); self.captured_white.clear(); self.captured_black.clear()
        setup = { # (col, row)
            (0,0):(PieceColor.BLACK,PieceType.ROOK),(1,0):(PieceColor.BLACK,PieceType.KNIGHT), (2,0):(PieceColor.BLACK,PieceType.BISHOP),(3,0):(PieceColor.BLACK,PieceType.QUEEN), (4,0):(PieceColor.BLACK,PieceType.KING),(5,0):(PieceColor.BLACK,PieceType.BISHOP), (6,0):(PieceColor.BLACK,PieceType.KNIGHT),(7,0):(PieceColor.BLACK,PieceType.ROOK),
            (0,1):(PieceColor.BLACK,PieceType.PAWN),(1,1):(PieceColor.BLACK,PieceType.PAWN), (2,1):(PieceColor.BLACK,PieceType.PAWN),(3,1):(PieceColor.BLACK,PieceType.PAWN), (4,1):(PieceColor.BLACK,PieceType.PAWN),(5,1):(PieceColor.BLACK,PieceType.PAWN), (6,1):(PieceColor.BLACK,PieceType.PAWN),(7,1):(PieceColor.BLACK,PieceType.PAWN),
            (0,7):(PieceColor.WHITE,PieceType.ROOK),(1,7):(PieceColor.WHITE,PieceType.KNIGHT), (2,7):(PieceColor.WHITE,PieceType.BISHOP),(3,7):(PieceColor.WHITE,PieceType.QUEEN), (4,7):(PieceColor.WHITE,PieceType.KING),(5,7):(PieceColor.WHITE,PieceType.BISHOP), (6,7):(PieceColor.WHITE,PieceType.KNIGHT),(7,7):(PieceColor.WHITE,PieceType.ROOK),
            (0,6):(PieceColor.WHITE,PieceType.PAWN),(1,6):(PieceColor.WHITE,PieceType.PAWN), (2,6):(PieceColor.WHITE,PieceType.PAWN),(3,6):(PieceColor.WHITE,PieceType.PAWN), (4,6):(PieceColor.WHITE,PieceType.PAWN),(5,6):(PieceColor.WHITE,PieceType.PAWN), (6,6):(PieceColor.WHITE,PieceType.PAWN),(7,6):(PieceColor.WHITE,PieceType.PAWN), }
        for pos, (color, ptype) in setup.items():
            self.pieces.append(ChessPiece(color, ptype, pos, board_squares=BOARD_SQUARES, square_size=self.square_size_px, coil_grid_size=COIL_GRID_SIZE))

    def get_piece_at_square(self, square_col_row):
        """Finds the active piece whose position rounds to the given integer square."""
        target_col, target_row = square_col_row
        for piece in self.pieces:
            if piece.active:
                p_col_int = int(round(piece.position[0]))
                p_row_int = int(round(piece.position[1]))
                if p_col_int == target_col and p_row_int == target_row:
                    return piece
        return None

    def is_square_occupied(self, square_col_row, ignore_piece=None):
        """Checks if a square is occupied by an active piece (based on rounded position)."""
        piece = self.get_piece_at_square(square_col_row); return piece is not None and piece != ignore_piece

    def handle_click(self, pixel_pos):
        """Processes mouse clicks. Target is the center of the clicked square."""
        x_pix, y_pix = pixel_pos
        x_pix_board = x_pix - self.board_x_offset # Click position relative to board

        # Check if click is within the board area first
        if 0 <= x_pix_board < self.board_size_px and 0 <= y_pix < self.board_size_px:
            clicked_col_int = x_pix_board // self.square_size_px
            clicked_row_int = y_pix // self.square_size_px
            clicked_square = (clicked_col_int, clicked_row_int)
            # print(f"Board Clicked on square: {clicked_square}") # Reduced spam
            if self.move_in_progress: return
            piece_in_clicked_square = self.get_piece_at_square(clicked_square)
            if self.selected_piece:
                selected_square = (int(round(self.selected_piece.position[0])), int(round(self.selected_piece.position[1])))
                if clicked_square == selected_square:
                    print("Deselected piece.")
                    self.selected_piece = None; self.target_position = None
                else:
                    target_col_int, target_row_int = clicked_square
                    piece_at_target = self.get_piece_at_square(clicked_square)
                    if piece_at_target and piece_at_target.color == self.selected_piece.color:
                        print(f"Switched selection to {piece_at_target.symbol}")
                        self.selected_piece = piece_at_target; self.target_position = None
                    else:
                        self.target_position = (float(target_col_int), float(target_row_int))
                        print(f"Target set: {self.target_position} (center of square {clicked_square})")
                        path_is_clear = True
                        if self.selected_piece.piece_type != PieceType.KNIGHT:
                             if not self.is_path_clear(self.selected_piece.position, self.target_position):
                                  print("Path blocked. Attempting clearance...")
                                  if self.attempt_to_clear_path(self.selected_piece.position, self.target_position): print("Path cleared.")
                                  else: print("Clearance failed. Move cancelled."); path_is_clear = False; self.target_position = None
                        if path_is_clear and self.target_position is not None:
                             if piece_at_target: # Capture
                                 self.captured_piece = piece_at_target; self.captured_piece.active = False
                                 print(f"Capturing {self.captured_piece.symbol}")
                                 capture_target_col_row = self.capture_target_white if self.captured_piece.color == PieceColor.WHITE else self.capture_target_black
                                 start_rc = (target_row_int, target_col_int); target_rc = (int(capture_target_col_row[1]), int(capture_target_col_row[0]))
                                 path_nodes_rc = self.path_finder.find_capture_path(start_rc, target_rc, self.pieces, self.captured_piece)
                                 if path_nodes_rc: self.captured_piece.capture_path = [(float(c), float(r)) for r, c in path_nodes_rc]; print(f"Generated capture path with {len(self.captured_piece.capture_path)} points.")
                                 else: print("Warning: Could not generate capture path!"); self.captured_piece.capture_path = []
                                 if self.captured_piece.color == PieceColor.WHITE: self.captured_white.append(self.captured_piece)
                                 else: self.captured_black.append(self.captured_piece)
                             else: self.captured_piece = None
                             self.start_move()
            else: # No piece selected
                if piece_in_clicked_square:
                    self.selected_piece = piece_in_clicked_square; self.target_position = None; self.captured_piece = None; self.temporarily_moved_pieces = []
                    print(f"Selected {self.selected_piece.symbol} at position {self.selected_piece.position}")
                else: print("Clicked empty square."); self.selected_piece = None; self.target_position = None
        # Else: Click was outside board

    def is_path_clear(self, start_pos_col_row, end_pos_col_row):
        """Checks for obstructions along a straight line path (uses float coords)."""
        start_c, start_r = start_pos_col_row; end_c, end_r = end_pos_col_row
        path_vec = np.array([end_c - start_c, end_r - start_r]); path_len_sq = np.sum(path_vec**2)
        if path_len_sq < 0.01: return True
        path_len = np.sqrt(path_len_sq); direction = path_vec / path_len
        moving_radius = (self.selected_piece.diameter / self.square_size_px) * 0.5
        for piece in self.pieces:
            if piece.active and piece != self.selected_piece and piece not in self.temporarily_moved_pieces:
                vec_to_piece = piece.position - np.array(start_pos_col_row); projection_dist = np.dot(vec_to_piece, direction)
                if 0 < projection_dist < path_len:
                    perp_dist_sq = max(0, np.sum(vec_to_piece**2) - projection_dist**2); other_radius = (piece.diameter / self.square_size_px) * 0.5
                    min_dist_sq = (moving_radius + other_radius)**2 * 0.8 # Tolerance
                    if perp_dist_sq < min_dist_sq: return False
        return True

    def attempt_to_clear_path(self, start_pos_col_row, end_pos_col_row):
        """Tries to nudge one blocking piece aside to an adjacent empty square."""
        blocking_pieces = []; start_c, start_r = start_pos_col_row; end_c, end_r = end_pos_col_row
        path_vec = np.array([end_c - start_c, end_r - start_r]); path_len_sq = np.sum(path_vec**2)
        if path_len_sq < 0.01: return True
        path_len = np.sqrt(path_len_sq); direction = path_vec / path_len; moving_radius = (self.selected_piece.diameter / self.square_size_px) * 0.5
        for piece in self.pieces:
            if piece.active and piece != self.selected_piece and piece not in self.temporarily_moved_pieces:
                vec_to_piece = piece.position - np.array(start_pos_col_row); projection_dist = np.dot(vec_to_piece, direction)
                if 0 < projection_dist < path_len:
                    perp_dist_sq = max(0, np.sum(vec_to_piece**2) - projection_dist**2); other_radius = (piece.diameter / self.square_size_px) * 0.5
                    min_dist_sq = (moving_radius + other_radius)**2 * 0.8
                    if perp_dist_sq < min_dist_sq: blocking_pieces.append(piece)
        if not blocking_pieces: return True
        piece_to_move = next((p for p in blocking_pieces if p.piece_type == PieceType.PAWN), blocking_pieces[0])
        print(f"Attempting to move blocking piece: {piece_to_move.symbol}")
        original_pos = piece_to_move.position.copy(); original_square_int = (int(round(original_pos[0])), int(round(original_pos[1])))
        possible_nudges = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dc, dr in possible_nudges:
            target_col_int = original_square_int[0] + dc; target_row_int = original_square_int[1] + dr
            target_square_int = (target_col_int, target_row_int)
            if 0 <= target_col_int < BOARD_SQUARES and 0 <= target_row_int < BOARD_SQUARES:
                if not self.is_square_occupied(target_square_int, ignore_piece=piece_to_move):
                    print(f"Temporarily moving {piece_to_move.symbol} from {original_square_int} to {target_square_int}.")
                    piece_to_move.position_before_temp_move = original_pos
                    piece_to_move.position = np.array([float(target_col_int), float(target_row_int)])
                    piece_to_move.velocity = np.array([0.0, 0.0]); piece_to_move.path = [piece_to_move.position.copy()]
                    self.temporarily_moved_pieces.append(piece_to_move)
                    return self.is_path_clear(start_pos_col_row, end_pos_col_row)
        print(f"Could not find a clear adjacent square for {piece_to_move.symbol}."); return False

    def start_move(self):
        """Initiates the move sequence and resets PID controller state."""
        if self.selected_piece and self.target_position:
            print(f"Starting move for {self.selected_piece.symbol} to {self.target_position}")
            self.move_in_progress = True; self.move_complete = False; self.last_move_pid_force_mag = 0.0
            self.capture_step_index = 0; self.capture_path_finished = (self.captured_piece is None or not self.captured_piece.capture_path)
            self.pid_integral.fill(0.0); self.pid_previous_error.fill(0.0)
            self.coil_grid.reset(); self.hardware_controller.reset_all_coils(); self.heatmap_needs_update = True

    def _create_keep_out_mask(self):
        """Identifies coils too close to stationary pieces (for viz/hardware)."""
        blocked_coils = set(); keep_out_radius_coils = 1.5; keep_out_radius_sq = keep_out_radius_coils**2
        pieces_to_avoid = [p for p in self.pieces if p.active and p != self.selected_piece and p not in self.temporarily_moved_pieces]
        if self.captured_piece and not self.capture_path_finished: pieces_to_avoid.append(self.captured_piece)
        for piece in pieces_to_avoid:
            piece_coil_c, piece_coil_r = piece.get_coil_position()
            min_r = max(0, int(math.floor(piece_coil_r - keep_out_radius_coils))); max_r = min(COIL_GRID_SIZE, int(math.ceil(piece_coil_r + keep_out_radius_coils)))
            min_c = max(0, int(math.floor(piece_coil_c - keep_out_radius_coils))); max_c = min(COIL_GRID_SIZE, int(math.ceil(piece_coil_c + keep_out_radius_coils)))
            for r in range(min_r, max_r):
                for c in range(min_c, max_c):
                    if (r - piece_coil_r)**2 + (c - piece_coil_c)**2 < keep_out_radius_sq: blocked_coils.add((r, c))
        return blocked_coils

    def update_move(self, dt):
        """Updates piece physics using direct PID force with dynamic gain scaling."""
        if not self.move_in_progress:
             self.last_move_pid_force_mag = 0.0; self.current_total_sim_amps = self.coil_grid.calculate_total_current()
             return
        effective_dt = dt * self.simulation_speed
        if effective_dt <= 0: return
        if self.selected_piece and self.target_position:
            current_pos = self.selected_piece.position.copy(); current_vel = self.selected_piece.velocity.copy()
            target_pos = np.array(self.target_position); error = target_pos - current_pos
            distance_to_target = np.linalg.norm(error)
            # Dynamic Gain Scaling
            gain_scale_distance = 1.5; min_gain_scale = 0.35
            scale_factor = np.clip(distance_to_target / gain_scale_distance, min_gain_scale, 1.0)
            effective_kp = self.pid_kp * scale_factor; effective_kd = self.pid_kd * scale_factor; effective_ki = self.pid_ki
            # PID Terms
            p_term = effective_kp * error
            if effective_ki > 1e-6:
                self.pid_integral += error * effective_dt; integral_mag = np.linalg.norm(self.pid_integral)
                if integral_mag > self.pid_integral_max: self.pid_integral = self.pid_integral * (self.pid_integral_max / integral_mag)
                i_term = effective_ki * self.pid_integral
            else: i_term = np.array([0.0, 0.0])
            d_term = -effective_kd * current_vel
            terminal_zone = 0.4
            if distance_to_target < terminal_zone:
                terminal_factor = 1.0 - (distance_to_target / terminal_zone)
                terminal_damping_force = -self.terminal_damping * current_vel * terminal_factor
                d_term += terminal_damping_force
            # Total PID Force
            pid_force = p_term + i_term + d_term; self.last_move_pid_force_mag = np.linalg.norm(pid_force)
            max_pid_force = 5000.0
            if self.last_move_pid_force_mag > max_pid_force:
                pid_force = pid_force * (max_pid_force / self.last_move_pid_force_mag); self.last_move_pid_force_mag = max_pid_force
                if self.debug_mode: print("  PID Force Clamped!")
            # Debug Output
            if self.debug_mode:
                print(f"\n--- Update Step dt={effective_dt:.4f} ---"); print(f"  Piece: {self.selected_piece.symbol} Pos:({current_pos[0]:.2f},{current_pos[1]:.2f}) Vel:({current_vel[0]:.2f},{current_vel[1]:.2f}) Dist:{distance_to_target:.3f}")
                print(f"  Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f})"); print(f"  PID Error: ({error[0]:.3f},{error[1]:.3f})"); print(f"  Gain Scale Factor: {scale_factor:.2f} (Min: {min_gain_scale})")
                print(f"  PID Terms: P:({p_term[0]:.1f},{p_term[1]:.1f}) I:({i_term[0]:.1f},{i_term[1]:.1f}) D:({d_term[0]:.1f},{d_term[1]:.1f})"); print(f"  PID Force: ({pid_force[0]:.2f},{pid_force[1]:.2f}) Mag: {self.last_move_pid_force_mag:.2f}")
            # Stop Condition
            stop_threshold = 0.02; velocity_threshold = 0.03; force_threshold = 0.1
            settled_condition = (self.last_move_pid_force_mag < force_threshold and np.linalg.norm(current_vel) < velocity_threshold * 2)
            move_finished = (distance_to_target < stop_threshold and np.linalg.norm(current_vel) < velocity_threshold) or \
                            (distance_to_target < stop_threshold * 2 and settled_condition)
            if move_finished:
                final_pos_before_snap = self.selected_piece.position.copy(); final_vel_before_snap = self.selected_piece.velocity.copy()
                self.selected_piece.position = target_pos.copy(); self.selected_piece.velocity.fill(0.0); self.pid_integral.fill(0.0)
                if self.debug_mode: print(f"STOP Condition Met: Dist {distance_to_target:.3f} (<{stop_threshold}), Vel {np.linalg.norm(current_vel):.3f} (<{velocity_threshold}), Force {self.last_move_pid_force_mag:.2f} (<{force_threshold})")
                print(f"Move complete. Snapped from ({final_pos_before_snap[0]:.3f},{final_pos_before_snap[1]:.3f}) Vel ({final_vel_before_snap[0]:.3f},{final_vel_before_snap[1]:.3f})"); print(f"Final Position: ({self.selected_piece.position[0]},{self.selected_piece.position[1]}) Velocity: ({self.selected_piece.velocity[0]},{self.selected_piece.velocity[1]})")
                self.move_in_progress = False; self.move_complete = True; self.target_position = None; self.last_move_pid_force_mag = 0.0
                self.coil_grid.reset(); self.hardware_controller.reset_all_coils(); self.heatmap_needs_update = True; self.current_total_sim_amps = 0.0
                for piece in self.temporarily_moved_pieces: piece.return_from_temporary_move()
                self.temporarily_moved_pieces = []
            else: # Move In Progress
                self.selected_piece.apply_force(pid_force, effective_dt)
                if self.debug_mode: print(f"  End Pos:({self.selected_piece.position[0]:.2f},{self.selected_piece.position[1]:.2f}) End Vel:({self.selected_piece.velocity[0]:.2f},{self.selected_piece.velocity[1]:.2f})")
                # Update Coil Simulation & Hardware
                self.field_update_timer += effective_dt
                if self.field_update_timer >= self.field_update_interval:
                    self.field_update_timer = 0; blocked_coils_set = self._create_keep_out_mask(); current_coil_pos = self.selected_piece.get_coil_position(); target_coil_pos = tuple(np.array(target_pos) * (COIL_GRID_SIZE / BOARD_SQUARES))
                    dx_board = target_pos[0] - current_pos[0]; dy_board = target_pos[1] - current_pos[1]
                    is_knight_shape = (abs(round(dx_board)) == 1 and abs(round(dy_board)) == 2) or (abs(round(dx_board)) == 2 and abs(round(dy_board)) == 1)
                    chosen_pattern = self.current_pattern; straight_threshold = 0.1
                    if self.selected_piece.piece_type == PieceType.KNIGHT:
                         if is_knight_shape and self.current_pattern in ["knight", "directed"]: chosen_pattern = "knight"
                    elif abs(dx_board) < straight_threshold and abs(dy_board) > straight_threshold: chosen_pattern = "straight_vertical"
                    elif abs(dy_board) < straight_threshold and abs(dx_board) > straight_threshold: chosen_pattern = "straight_horizontal"
                    else: chosen_pattern="directed"
                    scale_distance_threshold = 1.2; min_viz_scale = 0.1
                    if distance_to_target < scale_distance_threshold: ratio = distance_to_target / scale_distance_threshold; viz_scale_factor = min_viz_scale + (1.0 - min_viz_scale) * (ratio**2); viz_scale_factor = max(min_viz_scale, viz_scale_factor)
                    else: viz_scale_factor = 1.0
                    current_intensity = 100 * viz_scale_factor
                    self.coil_grid.activate_coil_pattern(pattern_type=chosen_pattern, position=current_coil_pos, target=target_coil_pos, intensity=current_intensity, radius=4, blocked_coils=blocked_coils_set)
                    self.current_total_sim_amps = self.coil_grid.calculate_total_current(); self.coil_grid.update_magnetic_field(); self.heatmap_needs_update = True
                    self.hardware_controller.apply_state(self.coil_grid.coil_power, self.coil_grid.coil_current)
        # Update Captured Piece Movement
        if self.captured_piece and not self.capture_path_finished:
            node_reached_or_finished = self.captured_piece.follow_capture_path(self.capture_step_index)
            if node_reached_or_finished:
                if self.capture_step_index < len(self.captured_piece.capture_path) - 1: self.capture_step_index += 1
                else: self.capture_path_finished = True; print(f"Capture movement finished for {self.captured_piece.symbol}.")

    def reset(self):
        """Resets the board, pieces, PID state (applies slider values), and hardware."""
        print("Resetting board...");
        if WIDGETS_AVAILABLE:
            self._update_slider_textboxes() # Get latest slider values before applying
            print(f"Applying PID values from sliders: Kp={self.temp_pid_kp:.1f}, Ki={self.temp_pid_ki:.1f}, Kd={self.temp_pid_kd:.1f}")
            self.pid_kp = self.temp_pid_kp; self.pid_ki = self.temp_pid_ki; self.pid_kd = self.temp_pid_kd
        else: print("Widgets not available, using default PID gains.")
        self.print_current_pid_settings("Active after Reset")
        self.initialize_pieces(); self.selected_piece = None; self.target_position = None
        self.move_in_progress = False; self.move_complete = False; self.last_move_pid_force_mag = 0.0; self.current_total_sim_amps = 0.0
        self.captured_piece = None; self.capture_step_index = 0; self.capture_path_finished = False; self.temporarily_moved_pieces = []
        self.pid_integral.fill(0.0); self.pid_previous_error.fill(0.0)
        self.coil_grid.reset(); self.hardware_controller.reset_all_coils(); self.heatmap_needs_update = True
        print("Board reset complete.")

    def cycle_pattern(self):
        """Cycles through available coil patterns (for viz/hardware)."""
        current_index = self.patterns.index(self.current_pattern); next_index = (current_index + 1) % len(self.patterns)
        self.current_pattern = self.patterns[next_index]; print(f"Switched to pattern: {self.current_pattern}")

    def load_heatmap(self):
        """Generates and loads the heatmap image if needed."""
        if self.heatmap_needs_update or self.heatmap_surface is None:
            heatmap_path = self.coil_grid.plot_heatmap(filename="field_heatmap.png")
            if heatmap_path:
                try:
                    # Check if file exists before loading
                    if not os.path.exists(heatmap_path):
                         print(f"Error: Heatmap file not found after generation: {heatmap_path}")
                         self.heatmap_surface = None; return
                    loaded_surface = pygame.image.load(heatmap_path)
                    if loaded_surface.get_alpha() is None: loaded_surface = loaded_surface.convert()
                    else: loaded_surface = loaded_surface.convert_alpha()
                    target_w = self.heatmap_size_px; target_h = self.heatmap_size_px
                    self.heatmap_surface = pygame.transform.smoothscale(loaded_surface, (target_w, target_h))
                    self.heatmap_needs_update = False
                except pygame.error as e:
                     print(f"Error loading Pygame heatmap image '{heatmap_path}': {e}")
                     self.heatmap_surface = None
                except Exception as e:
                     print(f"Error loading/scaling heatmap image '{heatmap_path}': {e}");
                     self.heatmap_surface = None
            else:
                 # Prevent repeated generation attempts if failed first time
                 # print("Heatmap generation failed.") # Reduced spam
                 self.heatmap_surface = None # Ensure it's None if generation fails
                 self.heatmap_needs_update = False # Avoid retrying every frame

    def run(self):
        """Main simulation loop."""
        running = True
        while running:
            dt_sec = min(self.clock.tick(FPS) / 1000.0, 0.1)
            events = pygame.event.get()

            # --- Update Widgets First ---
            if WIDGETS_AVAILABLE:
                # Update widget states based on events
                # This call processes events for the widgets
                pygame_widgets.update(events)
                # Update internal temp values and textbox displays AFTER processing events
                self._update_slider_textboxes()

            # --- Handle Game Events ---
            for event in events:
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Check if the click was handled by a widget; if not, process board click
                    # Note: pygame_widgets doesn't easily provide a 'handled' flag.
                    # We rely on checking coordinates in handle_click.
                    if event.button == 1:
                         self.handle_click(event.pos)
                    # Add right-click to cancel selection
                    elif event.button == 3:  # Right mouse button
                         if self.selected_piece:
                             print("Selection canceled (right-click).")
                             self.selected_piece = None
                             self.target_position = None
                             self.temporarily_moved_pieces = []
                             # Return pieces moved aside during clearance to original positions
                             for piece in self.temporarily_moved_pieces:
                                 piece.return_from_temporary_move()
                elif event.type == pygame.KEYDOWN:
                    key = event.key
                    if key == pygame.K_r: self.reset()
                    elif key == pygame.K_m: self.cycle_pattern()
                    elif key == pygame.K_PLUS or key == pygame.K_EQUALS: self.simulation_speed = min(5.0, self.simulation_speed + 0.2); print(f"Sim speed: {self.simulation_speed:.1f}x")
                    elif key == pygame.K_MINUS: self.simulation_speed = max(0.1, self.simulation_speed - 0.2); print(f"Sim speed: {self.simulation_speed:.1f}x")
                    elif key == pygame.K_c: self.show_coils = not self.show_coils
                    elif key == pygame.K_f: self.show_field = not self.show_field
                    elif key == pygame.K_p: self.show_paths = not self.show_paths
                    elif key == pygame.K_h: self.show_heatmap = not self.show_heatmap; self.heatmap_needs_update = True # Force regen on toggle
                    elif key == pygame.K_d: self.debug_mode = not self.debug_mode; print(f"Debug mode: {self.debug_mode}")
                    elif key == pygame.K_x: self.show_center_markers = not self.show_center_markers; print(f"Center markers: {self.show_center_markers}")
                    elif key == pygame.K_y: self.renderer.show_position_dots = not self.renderer.show_position_dots; print(f"Position dots: {self.renderer.show_position_dots}")
                    # Add Escape key to cancel selection
                    elif key == pygame.K_ESCAPE:
                        if self.selected_piece:
                            print("Selection canceled (Escape key).")
                            self.selected_piece = None
                            self.target_position = None
                            self.temporarily_moved_pieces = []
                            # Return pieces moved aside during clearance to original positions
                            for piece in self.temporarily_moved_pieces:
                                piece.return_from_temporary_move()
                        else:
                            running = False  # Only quit if no selection active
                elif event.type == pygame.VIDEORESIZE: self.handle_resize(event.w, event.h)

            # --- Update Simulation Logic ---
            self.update_move(dt_sec)

            # --- Drawing ---
            self.screen.fill(self.renderer.DARK_GRAY)
            # Heatmap
            if self.show_heatmap: self.load_heatmap(); self.renderer.draw_heatmap_beside_board(self.screen, self.heatmap_surface)
            else: self.renderer.draw_heatmap_beside_board(self.screen, None)
            # Board
            self.renderer.draw_board(self.screen)
            # Optional Center Markers
            if self.show_center_markers:
                for row in range(BOARD_SQUARES):
                    for col in range(BOARD_SQUARES):
                        px, py = self.renderer.board_to_pixel((col, row))
                        self.renderer.draw_center_marker(self.screen, px, py)
            # Optional Coil/Field Viz
            if self.show_coils: self.coil_grid.draw(self.screen, self.board_size_px, x_offset=self.board_x_offset)
            if self.show_field: self.coil_grid.draw_field_overlay(self.screen, self.board_size_px, x_offset=self.board_x_offset)
            # Paths
            if self.show_paths: self.renderer.draw_paths(self.screen, self.pieces, self.selected_piece)
            # Pieces
            self.renderer.draw_pieces(self.screen, self.pieces, self.selected_piece)
            # Control Panel & Capture Area
            info_dict = {
                'selected_piece': self.selected_piece, 'target_position': self.target_position if self.move_in_progress else None,
                'move_in_progress': self.move_in_progress, 'move_complete': self.move_complete,
                'show_coils': self.show_coils, 'show_field': self.show_field, 'show_paths': self.show_paths,
                'show_heatmap': self.show_heatmap, 'show_center_markers': self.show_center_markers,
                'current_pattern': self.current_pattern, 'simulation_speed': self.simulation_speed,
                'debug_mode': self.debug_mode, 'pid_gains_active': (self.pid_kp, self.pid_ki, self.pid_kd),
                'pid_gains_temp': (self.temp_pid_kp, self.temp_pid_ki, self.temp_pid_kd), # Pass temp values too
                'pid_force_mag': self.last_move_pid_force_mag, 'sim_current_amps': self.current_total_sim_amps,
                'captured_white': self.captured_white, 'captured_black': self.captured_black,
            }
            # Draw panel background and text info FIRST
            self.renderer.draw_controls(self.screen, info_dict, panel_x=self.panel_x, sliders_active=WIDGETS_AVAILABLE)

            # --- Explicitly Draw Widgets (Needed!) ---
            # Widgets need to be drawn after their background surface is drawn.
            # pygame_widgets.update handles events, but drawing is often separate.
            # This should make the sliders appear on top of the panel background.
            if WIDGETS_AVAILABLE:
                 # Manually redraw widgets on the main screen surface each frame
                 # This assumes widgets were created with self.screen
                 for slider in self.pid_sliders.values(): slider.draw()
                 for textbox in self.pid_textboxes.values(): textbox.draw()

            pygame.display.flip() # Update screen
            if self.move_complete: self.move_complete = False # Reset flag

        print("Simulation loop ended.")