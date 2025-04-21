# visualization.py

import pygame
import numpy as np
import math
from chess_pieces import ChessPiece, PieceColor, PieceType, PIECE_SYMBOLS # Import piece info

# Attempt to import pygame_widgets (optional dependency)
try:
    import pygame_widgets
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False


class ChessRenderer:
    """Handles all Pygame rendering for the chess simulation."""

    def __init__(self, board_size_px, squares, window_width, window_height, board_x_offset, heatmap_size_px): # Added heatmap_size_px
        self.board_size_px = board_size_px
        self.squares = squares
        self.square_size_px = board_size_px // squares
        self.window_width = window_width
        self.window_height = window_height
        self.board_x_offset = board_x_offset # Board's X position offset for heatmap
        self.heatmap_size_px = heatmap_size_px # Store heatmap size
        self.panel_x = board_x_offset + board_size_px # Default panel start X

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)
        self.LIGHT_SQUARE = (240, 217, 181)
        self.DARK_SQUARE = (181, 136, 99)
        self.LIGHT_GRAY = (211, 211, 211) # Lighter gray for panel
        self.DARK_GRAY = (100, 100, 100)
        self.HIGHLIGHT = (124, 252, 0) # Green highlight for selection
        self.PATH_COLOR = (0, 0, 255, 150) # Blueish semi-transparent for path
        self.SEL_PATH_COLOR = (0, 255, 0, 150) # Greenish semi-transparent for selected path
        self.CAPTURE_PATH_COLOR = (255, 0, 0, 150) # Reddish semi-transparent for capture path
        self.CENTER_MARKER_COLOR = (255, 0, 255) # Magenta for center markers
        self.POSITION_DOT_COLOR = (255, 0, 0) # Red for mathematical position indicators

        # Initialize fonts
        try:
            pygame.font.init() # Ensure font module is initialized
            self.font = pygame.font.SysFont('segoeui', 24) # Main font
            self.small_font = pygame.font.SysFont('segoeui', 18) # Smaller font for panel info
            self.very_small_font = pygame.font.SysFont('segoeui', 14) # For labels like slider names
            self.coord_font = pygame.font.SysFont('consolas', 12) # Tiny font for coords
            # Piece font created dynamically
        except pygame.error as e:
            print(f"Font Error: {e}. Using default fonts.")
            self.font = pygame.font.Font(None, 30)
            self.small_font = pygame.font.Font(None, 24)
            self.very_small_font = pygame.font.Font(None, 20)
            self.coord_font = pygame.font.Font(None, 16)

        self.show_position_dots = True  # Default to True for better debugging


    def draw_board(self, surface):
        """Draw the chessboard grid and labels."""
        # Draw squares
        for r in range(self.squares):
            for c in range(self.squares):
                color = self.LIGHT_SQUARE if (r + c) % 2 == 0 else self.DARK_SQUARE
                pygame.draw.rect(surface, color,
                                 (c * self.square_size_px + self.board_x_offset,
                                  r * self.square_size_px,
                                  self.square_size_px, self.square_size_px))

        # Draw rank/file labels (use smaller font)
        label_color = self.WHITE
        for i in range(self.squares):
            # Files (a-h) below board
            file_txt = self.very_small_font.render(chr(ord('a') + i), True, label_color)
            file_rect = file_txt.get_rect(center=(i * self.square_size_px + self.square_size_px // 2 + self.board_x_offset,
                                                  self.board_size_px + 10)) # Closer to board
            surface.blit(file_txt, file_rect)
            # Ranks (1-8) left of board
            rank_txt = self.very_small_font.render(str(self.squares - i), True, label_color) # 8 at top
            rank_rect = rank_txt.get_rect(center=(self.board_x_offset - 10, i * self.square_size_px + self.square_size_px // 2))
            surface.blit(rank_txt, rank_rect)

    def draw_center_marker(self, surface, x, y, size=5):
        """Draw a marker at the exact center of a square for debugging purposes."""
        pygame.draw.line(surface, self.CENTER_MARKER_COLOR, (x-size, y), (x+size, y), 2)
        pygame.draw.line(surface, self.CENTER_MARKER_COLOR, (x, y-size), (x, y+size), 2)
        pygame.draw.circle(surface, self.CENTER_MARKER_COLOR, (x, y), 2)


    def board_to_pixel(self, board_pos):
        """Convert board coordinates (col, row floats) to pixel coordinates (center of square)."""
        col, row = board_pos
        px = col * self.square_size_px + self.square_size_px // 2 + self.board_x_offset
        py = row * self.square_size_px + self.square_size_px // 2
        return int(px), int(py)


    def draw_piece(self, surface: pygame.Surface, piece: ChessPiece, selected=False):
        """Draws a single chess piece using its properties."""
        is_capturing = (not piece.active and piece.capture_path) # Check if being captured
        if not piece.active and not is_capturing: return # Don't draw inactive unless being captured

        # --- Get Pixel Position ---
        x_center_rel, y_center_rel = piece.get_pixel_position()
        x_center = x_center_rel + self.board_x_offset # Apply board offset HERE
        y_center = y_center_rel
        # --- --------------- ---

        symbol = piece.symbol
        text_color = self.WHITE if piece.color == PieceColor.WHITE else self.BLACK
        # Make captured pieces semi-transparent during animation
        alpha = 255 if piece.active else 150

        # Calculate dynamic font size
        base_size = self.square_size_px * 0.75 # Base size factor
        diameter_scale = max(0.8, min(1.2, piece.diameter / 40.0)) # Clamp scale factor
        size = int(base_size * diameter_scale)
        size = max(15, min(size, int(self.square_size_px * 0.9))) # Clamp absolute size

        try:
            piece_font = pygame.font.SysFont('segoeuisymbol', size)
        except Exception:
            piece_font = pygame.font.Font(None, size) # Fallback

        # Render with alpha if needed
        piece_text_surf = piece_font.render(symbol, True, text_color)
        if alpha < 255:
            piece_text_surf.set_alpha(alpha)

        text_rect = piece_text_surf.get_rect(center=(x_center, y_center))

        # Draw selection highlight UNDER the piece text
        if selected and piece.active:
            highlight_radius = int((piece.diameter / 2) * (self.square_size_px / 40) * 1.2) # Scale radius approx with square size
            highlight_radius = max(5, highlight_radius) # Min radius
            pygame.draw.circle(surface, self.HIGHLIGHT, (x_center, y_center), highlight_radius, 3)

        # Draw the piece symbol
        surface.blit(piece_text_surf, text_rect)

        # If enabled, draw a small dot at the exact mathematical position
        if self.show_position_dots and piece.active:
            pygame.draw.circle(surface, self.POSITION_DOT_COLOR, (x_center, y_center), 3)

            # Also draw the board coordinates near the dot (using tiny font)
            col, row = piece.position
            coord_text = self.coord_font.render(f"({col:.1f},{row:.1f})", True, self.POSITION_DOT_COLOR)
            surface.blit(coord_text, (x_center + 5, y_center - 15)) # Adjust position


    def draw_pieces(self, surface: pygame.Surface, pieces: list[ChessPiece], selected_piece: ChessPiece = None):
        """Draws all pieces, handling active, inactive (capturing), and selected."""
        # Draw inactive pieces first (those being captured)
        for piece in pieces:
            if not piece.active and piece.capture_path:
                self.draw_piece(surface, piece, selected=False) # Draw semi-transparent

        # Draw active, non-selected pieces
        for piece in pieces:
            if piece.active and piece != selected_piece:
                self.draw_piece(surface, piece, selected=False)

        # Draw selected piece last (on top)
        if selected_piece and selected_piece.active:
            self.draw_piece(surface, selected_piece, selected=True)


    def draw_paths(self, surface: pygame.Surface, pieces: list[ChessPiece], selected_piece: ChessPiece = None):
        """Draws the movement paths for pieces."""
        for piece in pieces:
            pixel_points = []
            path_to_draw = None
            path_color = self.PATH_COLOR

            if piece.active and len(piece.path) > 1:
                path_to_draw = piece.path
                path_color = self.SEL_PATH_COLOR if piece == selected_piece else self.PATH_COLOR
            elif not piece.active and piece.capture_path: # Draw capture path for inactive
                path_to_draw = piece.capture_path
                path_color = self.CAPTURE_PATH_COLOR

            if path_to_draw:
                # Convert path points (col, row) to pixel coordinates
                for pos in path_to_draw:
                    px, py = self.board_to_pixel(pos) # Use helper function
                    pixel_points.append((px, py))

                if len(pixel_points) > 1:
                    pygame.draw.lines(surface, path_color, False, pixel_points, 2)
                    # Mark end point of capture path differently
                    if not piece.active and piece.capture_path:
                         pygame.draw.circle(surface, self.RED, pixel_points[-1], 5)


    def draw_controls(self, surface: pygame.Surface, info: dict, panel_x, sliders_active=False):
        """Draws the control panel with info, stats, and placeholders for sliders."""
        self.panel_x = panel_x # Update panel_x based on current layout
        panel_width = self.window_width - self.panel_x
        # Background
        pygame.draw.rect(surface, self.LIGHT_GRAY, (self.panel_x, 0, panel_width, self.window_height))

        # --- Title ---
        title_text = self.font.render("EM Chess Control", True, self.BLACK)
        title_rect = title_text.get_rect(center=(self.panel_x + panel_width // 2, 30))
        surface.blit(title_text, title_rect)

        # --- Info Area Start ---
        info_y = 70
        line_height = 20 # Reduced line height slightly
        text_x = self.panel_x + 15

        def draw_text(text, y, color=self.BLACK, font=self.small_font):
            txt_surf = font.render(text, True, color)
            surface.blit(txt_surf, (text_x, y))
            return y + line_height

        # --- Basic Controls ---
        info_y = draw_text("[Click] Piece/Target", info_y)
        info_y = draw_text("[R] Reset & Apply PID", info_y) # Updated Reset description
        info_y = draw_text("[M] Cycle Pattern", info_y)
        info_y = draw_text("[+/-] Speed", info_y)
        info_y = draw_text("[Esc] Quit", info_y)
        info_y += 5

        # --- Toggles ---
        col1_x = text_x
        col2_x = text_x + (panel_width // 2) - 10

        def draw_toggle(key, label, value, y, x):
            state_text = "ON" if value else "OFF"
            color = self.GREEN if value else self.RED
            full_text = f"[{key}] {label}: {state_text}"
            txt_surf = self.very_small_font.render(full_text, True, color)
            surface.blit(txt_surf, (x, y))

        draw_toggle('C', "Coils", info.get('show_coils', False), info_y, col1_x)
        draw_toggle('F', "Field", info.get('show_field', False), info_y + line_height, col1_x)
        draw_toggle('P', "Paths", info.get('show_paths', True), info_y + 2*line_height, col1_x)

        draw_toggle('H', "Heatmap", info.get('show_heatmap', False), info_y, col2_x)
        draw_toggle('X', "Centers", info.get('show_center_markers', False), info_y + line_height, col2_x)
        draw_toggle('Y', "PosDots", self.show_position_dots, info_y + 2*line_height, col2_x)

        info_y += 3*line_height + 5 # Advance past toggles

        # --- Sim Info ---
        info_y = draw_text(f"Pattern: {info.get('current_pattern', 'N/A').upper()}", info_y)
        info_y = draw_text(f"Speed: {info.get('simulation_speed', 1.0):.1f}x", info_y)
        active_gains = info.get('pid_gains_active', (0,0,0))
        info_y = draw_text(f"Active PID: {active_gains[0]:.1f}/{active_gains[1]:.1f}/{active_gains[2]:.1f}", info_y, font=self.very_small_font)
        # Display temporary PID values from sliders if available
        if sliders_active and info.get('pid_gains_temp'):
             temp_gains = info.get('pid_gains_temp', active_gains)
             if temp_gains != active_gains: # Only show if different
                 info_y = draw_text(f"Sliders-> R: {temp_gains[0]:.1f}/{temp_gains[1]:.1f}/{temp_gains[2]:.1f}", info_y, self.BLUE, font=self.very_small_font)

        dbg_status = info.get('debug_mode', False)
        info_y = draw_text(f"[{'D'}] Debug Out: {'ON' if dbg_status else 'OFF'}", info_y, font=self.very_small_font)
        info_y += 5

        # --- Move Status & Stats ---
        selected = info.get('selected_piece')
        target = info.get('target_position') # Target only shown during move
        status_color = self.BLACK
        status_text = "Status: Select Piece"
        if info.get('move_in_progress', False):
            status_text = "Status: MOVE IN PROGRESS"
            status_color = self.ORANGE
        elif info.get('move_complete', False): # Use move_complete flag briefly set by ChessBoard
             status_text = "Status: Move Complete"
             status_color = self.GREEN
        elif selected:
             status_text = "Status: Target?"

        info_y = draw_text(status_text, info_y, status_color)

        if selected:
            sel_col, sel_row = selected.position
            info_y = draw_text(f"Selected: {selected.symbol} @ ({sel_col:.2f},{sel_row:.2f})", info_y, self.BLUE)
            if target:
                info_y = draw_text(f"Target: ({target[0]:.1f}, {target[1]:.1f})", info_y, self.GREEN)

        # Display Force and Current
        force_mag = info.get('pid_force_mag', 0.0)
        sim_amps = info.get('sim_current_amps', 0.0)
        info_y = draw_text(f"PID Force: {force_mag:.1f}", info_y)
        info_y = draw_text(f"Sim Current: {sim_amps:.2f} A", info_y)
        info_y += 10

        # --- PID Sliders Area ---
        if sliders_active:
            slider_y_start = info_y + 20 # Leave more space before sliders
            slider_spacing = 70
            label_y_offset = -16 # Position label above slider

            # Kp
            kp_label = self.very_small_font.render("Kp (Proportional)", True, self.BLACK)
            surface.blit(kp_label, (text_x, slider_y_start + label_y_offset))
            # Ki
            ki_label = self.very_small_font.render("Ki (Integral)", True, self.BLACK)
            surface.blit(ki_label, (text_x, slider_y_start + slider_spacing + label_y_offset))
            # Kd
            kd_label = self.very_small_font.render("Kd (Derivative)", True, self.BLACK)
            surface.blit(kd_label, (text_x, slider_y_start + 2*slider_spacing + label_y_offset))

            # Note: Sliders and TextBoxes themselves are drawn by pygame_widgets.update() in the main loop
            info_y = slider_y_start + 3 * slider_spacing # Update info_y to after sliders

        # --- Capture Area ---
        # Position capture area below sliders or stats
        self.draw_capture_area(surface, info.get('captured_white',[]), info.get('captured_black',[]), panel_x, info_y + 10)


    def draw_capture_area(self, surface: pygame.Surface, captured_white: list, captured_black: list, panel_x, y_start_offset):
        """Draws the display for captured pieces."""
        self.panel_x = panel_x # Ensure panel_x is up-to-date
        area_x = self.panel_x + 15
        area_width = self.window_width - self.panel_x - 30
        icon_size = 20 # Make icons slightly smaller
        spacing = 3
        # Position capture areas based on y_start_offset
        white_y_start = y_start_offset
        black_y_start = y_start_offset + 50 # Space between the two lists

        # Helper function to draw list of pieces
        def draw_captured_list(y_start, title, pieces_list, text_color):
            # Check if area is still visible
            if y_start > self.window_height - 20: return

            title_surf = self.small_font.render(title, True, self.BLACK)
            surface.blit(title_surf, (area_x, y_start - 20)) # Title above icons
            current_x = area_x
            current_y = y_start
            max_x = area_x + area_width - icon_size # Wrap condition
            try:
                cap_font = pygame.font.SysFont('segoeuisymbol', icon_size)
            except:
                cap_font = pygame.font.Font(None, icon_size) # Fallback

            for piece in pieces_list:
                 # Check if icon drawing goes off bottom
                 if current_y > self.window_height - icon_size: break
                 symbol_text = cap_font.render(piece.symbol, True, text_color)
                 surface.blit(symbol_text, (current_x, current_y))
                 current_x += icon_size + spacing
                 # Simple wrap logic
                 if current_x > max_x:
                     current_x = area_x
                     current_y += icon_size + spacing # Move to next row (basic)

        # Draw the lists
        draw_captured_list(white_y_start, "Captured White:", captured_white, self.WHITE)
        draw_captured_list(black_y_start, "Captured Black:", captured_black, self.BLACK)


    def draw_heatmap_beside_board(self, surface: pygame.Surface, heatmap_image: pygame.Surface = None):
        """Draw the heatmap to the left of the board."""
        heatmap_area_x = 0
        heatmap_area_y = 0
        # Use stored heatmap size
        heatmap_area_width = self.heatmap_size_px
        heatmap_area_height = self.heatmap_size_px # Keep it square

        # Background for heatmap area
        pygame.draw.rect(surface, self.DARK_GRAY,
                         (heatmap_area_x, heatmap_area_y, heatmap_area_width, heatmap_area_height))

        # Title (smaller)
        title_text = self.very_small_font.render("Field Strength Heatmap (H)", True, self.WHITE)
        title_rect = title_text.get_rect(midtop=(heatmap_area_x + heatmap_area_width // 2, 5))
        surface.blit(title_text, title_rect)

        if heatmap_image:
            # Image should already be scaled correctly by load_heatmap
            img_rect = heatmap_image.get_rect()
            # Center the pre-scaled image within the area
            display_x = heatmap_area_x + (heatmap_area_width - img_rect.width) // 2
            display_y = heatmap_area_y + 25 # Offset from top for title
            surface.blit(heatmap_image, (display_x, display_y))
        else:
            # Placeholder text if no heatmap image
            ph_text = self.small_font.render("Heatmap Off / Not Generated", True, self.YELLOW)
            ph_rect = ph_text.get_rect(center=(heatmap_area_x + heatmap_area_width // 2,
                                               heatmap_area_y + heatmap_area_height // 2))
            surface.blit(ph_text, ph_rect)

    # highlight_square method (if needed elsewhere, keep it)
    def highlight_square(self, surface, row, col, color=None):
        """Highlight a square on the board."""
        if color is None: color = self.HIGHLIGHT
        rect = pygame.Rect(col * self.square_size_px + self.board_x_offset, row * self.square_size_px,
                           self.square_size_px, self.square_size_px)
        s = pygame.Surface((self.square_size_px, self.square_size_px), pygame.SRCALPHA)
        s.fill((*color[:3], 100)) # Add alpha
        surface.blit(s, (rect.x, rect.y))