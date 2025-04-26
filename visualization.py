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

    def __init__(self, board_size_px, squares, window_width, window_height, board_x_offset, heatmap_size_px):
        self.board_size_px = board_size_px
        self.squares = squares
        self.square_size_px = board_size_px // squares
        self.window_width = window_width
        self.window_height = window_height
        self.board_x_offset = board_x_offset
        self.heatmap_size_px = heatmap_size_px
        self.panel_x = board_x_offset + board_size_px
    
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
        self.LIGHT_GRAY = (211, 211, 211)
        self.DARK_GRAY = (50, 50, 50)  # Darker gray for better contrast
        self.HIGHLIGHT = (124, 252, 0)
        self.PATH_COLOR = (0, 0, 255, 150)
        self.SEL_PATH_COLOR = (0, 255, 0, 150)
        self.CAPTURE_PATH_COLOR = (255, 0, 0, 150)
        self.CENTER_MARKER_COLOR = (255, 0, 255)
        self.POSITION_DOT_COLOR = (255, 0, 0)
    
        # Initialize fonts
        try:
            pygame.font.init()
            self.font = pygame.font.SysFont('segoeui', 24)
            self.small_font = pygame.font.SysFont('segoeui', 18)
            self.very_small_font = pygame.font.SysFont('segoeui', 14)
            self.coord_font = pygame.font.SysFont('consolas', 12)
        except pygame.error as e:
            print(f"Font Error: {e}. Using default fonts.")
            self.font = pygame.font.Font(None, 30)
            self.small_font = pygame.font.Font(None, 24)
            self.very_small_font = pygame.font.Font(None, 20)
            self.coord_font = pygame.font.Font(None, 16)
    
        self.show_position_dots = True  # Default to True for better debugging

    def draw_board(self, surface, y_offset=0):
        """Draw the chessboard grid and labels."""
        # Draw squares
        for r in range(self.squares):
            for c in range(self.squares):
                color = self.LIGHT_SQUARE if (r + c) % 2 == 0 else self.DARK_SQUARE
                pygame.draw.rect(surface, color,
                                (c * self.square_size_px + self.board_x_offset,
                                 r * self.square_size_px + y_offset,
                                 self.square_size_px, self.square_size_px))

        # Draw rank/file labels (use smaller font)
        label_color = self.WHITE
        for i in range(self.squares):
            # Files (a-h) below board
            file_txt = self.very_small_font.render(chr(ord('a') + i), True, label_color)
            file_rect = file_txt.get_rect(center=(i * self.square_size_px + self.square_size_px // 2 + self.board_x_offset,
                                                  self.board_size_px + 10 + y_offset)) # Closer to board
            surface.blit(file_txt, file_rect)
            # Ranks (1-8) left of board
            rank_txt = self.very_small_font.render(str(self.squares - i), True, label_color) # 8 at top
            rank_rect = rank_txt.get_rect(center=(self.board_x_offset - 10, i * self.square_size_px + self.square_size_px // 2 + y_offset))
            surface.blit(rank_txt, rank_rect)

    def draw_center_marker(self, surface, x, y, size=5):
        """Draw a marker at the exact center of a square for debugging purposes."""
        pygame.draw.line(surface, self.CENTER_MARKER_COLOR, (x-size, y), (x+size, y), 2)
        pygame.draw.line(surface, self.CENTER_MARKER_COLOR, (x, y-size), (x, y+size), 2)
        pygame.draw.circle(surface, self.CENTER_MARKER_COLOR, (x, y), 2)


    # Update the piece-related methods in visualization.py to support vertical offset

    def board_to_pixel(self, board_pos, y_offset=0):
        """Convert board coordinates (col, row floats) to pixel coordinates (center of square)."""
        col, row = board_pos
        px = col * self.square_size_px + self.square_size_px // 2 + self.board_x_offset
        py = row * self.square_size_px + self.square_size_px // 2 + y_offset
        return int(px), int(py)
    
    def draw_piece(self, surface: pygame.Surface, piece: ChessPiece, selected=False, y_offset=0):
        """Draws a single chess piece using its properties."""
        is_capturing = (not piece.active and piece.capture_path) # Check if being captured
        if not piece.active and not is_capturing: return # Don't draw inactive unless being captured
    
        # --- Get Pixel Position ---
        x_center_rel, y_center_rel = piece.get_pixel_position()
        x_center = x_center_rel + self.board_x_offset # Apply board offset HERE
        y_center = y_center_rel + y_offset  # Apply y_offset
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
    
    
    def draw_pieces(self, surface: pygame.Surface, pieces: list[ChessPiece], selected_piece: ChessPiece = None, y_offset=0):
        """Draws all pieces, handling active, inactive (capturing), and selected."""
        # Draw inactive pieces first (those being captured)
        for piece in pieces:
            if not piece.active and piece.capture_path:
                self.draw_piece(surface, piece, selected=False, y_offset=y_offset) # Draw semi-transparent
    
        # Draw active, non-selected pieces
        for piece in pieces:
            if piece.active and piece != selected_piece:
                self.draw_piece(surface, piece, selected=False, y_offset=y_offset)
    
        # Draw selected piece last (on top)
        if selected_piece and selected_piece.active:
            self.draw_piece(surface, selected_piece, selected=True, y_offset=y_offset)
    
    
    def draw_paths(self, surface: pygame.Surface, pieces: list[ChessPiece], selected_piece: ChessPiece = None, y_offset=0):
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
                    px, py = self.board_to_pixel(pos, y_offset=y_offset) # Use helper function with y_offset
                    pixel_points.append((px, py))
    
                if len(pixel_points) > 1:
                    pygame.draw.lines(surface, path_color, False, pixel_points, 2)
                    # Mark end point of capture path differently
                    if not piece.active and piece.capture_path:
                         pygame.draw.circle(surface, self.RED, pixel_points[-1], 5)
    
    def _draw_captured_group(self, surface, captured_pieces, panel_x, start_y, panel_width):
        """Draws a group of captured pieces in a grid layout."""
        if not captured_pieces:
            return
    
        # Arrange in a grid
        piece_size = 40  # Size for each piece display
        spacing = 10     # Space between pieces
        max_pieces_per_row = 4  # Number of pieces per row
    
        # Calculate grid positions
        row, col = 0, 0
        for piece in captured_pieces:
            # Calculate position
            x = panel_x + 10 + col * (piece_size + spacing)
            y = start_y + row * (piece_size + spacing)
    
            # Draw piece background
            piece_rect = pygame.Rect(x, y, piece_size, piece_size)
            pygame.draw.rect(surface, self.LIGHT_GRAY, piece_rect)
    
            # Draw piece symbol
            try:
                piece_font = pygame.font.SysFont('segoeuisymbol', 30)
                text_color = self.WHITE if piece.color == PieceColor.WHITE else self.BLACK
                symbol_text = piece_font.render(piece.symbol, True, text_color)
                symbol_rect = symbol_text.get_rect(center=(x + piece_size // 2, y + piece_size // 2))
                surface.blit(symbol_text, symbol_rect)
            except Exception as e:
                print(f"Error drawing captured piece: {e}")
    
            # Move to next position
            col += 1
            if col >= max_pieces_per_row:
                col = 0
                row += 1
    
        return row + 1  # Return the number of rows used

    # --- FIX 2: Updated draw_captured_pieces_panel Method ---
    # This ensures the captured pieces panel draws correctly and doesn't overlap

    def draw_captured_pieces_panel(self, surface, captured_white, captured_black, panel_x, window_height):
        """Draws the captured pieces panel on the right side of the screen."""
        # Panel dimensions
        panel_width = 250

        # Draw panel background
        panel_rect = pygame.Rect(panel_x, 0, panel_width, window_height)
        pygame.draw.rect(surface, self.DARK_GRAY, panel_rect)

        # Draw panel title
        title_text = self.font.render("Captured Pieces", True, self.WHITE)
        title_rect = title_text.get_rect(center=(panel_x + panel_width // 2, 30))
        surface.blit(title_text, title_rect)

        # Section divider
        divider_y = window_height // 2
        pygame.draw.line(surface, self.WHITE, (panel_x, divider_y), (panel_x + panel_width, divider_y), 2)

        # Draw black captured section
        black_title = self.small_font.render("Black captured:", True, self.WHITE)
        surface.blit(black_title, (panel_x + 10, 70))

        # Draw white captured section
        white_title = self.small_font.render("White captured:", True, self.WHITE)
        surface.blit(white_title, (panel_x + 10, divider_y + 20))

        # Draw captured black pieces
        self._draw_captured_group(surface, captured_black, panel_x, 100, panel_width)

        # Draw captured white pieces
        self._draw_captured_group(surface, captured_white, panel_x, divider_y + 50, panel_width)

    # --- FIX 3: Updated _create_pid_sliders Method ---
    # Update this method to place sliders in the control panel

    def _create_pid_sliders(self):
        """Creates the PID slider widgets in the control panel."""
        if not WIDGETS_AVAILABLE: return

        # Clear previous widgets if they exist (needed for resize)
        self.pid_sliders.clear()
        self.pid_textboxes.clear()

        # Calculate slider positions based on controls panel
        slider_width = self.CONTROLS_PANEL_WIDTH - 100  # Width of sliders, leave space for text
        slider_x = self.controls_x + 30  # X position of sliders 
        slider_y_start = 450  # Starting Y position (Adjust based on panel layout)
        slider_spacing = 70  # Vertical space between sliders

        # Ensure slider start position is reasonable within window height
        if slider_y_start > self.window_height - 3 * slider_spacing:
            slider_y_start = self.window_height - 3 * slider_spacing - 20  # Adjust if too low

        # --- Kp Slider ---
        self.pid_sliders['kp'] = Slider(self.screen, slider_x, slider_y_start, slider_width, 20, 
                                       min=0, max=200, step=1, initial=self.temp_pid_kp, 
                                       colour=(200,200,200), handleColour=(0,150,0))
        self.pid_textboxes['kp'] = TextBox(self.screen, slider_x + slider_width + 10, 
                                         slider_y_start - 5, 50, 30, fontSize=18, 
                                         borderThickness=0, colour=self.renderer.LIGHT_GRAY, 
                                         textColour=self.renderer.BLACK)
        self.pid_textboxes['kp'].disable()  # Read-only display

        # --- Ki Slider ---
        self.pid_sliders['ki'] = Slider(self.screen, slider_x, slider_y_start + slider_spacing, 
                                       slider_width, 20, min=0, max=50, step=0.1, 
                                       initial=self.temp_pid_ki, colour=(200,200,200), 
                                       handleColour=(0,0,150))
        self.pid_textboxes['ki'] = TextBox(self.screen, slider_x + slider_width + 10, 
                                         slider_y_start + slider_spacing - 5, 50, 30, 
                                         fontSize=18, borderThickness=0, 
                                         colour=self.renderer.LIGHT_GRAY, 
                                         textColour=self.renderer.BLACK)
        self.pid_textboxes['ki'].disable()

        # --- Kd Slider ---
        self.pid_sliders['kd'] = Slider(self.screen, slider_x, slider_y_start + 2 * slider_spacing, 
                                       slider_width, 20, min=0, max=200, step=1, 
                                       initial=self.temp_pid_kd, colour=(200,200,200), 
                                       handleColour=(150,0,0))
        self.pid_textboxes['kd'] = TextBox(self.screen, slider_x + slider_width + 10, 
                                         slider_y_start + 2 * slider_spacing - 5, 50, 30, 
                                         fontSize=18, borderThickness=0, 
                                         colour=self.renderer.LIGHT_GRAY, 
                                         textColour=self.renderer.BLACK)
        self.pid_textboxes['kd'].disable()

        self._update_slider_textboxes()  # Set initial text

    def draw_controls(self, surface: pygame.Surface, info: dict, panel_x, sliders_active=False):
        """Draws the control panel with info, stats, and placeholders for sliders."""
        self.panel_x = panel_x  # Update panel_x based on current layout
        panel_width = 250  # Fixed width for control panel

        # Background
        pygame.draw.rect(surface, self.LIGHT_GRAY, (self.panel_x, 0, panel_width, self.window_height))

        # --- Title ---
        title_text = self.font.render("EM Chess Control", True, self.BLACK)
        title_rect = title_text.get_rect(center=(self.panel_x + panel_width // 2, 30))
        surface.blit(title_text, title_rect)

        # --- Info Area Start ---
        info_y = 70
        line_height = 20  # Reduced line height slightly
        text_x = self.panel_x + 15

        def draw_text(text, y, color=self.BLACK, font=self.small_font):
            txt_surf = font.render(text, True, color)
            surface.blit(txt_surf, (text_x, y))
            return y + line_height

        # --- Basic Controls ---
        info_y = draw_text("[Click] Piece/Target", info_y)
        info_y = draw_text("[R] Reset & Apply PID", info_y)  # Updated Reset description
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

        info_y += 3*line_height + 5  # Advance past toggles

        # --- Sim Info ---
        info_y = draw_text(f"Pattern: {info.get('current_pattern', 'N/A').upper()}", info_y)
        info_y = draw_text(f"Speed: {info.get('simulation_speed', 1.0):.1f}x", info_y)
        active_gains = info.get('pid_gains_active', (0,0,0))
        info_y = draw_text(f"Active PID: {active_gains[0]:.1f}/{active_gains[1]:.1f}/{active_gains[2]:.1f}", info_y, font=self.very_small_font)
        # Display temporary PID values from sliders if available
        if sliders_active and info.get('pid_gains_temp'):
            temp_gains = info.get('pid_gains_temp', active_gains)
            if temp_gains != active_gains:  # Only show if different
                info_y = draw_text(f"Sliders-> R: {temp_gains[0]:.1f}/{temp_gains[1]:.1f}/{temp_gains[2]:.1f}", info_y, self.BLUE, font=self.very_small_font)

        dbg_status = info.get('debug_mode', False)
        info_y = draw_text(f"[{'D'}] Debug Out: {'ON' if dbg_status else 'OFF'}", info_y, font=self.very_small_font)
        info_y += 5

        # --- Move Status & Stats ---
        selected = info.get('selected_piece')
        target = info.get('target_position')  # Target only shown during move
        status_color = self.BLACK
        status_text = "Status: Select Piece"
        if info.get('move_in_progress', False):
            status_text = "Status: MOVE IN PROGRESS"
            status_color = self.ORANGE
        elif info.get('move_complete', False):  # Use move_complete flag briefly set by ChessBoard
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
            slider_y_start = info_y + 20  # Leave more space before sliders
            slider_spacing = 70
            label_y_offset = -16  # Position label above slider

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