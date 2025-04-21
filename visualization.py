# visualization.py

import pygame
import numpy as np
import math
from chess_pieces import ChessPiece, PieceColor, PieceType, PIECE_SYMBOLS # Import piece info

class ChessRenderer:
    """Handles all Pygame rendering for the chess simulation."""

    def __init__(self, board_size_px, squares, window_width, window_height):
        self.board_size_px = board_size_px
        self.squares = squares
        self.square_size_px = board_size_px // squares
        self.window_width = window_width
        self.window_height = window_height

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
        self.LIGHT_GRAY = (200, 200, 200)
        self.DARK_GRAY = (100, 100, 100)
        self.HIGHLIGHT = (124, 252, 0) # Green highlight for selection
        self.PATH_COLOR = (0, 0, 255, 150) # Blueish semi-transparent for path
        self.SEL_PATH_COLOR = (0, 255, 0, 150) # Greenish semi-transparent for selected path
        self.CAPTURE_PATH_COLOR = (255, 0, 0, 150) # Reddish semi-transparent for capture path

        # Initialize fonts
        try:
            pygame.font.init() # Ensure font module is initialized
            self.font = pygame.font.SysFont('segoeui', 24)
            self.small_font = pygame.font.SysFont('segoeui', 16)
            # Piece font will be created dynamically based on size in draw_piece
        except pygame.error as e:
            print(f"Font Error: {e}. Using default fonts.")
            self.font = pygame.font.Font(None, 30)
            self.small_font = pygame.font.Font(None, 20)


    def draw_board(self, surface):
        """Draw the chessboard grid and labels."""
        surface.fill(self.DARK_GRAY) # Fill entire window background

        # Draw squares
        for r in range(self.squares):
            for c in range(self.squares):
                color = self.LIGHT_SQUARE if (r + c) % 2 == 0 else self.DARK_SQUARE
                pygame.draw.rect(surface, color,
                                 (c * self.square_size_px, r * self.square_size_px,
                                  self.square_size_px, self.square_size_px))

        # Draw rank/file labels
        label_color = self.WHITE
        for i in range(self.squares):
            # Files (a-h) below board
            file_txt = self.small_font.render(chr(ord('a') + i), True, label_color)
            file_rect = file_txt.get_rect(center=(i * self.square_size_px + self.square_size_px // 2,
                                                  self.board_size_px + 15))
            surface.blit(file_txt, file_rect)
            # Ranks (1-8) left of board
            rank_txt = self.small_font.render(str(self.squares - i), True, label_color) # 8 at top
            rank_rect = rank_txt.get_rect(center=(-15, i * self.square_size_px + self.square_size_px // 2))
            surface.blit(rank_txt, rank_rect)

    def draw_piece(self, surface: pygame.Surface, piece: ChessPiece, selected=False):
        """Draws a single chess piece using its properties."""
        if not piece.active: return

        x_px, y_px = piece.get_pixel_position()
        symbol = piece.symbol
        text_color = self.WHITE if piece.color == PieceColor.WHITE else self.BLACK

        # Calculate dynamic font size
        size = int(self.square_size_px * 0.7 * (piece.diameter / 40)) # Scale based on diameter relative to king
        size = max(20, min(size, int(self.square_size_px * 0.9))) # Clamp size
        try:
            piece_font = pygame.font.SysFont('segoeuisymbol', size)
        except:
            piece_font = pygame.font.Font(None, size) # Fallback

        piece_text = piece_font.render(symbol, True, text_color)
        text_rect = piece_text.get_rect(center=(int(x_px), int(y_px)))

        # Draw selection highlight
        if selected:
            # Draw circle slightly larger than diameter, thicker line
            highlight_radius = int(piece.diameter / 2 * 1.2) # Use piece diameter property
            pygame.draw.circle(surface, self.HIGHLIGHT, (int(x_px), int(y_px)), highlight_radius, 3)

        # Draw the piece symbol
        surface.blit(piece_text, text_rect)

        # Optional: Draw debug info (center dot, coords)
        # pygame.draw.circle(surface, self.RED, (int(x_px), int(y_px)), 3)
        # col, row = piece.get_board_position()
        # coord_text = self.small_font.render(f"({col:.1f},{row:.1f})", True, self.YELLOW)
        # coord_rect = coord_text.get_rect(center=(int(x_px), int(y_px + 30)))
        # surface.blit(coord_text, coord_rect)


    def draw_pieces(self, surface: pygame.Surface, pieces: list[ChessPiece], selected_piece: ChessPiece = None):
        """Draws all active pieces, selected piece last."""
        # Draw non-selected active pieces
        for piece in pieces:
            if piece.active and piece != selected_piece:
                self.draw_piece(surface, piece, selected=False)

        # Draw captured pieces if they are still animating off board
        for piece in pieces:
             if not piece.active and len(piece.capture_path) > 0 : # If inactive but has path = being captured
                 # Check if it's still on board during animation
                 col, row = piece.position
                 if 0 <= col < self.squares and 0 <= row < self.squares:
                      # Draw semi-transparently or normally?
                      self.draw_piece(surface, piece, selected=False)


        # Draw selected piece last (on top)
        if selected_piece and selected_piece.active:
            self.draw_piece(surface, selected_piece, selected=True)

    # --- draw_paths CORRECTED ---
    def draw_paths(self, surface: pygame.Surface, pieces: list[ChessPiece], selected_piece: ChessPiece = None):
        """Draws the movement paths for pieces."""
        path_surface = pygame.Surface((self.board_size_px, self.board_size_px), pygame.SRCALPHA)
        path_drawn = False

        # Function to convert board coords (col, row) to pixel coords
        def board_to_pixel(board_pos):
            col, row = board_pos
            return (int(col * self.square_size_px + self.square_size_px // 2),
                    int(row * self.square_size_px + self.square_size_px // 2))

        for piece in pieces:
            # Draw regular path for active pieces
            if piece.active and len(piece.path) > 1:
                # Convert path points (which are board coords) to pixels
                pixel_points = [board_to_pixel(pos) for pos in piece.path] # Use helper function
                if len(pixel_points) > 1:
                    color = self.SEL_PATH_COLOR if piece == selected_piece else self.PATH_COLOR
                    pygame.draw.lines(path_surface, color, False, pixel_points, 2)
                    path_drawn = True

            # Draw capture path for pieces being captured (inactive with path)
            elif not piece.active and len(piece.capture_path) > 0:
                 # Convert capture path nodes (board coords) to pixels
                 pixel_points = [board_to_pixel(pos) for pos in piece.capture_path] # Use helper function
                 if len(pixel_points) > 1:
                      pygame.draw.lines(path_surface, self.CAPTURE_PATH_COLOR, False, pixel_points, 2)
                      # Mark end point of capture path
                      pygame.draw.circle(path_surface, self.RED, pixel_points[-1], 5)
                      path_drawn = True


        if path_drawn:
            surface.blit(path_surface, (0, 0))
    # --- End Correction ---


    def draw_controls(self, surface: pygame.Surface, info: dict):
        """Draws the control panel using info dictionary."""
        panel_x = self.board_size_px
        panel_width = self.window_width - self.board_size_px
        pygame.draw.rect(surface, self.LIGHT_GRAY, (panel_x, 0, panel_width, self.window_height))

        title_text = self.font.render("EM Chess Control", True, self.BLACK)
        title_rect = title_text.get_rect(center=(panel_x + panel_width // 2, 30))
        surface.blit(title_text, title_rect)

        info_y = 70
        line_height = 22

        def draw_text(text, y, color=self.BLACK, font=self.small_font):
            txt_surf = font.render(text, True, color)
            surface.blit(txt_surf, (panel_x + 15, y))
            return y + line_height

        # Basic Controls
        info_y = draw_text("[Click] Piece to Select/Target", info_y)
        info_y = draw_text("[R] Reset Board", info_y)
        info_y = draw_text("[M] Cycle Pattern", info_y)
        info_y = draw_text("[+/-] Speed", info_y)
        info_y = draw_text("[Esc] Quit", info_y)
        info_y += 5

        # Toggles
        info_y = draw_text("[C] Coils Viz: " + ("ON" if info.get('show_coils', False) else "OFF"), info_y, self.GREEN if info.get('show_coils', False) else self.RED)
        info_y = draw_text("[F] Field Viz: " + ("ON" if info.get('show_field', False) else "OFF"), info_y, self.GREEN if info.get('show_field', False) else self.RED)
        info_y = draw_text("[P] Paths Viz: " + ("ON" if info.get('show_paths', True) else "OFF"), info_y, self.GREEN if info.get('show_paths', True) else self.RED)
        info_y = draw_text("[H] Heatmap: " + ("ON" if info.get('show_heatmap', False) else "OFF"), info_y, self.GREEN if info.get('show_heatmap', False) else self.RED)
        info_y += 5

        # Sim Info
        info_y = draw_text(f"Pattern: {info.get('current_pattern', 'N/A').upper()}", info_y)
        info_y = draw_text(f"Speed: {info.get('simulation_speed', 1.0):.1f}x", info_y)
        info_y += 5

        # Move Status
        selected = info.get('selected_piece')
        target = info.get('target_position')
        if selected:
            sel_col, sel_row = selected.position
            info_y = draw_text(f"Selected: {selected.symbol} ({selected.color.name}) @ ({sel_col:.1f},{sel_row:.1f})", info_y, self.BLUE)
            if target:
                info_y = draw_text(f"Target: ({target[0]:.1f}, {target[1]:.1f})", info_y, self.GREEN)

        if info.get('move_in_progress', False):
            info_y = draw_text("Status: MOVE IN PROGRESS", info_y, self.ORANGE)
            # Optional progress calculation here if needed
        elif info.get('move_complete', False):
            info_y = draw_text("Status: Move Complete", info_y, self.GREEN)
        elif selected:
             info_y = draw_text("Status: Target?", info_y, self.BLACK)
        else:
             info_y = draw_text("Status: Select Piece", info_y, self.BLACK)


    def draw_capture_area(self, surface: pygame.Surface, captured_white: list, captured_black: list):
        """Draws the display for captured pieces."""
        area_x = self.board_size_px + 15
        area_width = self.window_width - self.board_size_px - 30
        icon_size = 25 # Smaller icons
        spacing = 4
        white_y_start = self.window_height - 200 # Position area lower down
        black_y_start = self.window_height - 100

        # Helper function to draw list of pieces
        def draw_captured_list(y_start, title, pieces_list, text_color):
            text_surf = self.small_font.render(title, True, self.WHITE)
            surface.blit(text_surf, (area_x, y_start - 20))
            current_x = area_x
            current_y = y_start
            max_x = area_x + area_width - icon_size
            try:
                cap_font = pygame.font.SysFont('segoeuisymbol', icon_size)
            except:
                cap_font = pygame.font.Font(None, icon_size)

            for piece in pieces_list:
                symbol_text = cap_font.render(piece.symbol, True, text_color)
                surface.blit(symbol_text, (current_x, current_y))
                current_x += icon_size + spacing
                if current_x > max_x:
                    current_x = area_x
                    current_y += icon_size + spacing

        # Draw the lists
        draw_captured_list(white_y_start, "Captured by Black:", captured_white, self.WHITE)
        draw_captured_list(black_y_start, "Captured by White:", captured_black, self.BLACK)


    def draw_heatmap(self, surface: pygame.Surface, heatmap_image: pygame.Surface = None):
        """Draws the magnetic field heatmap below the board."""
        heatmap_area_y = self.board_size_px + 30 # Area below board and labels
        heatmap_area_height = self.window_height - heatmap_area_y - 10
        heatmap_area_width = self.board_size_px # Use width of board
        heatmap_area_x = 0

        # Background for heatmap area
        pygame.draw.rect(surface, self.DARK_GRAY,
                         (heatmap_area_x, heatmap_area_y - 5, heatmap_area_width, heatmap_area_height + 5))

        # Title
        title_text = self.small_font.render("Field Strength Heatmap (H to toggle)", True, self.WHITE)
        title_rect = title_text.get_rect(midtop=(heatmap_area_x + heatmap_area_width // 2, heatmap_area_y))
        surface.blit(title_text, title_rect)

        if heatmap_image:
            img_rect = heatmap_image.get_rect()
            # Scale heatmap to fit, maintaining aspect ratio
            scale = min((heatmap_area_width * 0.95) / img_rect.width,
                        (heatmap_area_height - 25) / img_rect.height) # Leave space for title
            scaled_w = int(img_rect.width * scale)
            scaled_h = int(img_rect.height * scale)

            if scaled_w > 0 and scaled_h > 0:
                scaled_img = pygame.transform.smoothscale(heatmap_image, (scaled_w, scaled_h))
                # Center the scaled image
                display_x = heatmap_area_x + (heatmap_area_width - scaled_w) // 2
                display_y = heatmap_area_y + 25 + (heatmap_area_height - 25 - scaled_h) // 2
                surface.blit(scaled_img, (display_x, display_y))
        else:
            # Placeholder text if no heatmap image provided
            ph_text = self.small_font.render("Heatmap Off / Not Generated", True, self.YELLOW)
            ph_rect = ph_text.get_rect(center=(heatmap_area_x + heatmap_area_width // 2,
                                               heatmap_area_y + heatmap_area_height // 2))
            surface.blit(ph_text, ph_rect)


    # Add highlight_square etc. if needed by ChessBoard logic directly
    def highlight_square(self, surface, row, col, color=None):
        """Highlight a square on the board (e.g., for valid moves)."""
        if color is None: color = self.HIGHLIGHT
        rect = pygame.Rect(col * self.square_size_px, row * self.square_size_px,
                           self.square_size_px, self.square_size_px)
        # Draw semi-transparent rectangle overlay
        s = pygame.Surface((self.square_size_px, self.square_size_px), pygame.SRCALPHA)
        s.fill((*color[:3], 100)) # Use color but add alpha
        surface.blit(s, (rect.x, rect.y))
        # Optionally draw border too
        # pygame.draw.rect(surface, color, rect, 3)