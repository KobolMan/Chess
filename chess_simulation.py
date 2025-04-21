# chess_simulation.py

import pygame
import numpy as np
import math
import sys
import time
import heapq
from enum import Enum

# Import from local modules
from chess_pieces import ChessPiece, PieceColor, PieceType # Enums too
from coil_controller import CoilGrid
from pathfinding import PathFinder
from visualization import ChessRenderer
from hardware_interface import ElectromagnetController # Type hint

# --- Constants ---
BOARD_SIZE_PX = 800
BOARD_SQUARES = 8
SQUARE_SIZE_PX = BOARD_SIZE_PX // BOARD_SQUARES
COIL_GRID_SIZE = 20
WINDOW_WIDTH = BOARD_SIZE_PX + 400
WINDOW_HEIGHT = BOARD_SIZE_PX + 200
FPS = 60

class ChessBoard:
    """
    Orchestrates the chess simulation. PID *directly* calculates driving force.
    Coils activated separately for visuals/hardware, avoiding stationary pieces.
    """
    def __init__(self, hardware_controller: ElectromagnetController, debug_mode=False):
        self.hardware_controller = hardware_controller
        self.debug_mode = True # Default True for tuning PID
        # self.debug_mode = debug_mode

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Electromagnetic Chess Simulation (Direct PID Force)")
        self.renderer = ChessRenderer(BOARD_SIZE_PX, BOARD_SQUARES, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.coil_grid = CoilGrid(size=COIL_GRID_SIZE, board_squares=BOARD_SQUARES)
        self.path_finder = PathFinder(board_size=BOARD_SQUARES)

        # --- **** PID GAINS (for Direct Force - Needs Tuning!) **** ---
        self.pid_kp = 150.0  # Start moderate
        self.pid_ki = 0.0    # Start with Ki=0
        self.pid_kd = 80.0   # Start higher for damping
        # --- **** END GAINS **** ---

        self.pid_integral = np.array([0.0, 0.0])
        self.pid_previous_error = np.array([0.0, 0.0]) # Needed if using dError/dt D term
        self.pid_integral_max = 50.0 # Clamp I term

        # Game State
        self.pieces: list[ChessPiece] = []; self.captured_white: list[ChessPiece] = []; self.captured_black: list[ChessPiece] = []
        self.selected_piece: ChessPiece | None = None; self.target_position: tuple[float, float] | None = None
        self.move_in_progress: bool = False; self.move_complete: bool = False; self.captured_piece: ChessPiece | None = None
        self.capture_step_index: int = 0; self.capture_path_finished: bool = False; self.temporarily_moved_pieces: list[ChessPiece] = []

        # Simulation Parameters
        self.simulation_speed = 1.0; self.field_update_timer = 0.0; self.field_update_interval = 0.05
        # Damping coefficient removed - PID's Kd handles it

        # Visualization Options
        self.show_coils=False; self.show_field=False; self.show_paths=True; self.show_heatmap=False

        # Move Pattern Options
        self.current_pattern="directed"; self.patterns=["directed","knight","radial"]

        # Offboard capture targets
        self.capture_target_white = (BOARD_SQUARES+1.0, BOARD_SQUARES-1.5); self.capture_target_black = (BOARD_SQUARES+1.0, 0.5)

        self.clock = pygame.time.Clock(); self.heatmap_surface = None; self.heatmap_needs_update = True

        self.initialize_pieces()
        print(f"ChessBoard initialized. HW Sim: {self.hardware_controller.simulation_mode}, Debug: {self.debug_mode}")
        print(f"PID Gains (Direct Force): Kp={self.pid_kp}, Ki={self.pid_ki}, Kd={self.pid_kd}, Imax={self.pid_integral_max}")


    def initialize_pieces(self):
        """Sets up the pieces in their starting positions."""
        self.pieces.clear(); self.captured_white.clear(); self.captured_black.clear()
        setup = { # (col, row)
            (0,0):(PieceColor.BLACK,PieceType.ROOK),(1,0):(PieceColor.BLACK,PieceType.KNIGHT), (2,0):(PieceColor.BLACK,PieceType.BISHOP),(3,0):(PieceColor.BLACK,PieceType.QUEEN), (4,0):(PieceColor.BLACK,PieceType.KING),(5,0):(PieceColor.BLACK,PieceType.BISHOP), (6,0):(PieceColor.BLACK,PieceType.KNIGHT),(7,0):(PieceColor.BLACK,PieceType.ROOK),
            (0,1):(PieceColor.BLACK,PieceType.PAWN),(1,1):(PieceColor.BLACK,PieceType.PAWN), (2,1):(PieceColor.BLACK,PieceType.PAWN),(3,1):(PieceColor.BLACK,PieceType.PAWN), (4,1):(PieceColor.BLACK,PieceType.PAWN),(5,1):(PieceColor.BLACK,PieceType.PAWN), (6,1):(PieceColor.BLACK,PieceType.PAWN),(7,1):(PieceColor.BLACK,PieceType.PAWN),
            (0,7):(PieceColor.WHITE,PieceType.ROOK),(1,7):(PieceColor.WHITE,PieceType.KNIGHT), (2,7):(PieceColor.WHITE,PieceType.BISHOP),(3,7):(PieceColor.WHITE,PieceType.QUEEN), (4,7):(PieceColor.WHITE,PieceType.KING),(5,7):(PieceColor.WHITE,PieceType.BISHOP), (6,7):(PieceColor.WHITE,PieceType.KNIGHT),(7,7):(PieceColor.WHITE,PieceType.ROOK),
            (0,6):(PieceColor.WHITE,PieceType.PAWN),(1,6):(PieceColor.WHITE,PieceType.PAWN), (2,6):(PieceColor.WHITE,PieceType.PAWN),(3,6):(PieceColor.WHITE,PieceType.PAWN), (4,6):(PieceColor.WHITE,PieceType.PAWN),(5,6):(PieceColor.WHITE,PieceType.PAWN), (6,6):(PieceColor.WHITE,PieceType.PAWN),(7,6):(PieceColor.WHITE,PieceType.PAWN), }
        for pos, (color, ptype) in setup.items(): self.pieces.append(ChessPiece(color, ptype, pos, board_squares=BOARD_SQUARES, square_size=SQUARE_SIZE_PX, coil_grid_size=COIL_GRID_SIZE))

    def get_piece_at_square(self, square_col_row):
        """Finds the active piece occupying the given integer square."""
        target_col, target_row = square_col_row
        for piece in self.pieces:
            if piece.active: p_col_int=int(round(piece.position[0])); p_row_int=int(round(piece.position[1]));
            if p_col_int == target_col and p_row_int == target_row: return piece
        return None

    def is_square_occupied(self, square_col_row, ignore_piece=None):
        """Checks if a square is occupied by an active piece."""
        piece = self.get_piece_at_square(square_col_row); return piece is not None and piece != ignore_piece

    def handle_click(self, pixel_pos):
        """Processes mouse clicks using integer square checks."""
        x_pix, y_pix = pixel_pos;
        if not (0<=x_pix<BOARD_SIZE_PX and 0<=y_pix<BOARD_SIZE_PX): return
        clicked_col_int=x_pix//SQUARE_SIZE_PX; clicked_row_int=y_pix//SQUARE_SIZE_PX; clicked_square=(clicked_col_int, clicked_row_int)
        if self.move_in_progress: return
        piece_in_clicked_square = self.get_piece_at_square(clicked_square)
        if self.selected_piece:
            selected_square = (int(round(self.selected_piece.position[0])), int(round(self.selected_piece.position[1])))
            if clicked_square == selected_square: print("Deselected piece."); self.selected_piece=None; self.target_position=None
            else:
                target_col_int, target_row_int = clicked_square; piece_at_target = self.get_piece_at_square(clicked_square)
                if piece_at_target and piece_at_target.color==self.selected_piece.color: print(f"Switched selection to {piece_at_target.symbol}"); self.selected_piece=piece_at_target; self.target_position=None
                else:
                    self.target_position=(float(target_col_int)+0.5, float(target_row_int)+0.5); print(f"Target set: {self.target_position}")
                    path_is_clear = True
                    if self.selected_piece.piece_type != PieceType.KNIGHT:
                         if not self.is_path_clear(self.selected_piece.position, self.target_position):
                              print("Path blocked. Attempting clearance...")
                              if self.attempt_to_clear_path(self.selected_piece.position, self.target_position): print("Path cleared.")
                              else: print("Clearance failed. Move cancelled."); path_is_clear=False; self.target_position=None
                    if path_is_clear and self.target_position is not None:
                         if piece_at_target:
                             self.captured_piece=piece_at_target; self.captured_piece.active=False; print(f"Capturing {self.captured_piece.symbol}")
                             capture_target_col_row=self.capture_target_white if self.captured_piece.color==PieceColor.WHITE else self.capture_target_black
                             start_rc=(target_row_int, target_col_int); target_rc=(int(round(capture_target_col_row[1])), int(round(capture_target_col_row[0])))
                             path_nodes_rc=self.path_finder.find_capture_path(start_rc, target_rc, self.pieces, self.captured_piece)
                             if path_nodes_rc: self.captured_piece.capture_path=[(float(c)+0.5, float(r)+0.5) for r,c in path_nodes_rc]; print(f"Generated capture path with {len(self.captured_piece.capture_path)} points.")
                             else: print("Warning: Could not generate capture path!"); self.captured_piece.capture_path=[]
                             if self.captured_piece.color == PieceColor.WHITE: self.captured_white.append(self.captured_piece)
                             else: self.captured_black.append(self.captured_piece)
                         else: self.captured_piece=None
                         self.start_move() # Start the move process
        else:
            if piece_in_clicked_square: self.selected_piece=piece_in_clicked_square; self.target_position=None; self.captured_piece=None; self.temporarily_moved_pieces=[]; print(f"Selected {self.selected_piece.symbol}")
            else: print("Clicked empty square."); self.selected_piece=None; self.target_position=None

    def is_path_clear(self, start_pos_col_row, end_pos_col_row):
        """Checks for obstructions along a straight line path (uses float coords)."""
        start_c, start_r = start_pos_col_row; end_c, end_r = end_pos_col_row; delta_c = end_c-start_c; delta_r = end_r-start_r; steps = max(abs(delta_c), abs(delta_r))
        if steps < 1.5: return True
        num_checks = int(steps) + 1
        for i in range(1, num_checks):
            t = i / num_checks; check_c = start_c + delta_c * t; check_r = start_r + delta_r * t
            for piece in self.pieces:
                if piece.active and piece != self.selected_piece:
                    piece_c, piece_r = piece.position; collision_radius = (piece.diameter / SQUARE_SIZE_PX * 0.6)
                    if np.sum((np.array([piece_c, piece_r]) - np.array([check_c, check_r]))**2) < collision_radius**2: return False
        return True

    def attempt_to_clear_path(self, start_pos_col_row, end_pos_col_row):
        """Tries to nudge one blocking piece aside."""
        blocking_pieces = []; start_c, start_r = start_pos_col_row; end_c, end_r = end_pos_col_row; delta_c=end_c-start_c; delta_r=end_r-start_r; steps=max(abs(delta_c),abs(delta_r))
        if steps < 1.5: return True
        num_checks = int(steps) + 1
        for i in range(1, num_checks):
            t = i / num_checks; check_c = start_c + delta_c * t; check_r = start_r + delta_r * t
            for piece in self.pieces:
                if piece.active and piece != self.selected_piece and piece not in self.temporarily_moved_pieces:
                    piece_c, piece_r = piece.position; collision_radius = (piece.diameter / SQUARE_SIZE_PX * 0.6)
                    if np.sum((np.array([piece_c,piece_r])-np.array([check_c,check_r]))**2) < collision_radius**2 and piece not in blocking_pieces: blocking_pieces.append(piece)
        if not blocking_pieces: return True
        piece_to_move = next((p for p in blocking_pieces if p.piece_type == PieceType.PAWN), blocking_pieces[0])
        path_vec = np.array([delta_c, delta_r]); path_len = np.linalg.norm(path_vec);
        if path_len == 0: return True
        perp_vec = np.array([-delta_r / path_len, delta_c / path_len])
        move_distance = 0.6; original_pos = piece_to_move.position.copy()
        for move_dir in [perp_vec * move_distance, perp_vec * -move_distance]:
            temp_target_pos = original_pos + move_dir; target_square = (int(round(temp_target_pos[0])), int(round(temp_target_pos[1])))
            if 0<=target_square[0]<BOARD_SQUARES and 0<=target_square[1]<BOARD_SQUARES:
                 if not self.is_square_occupied(target_square, ignore_piece=piece_to_move):
                    print(f"Temporarily moving {piece_to_move.symbol} aside."); piece_to_move.position_before_temp_move = original_pos
                    piece_to_move.position = np.array([target_square[0]+0.5, target_square[1]+0.5]); piece_to_move.velocity *= 0.1; piece_to_move.path = [piece_to_move.position.copy()]
                    self.temporarily_moved_pieces.append(piece_to_move); return True
        print(f"Could not find clear spot for {piece_to_move.symbol}."); return False

    def start_move(self):
        """Initiates the move sequence and resets PID controller state."""
        if self.selected_piece and self.target_position:
            self.move_in_progress = True; self.move_complete = False; self.capture_step_index = 0
            self.capture_path_finished = (self.captured_piece is None or not self.captured_piece.capture_path)
            self.coil_grid.reset(); self.heatmap_needs_update = True
            self.pid_integral.fill(0.0); self.pid_previous_error.fill(0.0) # Reset PID

    def _create_keep_out_mask(self):
        """Identifies coils too close to stationary pieces."""
        blocked_coils = set()
        keep_out_radius_coils = 2.0; keep_out_radius_sq = keep_out_radius_coils**2
        pieces_to_avoid = [p for p in self.pieces if p.active and p != self.selected_piece and p not in self.temporarily_moved_pieces]
        if self.captured_piece and not self.capture_path_finished: pieces_to_avoid.append(self.captured_piece)
        for piece in pieces_to_avoid:
            piece_coil_c, piece_coil_r = piece.get_coil_position()
            min_r = max(0, int(math.floor(piece_coil_r - keep_out_radius_coils))); max_r = min(COIL_GRID_SIZE, int(math.ceil(piece_coil_r + keep_out_radius_coils)))
            min_c = max(0, int(math.floor(piece_coil_c - keep_out_radius_coils))); max_c = min(COIL_GRID_SIZE, int(math.ceil(piece_coil_c + keep_out_radius_coils)))
            for r in range(min_r, max_r):
                for c in range(min_c, max_c):
                    if (r-piece_coil_r)**2 + (c-piece_coil_c)**2 < keep_out_radius_sq: blocked_coils.add((r, c))
        return blocked_coils

    def update_move(self, dt):
        """Updates piece physics using direct PID force. Coils avoid other pieces."""
        if not self.move_in_progress: return

        effective_dt = dt * self.simulation_speed
        if effective_dt <= 0: return

        if self.selected_piece and self.target_position:
            current_pos = self.selected_piece.position.copy()
            current_vel = self.selected_piece.velocity.copy()
            target_pos = np.array(self.target_position)

            # --- PID Control Calculation ---
            error = target_pos - current_pos
            distance_to_target = np.linalg.norm(error)
            p_term = self.pid_kp * error
            i_term = np.array([0.0, 0.0])
            if self.pid_ki > 1e-6:
                self.pid_integral += error * effective_dt
                integral_mag = np.linalg.norm(self.pid_integral)
                if integral_mag > self.pid_integral_max: self.pid_integral = self.pid_integral * (self.pid_integral_max / integral_mag)
                i_term = self.pid_ki * self.pid_integral
            d_term = -self.pid_kd * current_vel # Velocity-based D term
            pid_force = p_term + i_term + d_term # Total desired force
            max_pid_force = 5000.0
            pid_force_mag = np.linalg.norm(pid_force)
            if pid_force_mag > max_pid_force: pid_force = pid_force * (max_pid_force / pid_force_mag)

            if self.debug_mode: print(f"\n--- Update Step dt={effective_dt:.4f} ---"); print(f"  Piece: {self.selected_piece.symbol} Pos:({current_pos[0]:.2f},{current_pos[1]:.2f}) Vel:({current_vel[0]:.2f},{current_vel[1]:.2f}) Dist:{distance_to_target:.2f}"); print(f"  PID Error: ({error[0]:.2f},{error[1]:.2f})"); print(f"  PID Terms: P:({p_term[0]:.1f},{p_term[1]:.1f}) I:({i_term[0]:.1f},{i_term[1]:.1f}) D:({d_term[0]:.1f},{d_term[1]:.1f})"); print(f"  PID Force: ({pid_force[0]:.2f},{pid_force[1]:.2f})") # PID Force applied

            # Stop condition
            stop_threshold = 0.05; velocity_threshold = 0.15
            move_finished = (distance_to_target < stop_threshold and np.linalg.norm(current_vel) < velocity_threshold) or distance_to_target < 0.02

            if move_finished:
                final_pos_before_snap=self.selected_piece.position.copy(); final_vel_before_snap=self.selected_piece.velocity.copy()
                self.selected_piece.position=target_pos; self.selected_piece.velocity.fill(0.0)
                if self.debug_mode: print(f"Move complete. Snapped from ({final_pos_before_snap[0]:.3f},{final_pos_before_snap[1]:.3f}) Vel ({final_vel_before_snap[0]:.3f},{final_vel_before_snap[1]:.3f})"); print(f"Final Snapped Position: ({self.selected_piece.position[0]},{self.selected_piece.position[1]}) Velocity: ({self.selected_piece.velocity[0]},{self.selected_piece.velocity[1]})")
                else: print("Move complete.")
                self.move_in_progress=False; self.move_complete=True; self.coil_grid.reset(); self.hardware_controller.reset_all_coils(); self.heatmap_needs_update=True
                for piece in self.temporarily_moved_pieces: piece.return_from_temporary_move()
                self.temporarily_moved_pieces=[]; self.target_position=None
            else:
                # --- Move In Progress ---
                # --- Coil Simulation & Hardware Update (For Visuals/Hardware ONLY) ---
                self.field_update_timer += effective_dt
                if self.field_update_timer >= self.field_update_interval:
                    self.field_update_timer = 0
                    blocked_coils_set = self._create_keep_out_mask() # Use mask
                    current_coil_pos = self.selected_piece.get_coil_position()
                    nominal_target_coil = np.array(target_pos) * (COIL_GRID_SIZE / BOARD_SQUARES) # Use NOMINAL target
                    dx_board=target_pos[0]-current_pos[0]; dy_board=target_pos[1]-current_pos[1]
                    is_knight_shape=(abs(round(dx_board))==1 and abs(round(dy_board))==2)or(abs(round(dx_board))==2 and abs(round(dy_board))==1)
                    chosen_pattern=self.current_pattern; straight_threshold=0.1
                    if self.selected_piece.piece_type==PieceType.KNIGHT:
                        if is_knight_shape and self.current_pattern in["knight","directed"]: chosen_pattern="knight"
                    elif abs(dx_board)<straight_threshold and abs(dy_board)>straight_threshold: chosen_pattern="straight_vertical"
                    elif abs(dy_board)<straight_threshold and abs(dx_board)>straight_threshold: chosen_pattern="straight_horizontal"
                    elif abs(abs(dx_board)-abs(dy_board))<straight_threshold*2 and abs(dx_board)>straight_threshold: chosen_pattern="directed"
                    else: chosen_pattern="directed"
                    scale_factor = 1.0; scale_distance_threshold = 1.2
                    if distance_to_target < scale_distance_threshold: min_scale=0.05; ratio=distance_to_target/scale_distance_threshold; scale_factor=min_scale+(1.0-min_scale)*(ratio**3); scale_factor=max(min_scale,scale_factor)
                    current_intensity = 100 * scale_factor
                    # Activate coils with mask
                    self.coil_grid.activate_coil_pattern(chosen_pattern,current_coil_pos,tuple(nominal_target_coil),intensity=current_intensity,radius=5,blocked_coils=blocked_coils_set)
                    self.coil_grid.update_magnetic_field(); self.heatmap_needs_update=True
                    self.hardware_controller.apply_state(self.coil_grid.coil_power, self.coil_grid.coil_current)

                # --- Force Application (Using DIRECT PID Force) ---
                # Apply PID force directly to the piece physics
                # apply_force no longer handles damping internally
                self.selected_piece.apply_force(pid_force, effective_dt)

                if self.debug_mode: print(f"  End Pos:({self.selected_piece.position[0]:.2f},{self.selected_piece.position[1]:.2f}) End Vel:({self.selected_piece.velocity[0]:.2f},{self.selected_piece.velocity[1]:.2f})")

        # --- Update Captured Piece ---
        if self.captured_piece and not self.capture_path_finished:
            node_reached_or_finished = self.captured_piece.follow_capture_path(self.capture_step_index)
            if node_reached_or_finished:
                if self.capture_step_index<len(self.captured_piece.capture_path)-1: self.capture_step_index+=1
                else: self.capture_path_finished=True; print(f"Capture movement finished for {self.captured_piece.symbol}.")


    def reset(self): # No changes
        print("Resetting board..."); self.initialize_pieces(); self.selected_piece=None; self.target_position=None; self.move_in_progress=False; self.move_complete=False; self.captured_piece=None; self.capture_step_index=0; self.capture_path_finished=False; self.temporarily_moved_pieces=[]; self.coil_grid.reset(); self.hardware_controller.reset_all_coils(); self.heatmap_needs_update=True; self.pid_integral.fill(0.0); self.pid_previous_error.fill(0.0); print("Board reset complete.")

    def cycle_pattern(self): # No changes
        current_index=self.patterns.index(self.current_pattern); next_index=(current_index+1)%len(self.patterns); self.current_pattern=self.patterns[next_index]; print(f"Switched to pattern: {self.current_pattern}")

    def load_heatmap(self): # No changes
        if self.heatmap_needs_update or self.heatmap_surface is None:
            heatmap_path=self.coil_grid.plot_heatmap(filename="field_heatmap.png")
            if heatmap_path:
                try: loaded_surface=pygame.image.load(heatmap_path).convert(); self.heatmap_surface=loaded_surface; self.heatmap_needs_update=False
                except Exception as e: print(f"Error loading heatmap image '{heatmap_path}': {e}"); self.heatmap_surface=None
            else: print("Heatmap generation failed."); self.heatmap_surface=None

    def run(self): # No changes
        running = True
        while running:
            dt_sec = min(self.clock.tick(FPS) / 1000.0, 0.1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    key = event.key
                    if key == pygame.K_r: self.reset()
                    elif key == pygame.K_c: self.show_coils = not self.show_coils
                    elif key == pygame.K_f: self.show_field = not self.show_field
                    elif key == pygame.K_p: self.show_paths = not self.show_paths
                    elif key == pygame.K_h: self.show_heatmap = not self.show_heatmap; self.heatmap_needs_update = True
                    elif key == pygame.K_m: self.cycle_pattern()
                    elif key == pygame.K_PLUS or key == pygame.K_EQUALS: self.simulation_speed = min(10.0, self.simulation_speed + 0.2); print(f"Sim speed: {self.simulation_speed:.1f}x")
                    elif key == pygame.K_MINUS: self.simulation_speed = max(0.1, self.simulation_speed - 0.2); print(f"Sim speed: {self.simulation_speed:.1f}x")
                    elif key == pygame.K_d: self.debug_mode = not self.debug_mode; print(f"Debug mode: {self.debug_mode}")
                    elif key == pygame.K_ESCAPE: running = False
            self.update_move(dt_sec)
            self.renderer.draw_board(self.screen);
            if self.show_coils: self.coil_grid.draw(self.screen, BOARD_SIZE_PX)
            if self.show_field: self.coil_grid.draw_field_overlay(self.screen, BOARD_SIZE_PX)
            self.renderer.draw_pieces(self.screen, self.pieces, self.selected_piece)
            if self.show_paths: self.renderer.draw_paths(self.screen, self.pieces, self.selected_piece)
            info_dict = { 'selected_piece':self.selected_piece,'target_position':self.target_position,'move_in_progress':self.move_in_progress,
                          'move_complete':self.move_complete,'show_coils':self.show_coils,'show_field':self.show_field,'show_paths':self.show_paths,
                          'show_heatmap':self.show_heatmap,'current_pattern':self.current_pattern,'simulation_speed':self.simulation_speed, }
            self.renderer.draw_controls(self.screen, info_dict); self.renderer.draw_capture_area(self.screen, self.captured_white, self.captured_black)
            if self.show_heatmap: self.load_heatmap()
            self.renderer.draw_heatmap(self.screen, self.heatmap_surface if self.show_heatmap else None)
            pygame.display.flip()
        print("Simulation loop ended.")