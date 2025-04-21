# pathfinding.py

import heapq
import math
import numpy as np
from chess_pieces import ChessPiece # Import needed for type hints and properties

class PathFinder:
    """
    A* pathfinding algorithm for chess piece movement planning.
    Handles grid-based pathfinding, obstacle avoidance, and capture paths.
    Uses (row, col) for internal grid representation.
    """

    def __init__(self, board_size=8, coil_grid_size=20):
        self.board_squares = board_size # Standard board size (e.g., 8x8)
        # coil_grid_size might not be directly needed here unless path influences coil activation

    def find_path(self, start_pos_rc, end_pos_rc, board_state, moving_piece: ChessPiece):
        """
        Find a path using A* from start to end, avoiding other pieces.

        Args:
            start_pos_rc: Tuple (row, col) for starting square.
            end_pos_rc: Tuple (row, col) for ending square.
            board_state: List of all ChessPiece objects on the board.
            moving_piece: The specific ChessPiece object that is moving.

        Returns:
            List of (row, col) integer tuples representing the path, or None if no path found.
        """
        # Knights jump over pieces, no A* needed for path itself (coil control handles jump)
        if moving_piece.piece_type == moving_piece.piece_type.KNIGHT:
             # Return a simple path (start, end) as A* isn't needed for collision
             # The coil controller needs to handle the jump pattern.
             return [start_pos_rc, end_pos_rc] # Or potentially more points for smoother coil control

        # --- A* for non-knight pieces ---
        obstacle_grid = self._create_obstacle_grid(board_state, moving_piece)
        return self._a_star_search(start_pos_rc, end_pos_rc, obstacle_grid)


    def find_capture_path(self, start_pos_rc, target_off_board_rc, board_state, moving_piece: ChessPiece):
        """
        Find a path for a captured piece to move off the board.

        Args:
            start_pos_rc: Starting square (row, col) where capture occurred.
            target_off_board_rc: Target position (row, col) just off the board edge.
            board_state: List of all ChessPiece objects.
            moving_piece: The piece being captured and moved off.

        Returns:
            List of (row, col) integer tuples for the path, or None.
        """
        # Create obstacle grid, ignoring the piece being moved
        obstacle_grid = self._create_obstacle_grid(board_state, moving_piece)

        # Define pathfinding grid boundaries including off-board target area
        # Assuming target is off the right edge
        min_r, max_r = 0, self.board_squares
        min_c, max_c = 0, self.board_squares + 2 # Allow 2 cols off right edge

        # Run A* allowing movement within the extended bounds
        return self._a_star_search(start_pos_rc, target_off_board_rc, obstacle_grid,
                                  grid_bounds=(min_r, max_r, min_c, max_c))


    def _create_obstacle_grid(self, board_state, ignore_piece: ChessPiece = None):
        """
        Create a 2D boolean grid where True indicates an obstacle.

        Args:
            board_state: List of ChessPiece objects.
            ignore_piece: The piece whose position should not be marked as an obstacle.

        Returns:
            A 2D list (rows x cols) of booleans.
        """
        grid = [[False for _ in range(self.board_squares)] for _ in range(self.board_squares)]

        for piece in board_state:
            # Obstacles are other *active* pieces
            if piece.active and piece != ignore_piece:
                # Get the integer square the piece occupies
                col_f, row_f = piece.position # col, row float
                r_idx, c_idx = int(round(row_f)), int(round(col_f)) # row, col integer index

                # Mark the primary square as an obstacle
                if 0 <= r_idx < self.board_squares and 0 <= c_idx < self.board_squares:
                    grid[r_idx][c_idx] = True

                # Optional: Mark adjacent squares too for larger pieces or more caution
                # radius_sq = (piece.diameter / piece.square_size * 0.5)**2 # Radius in squares
                # if radius_sq > 0.2: # If piece radius is significant
                #     for dr in [-1, 0, 1]:
                #         for dc in [-1, 0, 1]:
                #             if dr == 0 and dc == 0: continue
                #             nr, nc = r_idx + dr, c_idx + dc
                #             if 0 <= nr < self.board_squares and 0 <= nc < self.board_squares:
                #                 grid[nr][nc] = True # Mark neighbors
        return grid


    def _a_star_search(self, start_node_rc, end_node_rc, obstacle_grid, grid_bounds=None):
        """
        Core A* implementation.

        Args:
            start_node_rc: Start (row, col).
            end_node_rc: Goal (row, col).
            obstacle_grid: 2D boolean grid (True=obstacle).
            grid_bounds: Optional tuple (min_r, max_r, min_c, max_c) for search area.

        Returns:
            List of (row, col) path nodes or None.
        """
        if grid_bounds:
            min_r, max_r, min_c, max_c = grid_bounds
        else:
            min_r, max_r = 0, self.board_squares
            min_c, max_c = 0, self.board_squares

        open_set = [(0, start_node_rc)] # (f_score, node)
        came_from = {}
        g_score = {start_node_rc: 0}
        f_score = {start_node_rc: self._heuristic(start_node_rc, end_node_rc)}

        max_iterations = 500 # Prevent excessively long searches
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            current_f, current_rc = heapq.heappop(open_set)

            if current_rc == end_node_rc:
                return self._reconstruct_path(came_from, current_rc)

            for neighbor_rc in self._get_neighbors(current_rc, grid_bounds):
                r, c = neighbor_rc

                # Check obstacle grid (only if within standard board bounds)
                if 0 <= r < self.board_squares and 0 <= c < self.board_squares:
                    if obstacle_grid[r][c]:
                        continue # Skip obstacles on the board

                # Calculate cost to neighbor
                move_cost = 1.414 if abs(r - current_rc[0]) + abs(c - current_rc[1]) == 2 else 1.0
                tentative_g_score = g_score.get(current_rc, float('inf')) + move_cost

                if tentative_g_score < g_score.get(neighbor_rc, float('inf')):
                    came_from[neighbor_rc] = current_rc
                    g_score[neighbor_rc] = tentative_g_score
                    f_score[neighbor_rc] = tentative_g_score + self._heuristic(neighbor_rc, end_node_rc)
                    heapq.heappush(open_set, (f_score[neighbor_rc], neighbor_rc))

        print(f"A* Warning: No path found from {start_node_rc} to {end_node_rc} after {iterations} iterations.")
        return None # No path found

    def _get_neighbors(self, node_rc, grid_bounds):
        """Get valid neighbor nodes (8 directions) within specified bounds."""
        r, c = node_rc
        min_r, max_r, min_c, max_c = grid_bounds
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if min_r <= nr < max_r and min_c <= nc < max_c:
                    neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, node_a_rc, node_b_rc):
        """Manhattan distance heuristic."""
        return abs(node_a_rc[0] - node_b_rc[0]) + abs(node_a_rc[1] - node_b_rc[1])

    def _reconstruct_path(self, came_from, current_rc):
        """Build the path backwards from the goal."""
        path = [current_rc]
        while current_rc in came_from:
            current_rc = came_from[current_rc]
            path.append(current_rc)
        path.reverse()
        return path

    # Note: Removed _create_knight_path and _create_direct_path as fallbacks/alternatives
    # A* should handle pathing, knight jumps are handled by not pathfinding them,
    # and capture paths use A* with extended bounds. If A* truly fails, returning None is appropriate.