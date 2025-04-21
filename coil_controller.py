# coil_controller.py

import numpy as np
import math
import os # For path checking/deletion in plot_heatmap

# Pygame used for drawing simulation state (optional visualization)
import pygame

# Matplotlib and SciPy used for generating heatmap visualization
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# --- Constants ---
DARK_GRAY = (100, 100, 100); RED = (255, 0, 0); BLUE = (0, 0, 255)
# Example: Maximum current (Amps) a single coil draws at 100% power.
# Adjust based on your hardware specifications.
MAX_COIL_AMPS = 0.5

class CoilGrid:
    """
    Simulates a grid of electromagnets under the chessboard.
    Handles coil activation patterns, calculates the resulting magnetic field,
    and provides methods for visualization (coils, field vectors, heatmap).
    """
    def __init__(self, size=20, board_squares=8):
        """
        Initializes the coil grid.

        Args:
            size (int): The dimension of the square coil grid (e.g., 20 for 20x20).
            board_squares (int): The number of squares along one side of the chessboard (typically 8).
        """
        if size <= 0 or board_squares <= 0:
            raise ValueError("Grid size and board squares must be positive.")
        self.size = size
        self.board_squares = board_squares

        # --- Grid State ---
        self.coil_power = np.zeros((size, size), dtype=float)
        self.coil_current = np.zeros((size, size), dtype=int)
        self.magnetic_field = np.zeros((size, size, 2), dtype=float)

    def reset(self):
        """Resets all coil powers, currents, and the calculated magnetic field to zero."""
        self.coil_power.fill(0)
        self.coil_current.fill(0)
        self.magnetic_field.fill(0)

    def update_coil(self, row, col, power, current_direction=1):
        """
        Sets the power and current direction for a single coil.

        Args:
            row (int): Coil row index.
            col (int): Coil column index.
            power (float): Desired power level (0-100). Will be clipped.
            current_direction (int): Desired current direction (+1 or -1).
        """
        if 0 <= row < self.size and 0 <= col < self.size:
            safe_power = np.clip(power, 0, 100)
            self.coil_power[row, col] = safe_power
            self.coil_current[row, col] = np.sign(current_direction) if safe_power > 0 else 0

    def activate_coil_pattern(self, pattern_type, position, target=None,
                              intensity=100, radius=4, blocked_coils=None):
        """
        Activates a predefined pattern of coils centered around a position,
        potentially directed towards a target, avoiding blocked coils.
        Used for visualization and sending commands to hardware interface.
        """
        if blocked_coils is None:
            blocked_coils = set()

        center_x, center_y = position # Coil grid coordinates
        center_x_int = int(round(center_x))
        center_y_int = int(round(center_y))

        search_radius = radius * 2
        min_r_loop = max(0, center_y_int - search_radius)
        max_r_loop = min(self.size, center_y_int + search_radius + 1)
        min_c_loop = max(0, center_x_int - search_radius)
        max_c_loop = min(self.size, center_x_int + search_radius + 1)

        # Reset grid state before applying the new pattern
        self.reset()

        # Helper to update coil only if valid and not blocked
        def safe_update_coil(r, c, power, current):
            if 0 <= r < self.size and 0 <= c < self.size and (r, c) not in blocked_coils:
                 self.update_coil(r, c, power, current)

        # --- Pattern Implementations ---
        if pattern_type == "radial":
             for r in range(min_r_loop, max_r_loop):
                 for c in range(min_c_loop, max_c_loop):
                     distance = math.sqrt((r - center_y)**2 + (c - center_x)**2)
                     if distance <= radius:
                         power = intensity * max(0, (1 - distance / radius))**2
                         safe_update_coil(r, c, power, -1) # Assume attract

        elif pattern_type == "straight_horizontal":
             if target is None: return
             target_x, _ = target
             direction_x = np.sign(target_x - center_x) if abs(target_x - center_x) > 1e-3 else 0
             current_row = center_y_int
             if 0 <= current_row < self.size:
                 for c in range(min_c_loop, max_c_loop):
                      proj = c - center_x; distance_from_center = abs(proj)
                      if distance_from_center <= radius * 1.5:
                           power_factor = max(0, 1 - distance_from_center / (radius * 1.5))**2
                           power = intensity * power_factor
                           if proj * direction_x > 0.1: current = -1
                           elif proj * direction_x < -0.1: current = 1
                           else: current = 0; power = 0
                           safe_update_coil(current_row, c, power, current)

        elif pattern_type == "straight_vertical":
             if target is None: return
             _, target_y = target
             direction_y = np.sign(target_y - center_y) if abs(target_y - center_y) > 1e-3 else 0
             current_col = center_x_int
             if 0 <= current_col < self.size:
                 for r in range(min_r_loop, max_r_loop):
                      proj = r - center_y; distance_from_center = abs(proj)
                      if distance_from_center <= radius * 1.5:
                           power_factor = max(0, 1 - distance_from_center / (radius * 1.5))**2
                           power = intensity * power_factor
                           if proj * direction_y > 0.1: current = -1
                           elif proj * direction_y < -0.1: current = 1
                           else: current = 0; power = 0
                           safe_update_coil(r, current_col, power, current)

        elif pattern_type == "directed" and target is not None:
            target_x, target_y = target
            direction_vec = np.array([target_x - center_x, target_y - center_y])
            distance_to_target = np.linalg.norm(direction_vec)
            if distance_to_target < 0.1:
                 self.activate_coil_pattern("radial", position, target, intensity * 0.5, radius, blocked_coils)
                 return
            direction_norm = direction_vec / distance_to_target
            direction_focus_factor = 1.0
            for r in range(min_r_loop, max_r_loop):
                for c in range(min_c_loop, max_c_loop):
                    rel_pos = np.array([c - center_x, r - center_y])
                    distance = np.linalg.norm(rel_pos)
                    if distance <= radius * 1.5:
                        proj = np.dot(rel_pos, direction_norm)
                        perp_dist_sq = max(0, distance**2 - proj**2)
                        perp_distance = math.sqrt(perp_dist_sq)
                        power_factor = max(0, 1 - distance / (radius * 1.5))**2
                        direction_focus = max(0, 1 - perp_distance / (radius * 0.7))**direction_focus_factor
                        power = intensity * power_factor * direction_focus
                        proj_threshold = 0.15 * radius
                        if proj > proj_threshold: current = -1
                        elif proj < -proj_threshold: current = 1
                        else: current = -1; power *= 0.1
                        power = max(0, power)
                        safe_update_coil(r, c, power, current)

        elif pattern_type == "knight" and target is not None:
             self.activate_coil_pattern("directed", position, target, intensity, radius, blocked_coils)

        else: # Default or unknown pattern
            print(f"Warning: Unknown coil pattern type '{pattern_type}'. Using radial.")
            self.activate_coil_pattern("radial", position, target, intensity * 0.5, radius, blocked_coils)


    def update_magnetic_field(self):
        """
        Calculates the simulated magnetic field vector at each grid point based on
        the current state of coil_power and coil_current. Uses a vectorized approach.
        """
        self.magnetic_field.fill(0); influence_radius_sq = 8**2
        active_coils_indices = np.argwhere(self.coil_power > 0)
        if active_coils_indices.size == 0: return
        active_coil_pos = np.array([[c, r] for r, c in active_coils_indices])
        active_powers = self.coil_power[active_coils_indices[:, 0], active_coils_indices[:, 1]]
        active_currents = self.coil_current[active_coils_indices[:, 0], active_coils_indices[:, 1]]
        grid_y, grid_x = np.indices((self.size, self.size)); grid_pos = np.stack((grid_x, grid_y), axis=-1)
        vec_coil_to_field = grid_pos[:, :, np.newaxis, :] - active_coil_pos[np.newaxis, np.newaxis, :, :]
        dist_sq = np.sum(vec_coil_to_field**2, axis=3); dist_sq[dist_sq == 0] = 1e-6
        within_influence = dist_sq <= influence_radius_sq; dist = np.sqrt(dist_sq)
        direction_vec = vec_coil_to_field / dist[..., np.newaxis]
        strength = active_powers * (1.0 / (1.0 + dist_sq))
        field_contribution = direction_vec * (strength * active_currents)[..., np.newaxis]
        masked_contribution = field_contribution * within_influence[..., np.newaxis]
        self.magnetic_field = np.sum(masked_contribution, axis=2)


    def calculate_force(self, piece_position, piece_magnet_strength):
        """
        Calculate the magnetic force vector on a piece at a given board position.
        Uses bilinear interpolation on the pre-calculated magnetic_field.
        """
        col_board, row_board = piece_position; col_grid = col_board * (self.size / self.board_squares); row_grid = row_board * (self.size / self.board_squares)
        col_idx = int(col_grid); row_idx = int(row_grid); dx = col_grid - col_idx; dy = row_grid - row_idx
        if not (0 <= col_idx < self.size - 1 and 0 <= row_idx < self.size - 1):
            col_idx_safe = np.clip(int(round(col_grid)), 0, self.size - 1); row_idx_safe = np.clip(int(round(row_grid)), 0, self.size - 1)
            interpolated_field = self.magnetic_field[row_idx_safe, col_idx_safe]
        else:
            field_00=self.magnetic_field[row_idx, col_idx]; field_01=self.magnetic_field[row_idx, col_idx + 1]; field_10=self.magnetic_field[row_idx + 1, col_idx]; field_11=self.magnetic_field[row_idx + 1, col_idx + 1]
            interp_x_top = (1 - dx) * field_00[0] + dx * field_01[0]; interp_x_bottom = (1 - dx) * field_10[0] + dx * field_11[0]; field_x = (1 - dy) * interp_x_top + dy * interp_x_bottom
            interp_y_top = (1 - dx) * field_00[1] + dx * field_01[1]; interp_y_bottom = (1 - dx) * field_10[1] + dx * field_11[1]; field_y = (1 - dy) * interp_y_top + dy * interp_y_bottom
            interpolated_field = np.array([field_x, field_y])
        force = interpolated_field * piece_magnet_strength
        return force


    def calculate_total_current(self):
        """Calculates the total estimated current draw based on coil power levels."""
        active_powers = self.coil_power[self.coil_power > 0]
        total_amps = np.sum(active_powers / 100.0 * MAX_COIL_AMPS)
        return total_amps


    def draw(self, surface, board_pixel_size, x_offset=0):
        """Draws the coil grid visualization onto the provided surface."""
        coil_pixel_size = board_pixel_size / self.size
        coil_surface = pygame.Surface((board_pixel_size, board_pixel_size), pygame.SRCALPHA)
        for r in range(self.size):
            for c in range(self.size):
                x = c * coil_pixel_size + coil_pixel_size / 2; y = r * coil_pixel_size + coil_pixel_size / 2
                power = self.coil_power[r, c]; current = self.coil_current[r, c]
                outline_color = (*DARK_GRAY[:3], 100)
                pygame.draw.circle(coil_surface, outline_color, (int(x), int(y)), int(coil_pixel_size / 2 * 0.8), 1)
                if power > 0:
                    alpha = int(np.clip(power * 2.0, 50, 200))
                    color = (255, 100, 100, alpha) if current > 0 else (100, 100, 255, alpha)
                    radius = int(coil_pixel_size / 2 * 0.7 * (0.6 + 0.4 * power / 100))
                    pygame.draw.circle(coil_surface, color, (int(x), int(y)), radius)
        surface.blit(coil_surface, (x_offset, 0))


    def draw_field_overlay(self, surface, board_pixel_size, resolution=20, x_offset=0):
        """Draws the calculated magnetic field vectors as arrows."""
        step_size = board_pixel_size / resolution
        field_surface = pygame.Surface((board_pixel_size, board_pixel_size), pygame.SRCALPHA)
        field_magnitudes = np.linalg.norm(self.magnetic_field, axis=2)
        max_field_strength_observed = field_magnitudes.max() if field_magnitudes.max() > 1e-6 else 1.0

        for r_idx in range(resolution):
            for c_idx in range(resolution):
                x_pix = c_idx * step_size + step_size / 2
                y_pix = r_idx * step_size + step_size / 2
                col_grid = c_idx * (self.size / resolution) + (self.size / resolution / 2)
                row_grid = r_idx * (self.size / resolution) + (self.size / resolution / 2)

                col_int = int(col_grid)
                row_int = int(row_grid)
                # --- FIX: Calculate BOTH dx and dy ---
                dx = col_grid - col_int
                dy = row_grid - row_int
                # --- -------------------------- ---

                if not (0 <= col_int < self.size - 1 and 0 <= row_int < self.size - 1): continue

                # Interpolate field vector at this point
                field_00=self.magnetic_field[row_int, col_int]; field_01=self.magnetic_field[row_int, col_int + 1]
                field_10=self.magnetic_field[row_int + 1, col_int]; field_11=self.magnetic_field[row_int + 1, col_int + 1]
                interp_x_top=(1-dx)*field_00[0]+dx*field_01[0]; interp_x_bottom=(1-dx)*field_10[0]+dx*field_11[0]; field_x=(1-dy)*interp_x_top+dy*interp_x_bottom
                interp_y_top=(1-dx)*field_00[1]+dx*field_01[1]; interp_y_bottom=(1-dx)*field_10[1]+dx*field_11[1]; field_y=(1-dy)*interp_y_top+dy*interp_y_bottom
                field_vec = np.array([field_x, field_y])
                field_strength = np.linalg.norm(field_vec)
                min_draw_strength = 0.01 * max_field_strength_observed

                if field_strength > min_draw_strength:
                    # Arrow drawing logic
                    field_normalized = field_vec / field_strength
                    log_strength = np.log1p(field_strength / max_field_strength_observed * 10)
                    max_arrow_len = step_size * 0.7; arrow_len = min(log_strength * max_arrow_len / np.log1p(10), max_arrow_len); arrow_len = max(2, arrow_len)
                    end_x = x_pix + field_normalized[0] * arrow_len; end_y = y_pix + field_normalized[1] * arrow_len
                    color_intensity = min(1.0, field_strength / max_field_strength_observed)
                    arrow_color_rgb = (255, 255 * (1 - color_intensity**0.5), 255 * (1 - color_intensity**0.5))
                    alpha = int(np.clip(100 + 155 * color_intensity, 100, 255)); arrow_color = (*arrow_color_rgb, alpha)
                    start_point = (int(x_pix), int(y_pix)); end_point = (int(end_x), int(end_y))
                    pygame.draw.line(field_surface, arrow_color, start_point, end_point, 1)
                    head_length = min(5, arrow_len * 0.4)
                    if head_length > 2:
                        angle = math.atan2(field_normalized[1], field_normalized[0])
                        p1 = (end_x, end_y); p2 = (end_x - head_length * math.cos(angle - math.pi / 6), end_y - head_length * math.sin(angle - math.pi / 6))
                        p3 = (end_x - head_length * math.cos(angle + math.pi / 6), end_y - head_length * math.sin(angle + math.pi / 6))
                        try: poly_points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]; pygame.draw.polygon(field_surface, arrow_color, poly_points)
                        except ValueError: pass
        surface.blit(field_surface, (x_offset, 0))


    # --- REVERTED generate_heatmap ---
    def generate_heatmap(self, resolution=200):
        """
        Generates heatmap data using manual bilinear interpolation.
        Applies Gaussian filter after interpolation and normalization.
        (Reverted to logic matching previous visually correct output)
        """
        try:
            heatmap = np.zeros((resolution, resolution))
            # Calculate magnitudes from the pre-computed field
            field_magnitudes = np.linalg.norm(self.magnetic_field, axis=2)
            # Find max magnitude *before* interpolation/filtering for normalization
            max_mag = field_magnitudes.max()
            # Use a small epsilon to prevent division by zero if field is completely flat zero
            epsilon = 1e-9
            if max_mag < epsilon:
                max_mag = epsilon # Avoid division by zero

            for r in range(resolution):
                for c in range(resolution):
                    # Map heatmap pixel (r, c) to coil grid coordinates (grid_r, grid_c)
                    # Ensure mapping covers the full grid range [0, size-1]
                    grid_c = c * (self.size - 1) / (resolution - 1) if resolution > 1 else 0
                    grid_r = r * (self.size - 1) / (resolution - 1) if resolution > 1 else 0
                    col_idx, row_idx = int(grid_c), int(grid_r)

                    # Perform bilinear interpolation using the original field_magnitudes
                    if 0 <= col_idx < self.size - 1 and 0 <= row_idx < self.size - 1:
                        dx, dy = grid_c - col_idx, grid_r - row_idx
                        f00 = field_magnitudes[row_idx, col_idx]
                        f01 = field_magnitudes[row_idx, col_idx + 1]
                        f10 = field_magnitudes[row_idx + 1, col_idx]
                        f11 = field_magnitudes[row_idx + 1, col_idx + 1]
                        interp_top = (1 - dx) * f00 + dx * f01
                        interp_bottom = (1 - dx) * f10 + dx * f11
                        heatmap[r, c] = (1 - dy) * interp_top + dy * interp_bottom
                    elif 0 <= col_idx < self.size and 0 <= row_idx < self.size:
                        # Fallback for edges: Use nearest neighbor value
                        heatmap[r, c] = field_magnitudes[row_idx, col_idx]
                    # else: leave as 0 (outside original grid)

            # Normalize the interpolated heatmap data
            heatmap /= max_mag

            # Apply Gaussian filter for smoothness AFTER normalization
            sigma_val = max(1.0, resolution / 100.0) # Sigma relative to heatmap resolution
            heatmap = gaussian_filter(heatmap, sigma=sigma_val, order=0, mode='constant', cval=0.0)

            # Clip values strictly between 0 and 1 AFTER filtering
            heatmap = np.clip(heatmap, 0.0, 1.0)

            # Final check for NaNs (should be less likely but safe)
            if np.isnan(heatmap).any():
                print("Warning: NaNs detected in heatmap data after final processing. Replacing with 0.")
                heatmap = np.nan_to_num(heatmap, nan=0.0)

            return heatmap
        except Exception as e:
            print(f"Error during heatmap generation: {e}")
            return np.zeros((resolution, resolution)) # Return blank on error
    # --- END REVERTED generate_heatmap ---


    def plot_heatmap(self, filename="field_heatmap.png", figsize=(8, 8)):
        """Generates and saves a heatmap plot using matplotlib."""
        try:
            heatmap_data = self.generate_heatmap()
            if heatmap_data is None: return None

            plt.figure(figsize=figsize)
            cmap = plt.cm.jet # Use 'jet' colormap as preferred
            im = plt.imshow(heatmap_data, cmap=cmap, aspect='equal', origin='upper', interpolation='bilinear', vmin=0.0, vmax=1.0)
            plt.colorbar(im, label='Normalized Field Strength')
            plt.xticks([]); plt.yticks([])
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close()
            return filename
        except Exception as e:
            print(f"Error generating or saving heatmap plot: {e}")
            if os.path.exists(filename):
                try: os.remove(filename)
                except OSError: pass
            return None