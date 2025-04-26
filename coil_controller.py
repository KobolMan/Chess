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
        Includes robust error handling to prevent NaN values.
        """
        # Clear previous field
        self.magnetic_field.fill(0)
        
        # Define influence radius for optimization
        influence_radius_sq = 8**2
        
        # Find active coils (non-zero power)
        active_coils_indices = np.argwhere(self.coil_power > 0)
        if active_coils_indices.size == 0:
            # No active coils, return early with zeroed field
            return
        
        # Extract active coil positions, powers, and currents
        active_coil_pos = np.array([[c, r] for r, c in active_coils_indices])
        active_powers = self.coil_power[active_coils_indices[:, 0], active_coils_indices[:, 1]]
        active_currents = self.coil_current[active_coils_indices[:, 0], active_coils_indices[:, 1]]
        
        # Create grid positions
        grid_y, grid_x = np.indices((self.size, self.size))
        grid_pos = np.stack((grid_x, grid_y), axis=-1)
        
        # Calculate vectors from each coil to each grid point
        vec_coil_to_field = grid_pos[:, :, np.newaxis, :] - active_coil_pos[np.newaxis, np.newaxis, :, :]
        
        # Calculate squared distances
        dist_sq = np.sum(vec_coil_to_field**2, axis=3)
        
        # Avoid division by zero - replace zeros with a small value
        # Use a larger epsilon to prevent very small values that could cause instability
        epsilon = 1e-5
        dist_sq = np.maximum(dist_sq, epsilon)
        
        # Determine which grid points are within influence radius of coils
        within_influence = dist_sq <= influence_radius_sq
        
        # Calculate distances
        dist = np.sqrt(dist_sq)
        
        # Normalize vectors to get direction
        # Explicitly handle division to prevent NaN values
        direction_vec = np.zeros_like(vec_coil_to_field, dtype=float)
        np.divide(vec_coil_to_field, dist[..., np.newaxis], out=direction_vec, where=dist[..., np.newaxis] > epsilon)
        
        # Calculate field strength based on distance
        strength = active_powers * (1.0 / (1.0 + dist_sq))
        
        # Calculate field contribution from each coil
        field_contribution = direction_vec * (strength * active_currents)[..., np.newaxis]
        
        # Mask contributions to only include those within influence radius
        masked_contribution = field_contribution * within_influence[..., np.newaxis]
        
        # Sum all contributions
        field_sum = np.sum(masked_contribution, axis=2)
        
        # Final safety check for NaN values - replace with zeros if any remain
        self.magnetic_field = np.nan_to_num(field_sum, nan=0.0)

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

    def generate_heatmap(self, resolution=200):
        """
        Generates heatmap data using robust bilinear interpolation with
        comprehensive error handling.
        """
        try:
            # Create empty heatmap array
            heatmap = np.zeros((resolution, resolution))
            
            # Immediately check if magnetic field is all zeros - return blank if so
            if np.max(np.abs(self.magnetic_field)) < 1e-10:
                print("Field is effectively zero - creating blank heatmap")
                return heatmap
            
            # Calculate field magnitudes - using safe L2 norm
            field_magnitudes = np.sqrt(np.sum(self.magnetic_field * self.magnetic_field, axis=2) + 1e-10)
            
            # Check if all magnitudes are very small
            max_mag = field_magnitudes.max()
            if max_mag < 1e-4:  # Use a larger threshold
                print(f"Maximum field magnitude too small ({max_mag:.2e}) - creating blank heatmap")
                return heatmap
            
            # Create a small sample version of field_magnitudes for debugging
            sample_size = min(5, self.size)
            sample = field_magnitudes[:sample_size, :sample_size]
            print(f"Field magnitudes sample (first {sample_size}x{sample_size}):\n{sample}")
            
            # Safety check for NaN in input
            if np.isnan(field_magnitudes).any():
                print("NaN values in field magnitudes - cleaning up before processing")
                field_magnitudes = np.nan_to_num(field_magnitudes, nan=0.0)
            
            # Bilinear interpolation with careful bounds checking
            for r in range(resolution):
                for c in range(resolution):
                    # Calculate corresponding position in field array
                    grid_c = c * (self.size - 1) / (resolution - 1) if resolution > 1 else 0
                    grid_r = r * (self.size - 1) / (resolution - 1) if resolution > 1 else 0
                    
                    # Get integer indices and interpolation factors
                    col_idx, row_idx = int(grid_c), int(grid_r)
                    dx, dy = grid_c - col_idx, grid_r - row_idx
                    
                    # Safe interpolation with bounds checking
                    if 0 <= col_idx < self.size - 1 and 0 <= row_idx < self.size - 1:
                        # Regular bilinear interpolation
                        f00 = field_magnitudes[row_idx, col_idx]
                        f01 = field_magnitudes[row_idx, col_idx + 1]
                        f10 = field_magnitudes[row_idx + 1, col_idx]
                        f11 = field_magnitudes[row_idx + 1, col_idx + 1]
                        
                        # Check for NaN values from a problematic magnetic_field calculation
                        if np.isnan([f00, f01, f10, f11]).any():
                            # If any corner is NaN, use nearest valid neighbor
                            valid_values = [v for v in [f00, f01, f10, f11] if not np.isnan(v)]
                            heatmap[r, c] = np.mean(valid_values) if valid_values else 0.0
                        else:
                            # Normal bilinear interpolation
                            interp_top = (1 - dx) * f00 + dx * f01
                            interp_bottom = (1 - dx) * f10 + dx * f11
                            heatmap[r, c] = (1 - dy) * interp_top + dy * interp_bottom
                    elif 0 <= col_idx < self.size and 0 <= row_idx < self.size:
                        # Edge case - use nearest value
                        heatmap[r, c] = field_magnitudes[row_idx, col_idx]
                    # else: leave as 0 (outside grid)
            
            # Normalize heatmap - with safety checks
            max_val = heatmap.max()
            if max_val > 1e-6:
                heatmap /= max_val
            else:
                print("Warning: Maximum heatmap value too small for normalization")
                # Return blank heatmap instead of potentially unstable values
                return np.zeros((resolution, resolution))
            
            # Apply Gaussian smoothing for a nicer visualization
            sigma_val = max(1.0, resolution / 100.0)
            try:
                smoothed = gaussian_filter(heatmap, sigma=sigma_val, mode='constant', cval=0.0)
                heatmap = smoothed
            except Exception as e:
                print(f"Warning: Gaussian smoothing failed: {e}")
                # Continue with unsmoothed heatmap
            
            # Final clipping and NaN check
            heatmap = np.clip(heatmap, 0.0, 1.0)
            if np.isnan(heatmap).any():
                print("NaN values in final heatmap - replacing with zeros")
                heatmap = np.nan_to_num(heatmap, nan=0.0)
            
            return heatmap
        except Exception as e:
            print(f"Error during heatmap generation: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((resolution, resolution))

    def plot_heatmap(self, filename="field_heatmap.png", figsize=(8, 8)):
        """Generates and saves a heatmap plot using matplotlib with enhanced reliability."""
        try:
            # Generate heatmap data
            heatmap_data = self.generate_heatmap()
            if heatmap_data is None:
                print("Failed to generate heatmap data")
                return None
            
            # Check if heatmap is all zeros
            if np.max(heatmap_data) < 1e-6:
                print("Heatmap is effectively blank, but continuing with visualization")
                # Instead of returning None, we'll still create a blue visualization
            
            # Create plot
            plt.figure(figsize=figsize)
            
            # Use plasma colormap (blues for low values, reds for high)
            # Alternative colormaps: 'viridis', 'jet', 'hot', 'magma'
            cmap = plt.cm.plasma
            
            # Plot the heatmap
            im = plt.imshow(
                heatmap_data, 
                cmap=cmap, 
                aspect='equal', 
                origin='upper', 
                interpolation='bilinear', 
                vmin=0.0, 
                vmax=1.0
            )
            
            # Add colorbar and labels
            plt.colorbar(im, label='Normalized Field Strength')
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            
            # Save to file with error handling
            try:
                plt.savefig(filename, dpi=150)
                plt.close()
                
                # Verify file was created successfully
                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    return filename
                else:
                    print(f"Warning: Heatmap file {filename} not created or empty")
                    return None
            except Exception as e:
                print(f"Error saving heatmap to file: {e}")
                plt.close()
                return None
        except Exception as e:
            print(f"Error in heatmap plotting: {e}")
            import traceback
            traceback.print_exc()
            if os.path.exists(filename):
                try: os.remove(filename)
                except OSError: pass
            return None