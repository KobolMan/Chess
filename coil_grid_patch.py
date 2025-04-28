# coil_grid_patch.py
import types
import numpy as np

def patch_chess_simulation(board_instance):
    """
    Apply patches to the ChessBoard instance to use the optimized layout.
    
    Args:
        board_instance: The ChessBoard instance to patch
    """
    from optimized_coil_layout import OptimizedCoilLayout
    
    # Create the optimized layout
    optimized_layout = OptimizedCoilLayout(
        board_squares=board_instance.coil_grid.board_squares,
        coil_grid_size=board_instance.coil_grid.size
    )
    
    # Store the optimized layout in the board instance
    board_instance.optimized_layout = optimized_layout
    
    # Store the original update_move method for later use
    original_update_move = board_instance.update_move
    
    # Override the update_move method to use optimized layout
    def patched_update_move(self, dt):
        if not self.move_in_progress:
            self.last_move_pid_force_mag = 0.0
            self.current_total_sim_amps = self.coil_grid.calculate_total_current()
            return
        
        effective_dt = dt * self.simulation_speed
        if effective_dt <= 0: return
        
        if self.selected_piece and self.target_position:
            # Run the original PID calculation
            current_pos = self.selected_piece.position.copy()
            current_vel = self.selected_piece.velocity.copy()
            target_pos = np.array(self.target_position)
            error = target_pos - current_pos
            distance_to_target = np.linalg.norm(error)
            
            # Dynamic Gain Scaling
            gain_scale_distance = 1.5
            min_gain_scale = 0.35
            scale_factor = np.clip(distance_to_target / gain_scale_distance, min_gain_scale, 1.0)
            effective_kp = self.pid_kp * scale_factor
            effective_kd = self.pid_kd * scale_factor
            effective_ki = self.pid_ki
            
            # PID Terms
            p_term = effective_kp * error
            if effective_ki > 1e-6:
                self.pid_integral += error * effective_dt
                integral_mag = np.linalg.norm(self.pid_integral)
                if integral_mag > self.pid_integral_max:
                    self.pid_integral = self.pid_integral * (self.pid_integral_max / integral_mag)
                i_term = effective_ki * self.pid_integral
            else:
                i_term = np.array([0.0, 0.0])
            
            d_term = -effective_kd * current_vel
            terminal_zone = 0.4
            if distance_to_target < terminal_zone:
                terminal_factor = 1.0 - (distance_to_target / terminal_zone)
                terminal_damping_force = -self.terminal_damping * current_vel * terminal_factor
                d_term += terminal_damping_force
            
            # Total PID Force
            pid_force = p_term + i_term + d_term
            self.last_move_pid_force_mag = np.linalg.norm(pid_force)
            max_pid_force = 5000.0
            if self.last_move_pid_force_mag > max_pid_force:
                pid_force = pid_force * (max_pid_force / self.last_move_pid_force_mag)
                self.last_move_pid_force_mag = max_pid_force
                if self.debug_mode: print("  PID Force Clamped!")
            
            # Debug Output
            if self.debug_mode:
                print(f"\n--- Update Step dt={effective_dt:.4f} ---")
                print(f"  Piece: {self.selected_piece.symbol} Pos:({current_pos[0]:.2f},{current_pos[1]:.2f}) " +
                      f"Vel:({current_vel[0]:.2f},{current_vel[1]:.2f}) Dist:{distance_to_target:.3f}")
                print(f"  Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f})")
                print(f"  PID Error: ({error[0]:.3f},{error[1]:.3f})")
                print(f"  Gain Scale Factor: {scale_factor:.2f} (Min: {min_gain_scale})")
                print(f"  PID Terms: P:({p_term[0]:.1f},{p_term[1]:.1f}) " +
                      f"I:({i_term[0]:.1f},{i_term[1]:.1f}) D:({d_term[0]:.1f},{d_term[1]:.1f})")
                print(f"  PID Force: ({pid_force[0]:.2f},{pid_force[1]:.2f}) Mag: {self.last_move_pid_force_mag:.2f}")
            
            # Stop Condition
            stop_threshold = 0.02
            velocity_threshold = 0.03
            force_threshold = 0.1
            settled_condition = (self.last_move_pid_force_mag < force_threshold and
                                np.linalg.norm(current_vel) < velocity_threshold * 2)
            move_finished = (distance_to_target < stop_threshold and np.linalg.norm(current_vel) < velocity_threshold) or \
                            (distance_to_target < stop_threshold * 2 and settled_condition)
            
            if move_finished:
                final_pos_before_snap = self.selected_piece.position.copy()
                final_vel_before_snap = self.selected_piece.velocity.copy()
                self.selected_piece.position = target_pos.copy()
                self.selected_piece.velocity.fill(0.0)
                self.pid_integral.fill(0.0)
                
                if self.debug_mode:
                    print(f"STOP Condition Met: Dist {distance_to_target:.3f} (<{stop_threshold}), " +
                          f"Vel {np.linalg.norm(current_vel):.3f} (<{velocity_threshold}), " +
                          f"Force {self.last_move_pid_force_mag:.2f} (<{force_threshold})")
                
                print(f"Move complete. Snapped from ({final_pos_before_snap[0]:.3f},{final_pos_before_snap[1]:.3f}) " +
                      f"Vel ({final_vel_before_snap[0]:.3f},{final_vel_before_snap[1]:.3f})")
                print(f"Final Position: ({self.selected_piece.position[0]},{self.selected_piece.position[1]}) " +
                      f"Velocity: ({self.selected_piece.velocity[0]},{self.selected_piece.velocity[1]})")
                
                self.move_in_progress = False
                self.move_complete = True
                self.target_position = None
                self.last_move_pid_force_mag = 0.0
                self.coil_grid.reset()
                self.hardware_controller.reset_all_coils()
                self.heatmap_needs_update = True
                self.current_total_sim_amps = 0.0
                
                for piece in self.temporarily_moved_pieces:
                    piece.return_from_temporary_move()
                self.temporarily_moved_pieces = []
            else:
                # Move In Progress - Apply force to the piece
                self.selected_piece.apply_force(pid_force, effective_dt)
                
                if self.debug_mode:
                    print(f"  End Pos:({self.selected_piece.position[0]:.2f},{self.selected_piece.position[1]:.2f}) " +
                          f"End Vel:({self.selected_piece.velocity[0]:.2f},{self.selected_piece.velocity[1]:.2f})")
                
                # Update Coil Simulation & Hardware
                self.field_update_timer += effective_dt
                if self.field_update_timer >= self.field_update_interval:
                    self.field_update_timer = 0
                    
                    # Create keep-out mask for stationary pieces
                    blocked_coils_set = self._create_keep_out_mask()
                    
                    # Get current and target positions
                    current_coil_pos = self.selected_piece.get_coil_position()
                    target_coil_pos = tuple(np.array(target_pos) * (self.coil_grid.size / self.coil_grid.board_squares))
                    
                    # Determine movement pattern
                    dx_board = target_pos[0] - current_pos[0]
                    dy_board = target_pos[1] - current_pos[1]
                    is_knight_shape = (abs(round(dx_board)) == 1 and abs(round(dy_board)) == 2) or \
                                     (abs(round(dx_board)) == 2 and abs(round(dy_board)) == 1)
                    
                    chosen_pattern = self.current_pattern
                    straight_threshold = 0.1
                    
                    if self.selected_piece.piece_type.name == "KNIGHT":
                        if is_knight_shape and self.current_pattern in ["knight", "directed"]:
                            chosen_pattern = "knight"
                    elif abs(dx_board) < straight_threshold and abs(dy_board) > straight_threshold:
                        chosen_pattern = "straight_vertical"
                    elif abs(dy_board) < straight_threshold and abs(dx_board) > straight_threshold:
                        chosen_pattern = "straight_horizontal"
                    else:
                        chosen_pattern = "directed"
                    
                    # Scale intensity based on distance
                    scale_distance_threshold = 1.2
                    min_viz_scale = 0.1
                    if distance_to_target < scale_distance_threshold:
                        ratio = distance_to_target / scale_distance_threshold
                        viz_scale_factor = min_viz_scale + (1.0 - min_viz_scale) * (ratio**2)
                        viz_scale_factor = max(min_viz_scale, viz_scale_factor)
                    else:
                        viz_scale_factor = 1.0
                    
                    current_intensity = 100 * viz_scale_factor
                    
                    # *** Use optimized layout instead of direct coil grid activation ***
                    self.optimized_layout.activate_pattern(
                        pattern_type=chosen_pattern,
                        piece_position=current_pos,
                        target=target_pos,
                        intensity=current_intensity,
                        blocked_coils=blocked_coils_set
                    )
                    
                    # Update the coil grid for visualization and field calculation
                    self.optimized_layout.update_magnetic_field(self.coil_grid)
                    
                    self.current_total_sim_amps = self.coil_grid.calculate_total_current()
                    self.heatmap_needs_update = True
                    self.hardware_controller.apply_state(self.coil_grid.coil_power, self.coil_grid.coil_current)
        
        # Update Captured Piece Movement
        if self.captured_piece and not self.capture_path_finished:
            node_reached_or_finished = self.captured_piece.follow_capture_path(self.capture_step_index)
            if node_reached_or_finished:
                if self.capture_step_index < len(self.captured_piece.capture_path) - 1:
                    self.capture_step_index += 1
                else:
                    self.capture_path_finished = True
                    print(f"Capture movement finished for {self.captured_piece.symbol}.")
    
    # Replace the original method with our patched version
    board_instance.update_move = types.MethodType(patched_update_move, board_instance)
    
    # Override the draw method to show our custom visualization
    original_coil_grid_draw = board_instance.coil_grid.draw
    
    def patched_coil_grid_draw(self, surface, board_pixel_size, x_offset=0, y_offset=0):
        """Custom visualization showing the optimized layout"""
        if hasattr(board_instance, 'optimized_layout'):
            # Use the optimized layout's custom visualization
            board_instance.optimized_layout.visualize(surface, board_pixel_size, x_offset, y_offset)
        else:
            # Fall back to the original visualization
            original_coil_grid_draw(surface, board_pixel_size, x_offset, y_offset)
    
    # Replace the draw method
    board_instance.coil_grid.draw = types.MethodType(patched_coil_grid_draw, board_instance.coil_grid)