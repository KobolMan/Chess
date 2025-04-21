# coil_controller.py

import numpy as np
import math
import pygame # For drawing simulation state
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# Constants...
DARK_GRAY = (100, 100, 100); RED = (255, 0, 0); BLUE = (0, 0, 255)

class CoilGrid:
    def __init__(self, size=20, board_squares=8):
        self.size = size; self.board_squares = board_squares
        self.coil_power = np.zeros((size, size)); self.coil_current = np.zeros((size, size))
        self.magnetic_field = np.zeros((size, size, 2))

    def reset(self): self.coil_power.fill(0); self.coil_current.fill(0); self.magnetic_field.fill(0)

    def update_coil(self, row, col, power, current_direction=1):
        if 0 <= row < self.size and 0 <= col < self.size:
            safe_power = np.clip(power, 0, 100); self.coil_power[row, col] = safe_power
            self.coil_current[row, col] = np.sign(current_direction) if safe_power > 0 else 0

    # --- **** ADDED blocked_coils PARAMETER **** ---
    def activate_coil_pattern(self, pattern_type, position, target=None, intensity=100, radius=4, blocked_coils=None):
        """
        Activate a simulated pattern of coils, avoiding blocked coils.
        """
        if blocked_coils is None: blocked_coils = set()

        center_x, center_y = position
        center_x_int=int(round(center_x)); center_y_int=int(round(center_y))
        min_r_loop=max(0,center_y_int-radius*2); max_r_loop=min(self.size,center_y_int+radius*2+1)
        min_c_loop=max(0,center_x_int-radius*2); max_c_loop=min(self.size,center_x_int+radius*2+1)
        self.coil_power.fill(0); self.coil_current.fill(0)

        def safe_update_coil(r, c, power, current): # Helper
            if (r, c) not in blocked_coils: self.update_coil(r, c, power, current)

        if pattern_type == "radial":
             for r in range(min_r_loop, max_r_loop):
                 for c in range(min_c_loop, max_c_loop):
                     distance = np.sqrt((r - center_y)**2 + (c - center_x)**2)
                     if distance <= radius: power = intensity * max(0, (1 - distance / radius))**2; safe_update_coil(r, c, power, -1)
        elif pattern_type == "straight_horizontal":
             if target is None: return
             target_x, _ = target; direction_x = np.sign(target_x - center_x)
             current_row = center_y_int
             if 0 <= current_row < self.size:
                 for c in range(min_c_loop, max_c_loop):
                      proj = c - center_x; distance_from_center = abs(proj)
                      if distance_from_center <= radius * 1.5:
                           power_factor = max(0, 1 - distance_from_center / (radius * 1.5))**2; power = intensity * power_factor
                           if proj * direction_x > 0.1: current = -1
                           elif proj * direction_x < -0.1: current = 1
                           else: current = 0; power = 0
                           safe_update_coil(current_row, c, power, current)
        elif pattern_type == "straight_vertical":
             if target is None: return
             _, target_y = target; direction_y = np.sign(target_y - center_y)
             current_col = center_x_int
             if 0 <= current_col < self.size:
                 for r in range(min_r_loop, max_r_loop):
                      proj = r - center_y; distance_from_center = abs(proj)
                      if distance_from_center <= radius * 1.5:
                           power_factor = max(0, 1 - distance_from_center / (radius * 1.5))**2; power = intensity * power_factor
                           if proj * direction_y > 0.1: current = -1
                           elif proj * direction_y < -0.1: current = 1
                           else: current = 0; power = 0
                           safe_update_coil(r, current_col, power, current)
        elif pattern_type == "directed" and target is not None:
            target_x, target_y = target; direction = np.array([target_x-center_x, target_y-center_y]); distance_to_target = np.linalg.norm(direction)
            if distance_to_target < 0.1: self.activate_coil_pattern("radial",position,target,intensity*0.5,radius,blocked_coils); return
            direction_norm = direction/distance_to_target
            for r in range(min_r_loop, max_r_loop):
                for c in range(min_c_loop, max_c_loop):
                    rel_pos=np.array([c-center_x,r-center_y]); distance=np.linalg.norm(rel_pos)
                    if distance <= radius * 1.5:
                        proj=np.dot(rel_pos,direction_norm); perp_dist_sq=max(0,distance**2-proj**2); perp_distance=np.sqrt(perp_dist_sq)
                        power_factor=max(0,1-distance/(radius*1.5))**2; direction_focus=max(0,1-perp_distance/(radius*0.8))**2
                        power=intensity*power_factor*direction_focus; proj_threshold=0.2*radius
                        if proj>proj_threshold: current=-1
                        elif proj<-proj_threshold: current=1
                        else: current=-1; power*=0.2
                        power=max(0,power); safe_update_coil(r,c,power,current)
        elif pattern_type == "knight" and target is not None:
            target_x, target_y = target; dx=target_x-center_x; dy=target_y-center_y
            if abs(dx)>abs(dy): point1_x=center_x+np.sign(dx)*(abs(dx)*2/3); point1_y=center_y; point2_x=point1_x; point2_y=center_y+np.sign(dy)*abs(dy)
            else: point1_x=center_x; point1_y=center_y+np.sign(dy)*(abs(dy)*2/3); point2_x=center_x+np.sign(dx)*abs(dx); point2_y=point1_y
            path_points=[(center_x,center_y),(point1_x,point1_y),(point2_x,point2_y),(target_x,target_y)]
            for r in range(min_r_loop, max_r_loop):
                for c in range(min_c_loop, max_c_loop):
                    min_dist_sq=float('inf'); closest_segment_proj=0; coil_pos=np.array([c,r])
                    for i in range(len(path_points)-1):
                        p1=np.array(path_points[i]); p2=np.array(path_points[i+1]); segment_vec=p2-p1; segment_len_sq=np.dot(segment_vec,segment_vec)
                        if segment_len_sq<1e-6: dist_sq=np.sum((coil_pos-p1)**2); proj=0
                        else: t=np.clip(np.dot(coil_pos-p1,segment_vec)/segment_len_sq,0,1); closest_point=p1+t*segment_vec; dist_sq=np.sum((coil_pos-closest_point)**2); proj=np.dot(coil_pos-np.array([center_x,center_y]),segment_vec/np.sqrt(segment_len_sq)) if segment_len_sq>0 else 0
                        if dist_sq<min_dist_sq: min_dist_sq=dist_sq; closest_segment_proj=proj
                    min_dist=np.sqrt(min_dist_sq)
                    if min_dist<=radius*1.2:
                        power_factor=max(0,1-min_dist/(radius*1.2))**2; power=intensity*power_factor
                        if closest_segment_proj>0.1: current=-1
                        elif closest_segment_proj<-0.1: current=1
                        else: current=-1; power*=0.3
                        safe_update_coil(r,c,power,current) # Use safe update
        else: # Default
            self.activate_coil_pattern("radial", position, target, intensity * 0.5, radius, blocked_coils) # Pass mask

    def update_magnetic_field(self): # No changes
        self.magnetic_field.fill(0); influence_radius_sq = 8**2
        active_coils = np.argwhere(self.coil_power > 0)
        for r_field in range(self.size):
            for c_field in range(self.size):
                field_vec=np.array([0.0,0.0]); pos_field=np.array([c_field,r_field])
                for r_coil, c_coil in active_coils:
                    pos_coil=np.array([c_coil,r_coil]); vec_coil_to_field=pos_field-pos_coil; dist_sq=np.sum(vec_coil_to_field**2)
                    if 0<dist_sq<=influence_radius_sq:
                        dist=np.sqrt(dist_sq); direction_vec=vec_coil_to_field/dist
                        power=self.coil_power[r_coil,c_coil]; current=self.coil_current[r_coil,c_coil]; strength=power*(1.0/(1.0+dist_sq))
                        field_contribution=direction_vec*strength*current; field_vec+=field_contribution
                self.magnetic_field[r_field,c_field]=field_vec

    # --- **** calculate_force NO LONGER USES force_scale **** ---
    def calculate_force(self, piece_position, piece_magnet_strength): # Removed force_scale
        """Calculate base magnetic force (before PID scaling/damping)"""
        col_board, row_board = piece_position; col_grid=col_board*(self.size/self.board_squares); row_grid=row_board*(self.size/self.board_squares)
        col_idx=int(col_grid); row_idx=int(row_grid); dx=col_grid-col_idx; dy=row_grid-row_idx
        if not (0<=col_idx<self.size-1 and 0<=row_idx<self.size-1):
            col_idx=np.clip(col_idx,0,self.size-1); row_idx=np.clip(row_idx,0,self.size-1)
            # Return base field scaled only by magnet strength
            return self.magnetic_field[row_idx,col_idx] * piece_magnet_strength
        field_00=self.magnetic_field[row_idx,col_idx]; field_01=self.magnetic_field[row_idx,col_idx+1]; field_10=self.magnetic_field[row_idx+1,col_idx]; field_11=self.magnetic_field[row_idx+1,col_idx+1]
        interp_x_top=(1-dx)*field_00[0]+dx*field_01[0]; interp_x_bottom=(1-dx)*field_10[0]+dx*field_11[0]; field_x=(1-dy)*interp_x_top+dy*interp_x_bottom
        interp_y_top=(1-dx)*field_00[1]+dx*field_01[1]; interp_y_bottom=(1-dx)*field_10[1]+dx*field_11[1]; field_y=(1-dy)*interp_y_top+dy*interp_y_bottom
        interpolated_field=np.array([field_x,field_y])
        # Force is just field * magnet_strength. PID gains handle magnitude.
        force = interpolated_field * piece_magnet_strength
        return force
    # --- **** END CHANGE **** ---

    def draw(self, surface, board_pixel_size): # No changes
        coil_pixel_size=board_pixel_size/self.size; coil_surface=pygame.Surface((board_pixel_size,board_pixel_size),pygame.SRCALPHA)
        for r in range(self.size):
            for c in range(self.size):
                x=c*coil_pixel_size+coil_pixel_size/2; y=r*coil_pixel_size+coil_pixel_size/2; power=self.coil_power[r,c]; current=self.coil_current[r,c]
                outline_color=(*DARK_GRAY[:3],100); pygame.draw.circle(coil_surface,outline_color,(int(x),int(y)),int(coil_pixel_size/2*0.8),1)
                if power>0: alpha=int(np.clip(power*2.0,50,200)); color=(255,100,100,alpha) if current>0 else (100,100,255,alpha); radius=int(coil_pixel_size/2*0.7*(0.6+0.4*power/100)); pygame.draw.circle(coil_surface,color,(int(x),int(y)),radius)
        surface.blit(coil_surface,(0,0))

    # --- **** draw_field_overlay CORRECTED **** ---
    def draw_field_overlay(self, surface, board_pixel_size, resolution=20):
        """Draw the SIMULATED magnetic field vectors"""
        step_size=board_pixel_size/resolution
        field_surface=pygame.Surface((board_pixel_size,board_pixel_size),pygame.SRCALPHA)
        field_magnitudes=np.linalg.norm(self.magnetic_field,axis=2)
        max_field_strength_observed=field_magnitudes.max() if field_magnitudes.max()>0 else 1.0

        for r_idx in range(resolution):
            for c_idx in range(resolution):
                x_pix=c_idx*step_size+step_size/2; y_pix=r_idx*step_size+step_size/2
                col_grid=c_idx*(self.size/resolution)+(self.size/resolution/2); row_grid=r_idx*(self.size/resolution)+(self.size/resolution/2)
                col_int=int(col_grid); row_int=int(row_grid); dx=col_grid-col_int; dy=row_grid-row_int

                if not (0<=col_int<self.size-1 and 0<=row_int<self.size-1): continue

                field_00=self.magnetic_field[row_int,col_int]; field_01=self.magnetic_field[row_int,col_int+1]
                field_10=self.magnetic_field[row_int+1,col_int]; field_11=self.magnetic_field[row_int+1,col_int+1]
                interp_x_top=(1-dx)*field_00[0]+dx*field_01[0]; interp_x_bottom=(1-dx)*field_10[0]+dx*field_11[0]; field_x=(1-dy)*interp_x_top+dy*interp_x_bottom
                interp_y_top=(1-dx)*field_00[1]+dx*field_01[1]; interp_y_bottom=(1-dx)*field_10[1]+dx*field_11[1]; field_y=(1-dy)*interp_y_top+dy*interp_y_bottom
                field_vec=np.array([field_x,field_y]); field_strength=np.linalg.norm(field_vec)
                min_draw_strength=0.01*max_field_strength_observed

                if field_strength>min_draw_strength:
                    field_normalized=field_vec/field_strength
                    log_strength=np.log1p(field_strength/max_field_strength_observed*10)
                    max_arrow_len=step_size*0.7
                    arrow_len=min(log_strength*max_arrow_len/np.log1p(10),max_arrow_len); arrow_len=max(2,arrow_len)
                    end_x=x_pix+field_normalized[0]*arrow_len; end_y=y_pix+field_normalized[1]*arrow_len
                    color_intensity=min(1.0,field_strength/max_field_strength_observed); arrow_color_rgb=(255,255*(1-color_intensity**0.5),255*(1-color_intensity**0.5))
                    alpha=int(np.clip(100+155*color_intensity,100,255)); arrow_color=(*arrow_color_rgb,alpha)

                    pygame.draw.line(field_surface,arrow_color,(int(x_pix),int(y_pix)),(int(end_x),int(end_y)),1)

                    head_length=min(5,arrow_len*0.4)
                    if head_length>2: # Check if head should be drawn
                        angle=math.atan2(field_normalized[1],field_normalized[0])
                        # Calculate points for arrowhead triangle
                        p1=(end_x,end_y)
                        p2=(end_x-head_length*math.cos(angle-math.pi/6), end_y-head_length*math.sin(angle-math.pi/6))
                        p3=(end_x-head_length*math.cos(angle+math.pi/6), end_y-head_length*math.sin(angle+math.pi/6))
                        # **** Moved draw_polygon INSIDE the if block ****
                        try:
                             pygame.draw.polygon(field_surface, arrow_color, [(int(p1[0]),int(p1[1])), (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1]))])
                        except ValueError: pass # Ignore potential errors
                    # **** END MOVE ****

        surface.blit(field_surface,(0,0))
    # --- **** END CORRECTION **** ---

    def generate_heatmap(self, resolution=100): # No changes needed
        heatmap=np.zeros((resolution,resolution)); field_magnitudes=np.linalg.norm(self.magnetic_field,axis=2)
        max_mag=field_magnitudes.max() if field_magnitudes.max()>1e-6 else 1.0
        for r in range(resolution):
            for c in range(resolution):
                grid_c=c*(self.size/resolution); grid_r=r*(self.size/resolution); col_idx,row_idx=int(grid_c),int(grid_r)
                if 0<=col_idx<self.size-1 and 0<=row_idx<self.size-1: dx,dy=grid_c-col_idx,grid_r-row_idx; f00=field_magnitudes[row_idx,col_idx]; f01=field_magnitudes[row_idx,col_idx+1]; f10=field_magnitudes[row_idx+1,col_idx]; f11=field_magnitudes[row_idx+1,col_idx+1]; interp_top=(1-dx)*f00+dx*f01; interp_bottom=(1-dx)*f10+dx*f11; heatmap[r,c]=(1-dy)*interp_top+dy*interp_bottom
                elif 0<=col_idx<self.size and 0<=row_idx<self.size: heatmap[r,c]=field_magnitudes[row_idx,col_idx]
        heatmap=gaussian_filter(heatmap,sigma=max(1.0,resolution/100.0)); heatmap/=max_mag; return heatmap

    def plot_heatmap(self, filename="field_heatmap.png", figsize=(6, 6)): # No changes needed
        try: heatmap_data=self.generate_heatmap(); plt.figure(figsize=figsize); cmap=plt.cm.viridis; plt.imshow(heatmap_data,cmap=cmap,aspect='equal',origin='upper',interpolation='bilinear',vmin=0,vmax=1); plt.colorbar(label='Normalized Field Strength'); plt.title('Magnetic Field Strength'); plt.xticks([]); plt.yticks([]); plt.tight_layout(); plt.savefig(filename,dpi=150); plt.close(); return filename
        except Exception as e: print(f"Error generating heatmap plot: {e}"); return None