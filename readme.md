# Electromagnetic Chess Simulation

This project simulates a smart chessboard that uses a grid of electromagnets to control the movement of chess pieces. It includes a physics simulation for piece movement driven by a PID controller calculating direct forces, and provides visualizations of the board, pieces, coil activations, and magnetic fields.

## Goal

The primary goal is to develop and simulate the control system required to move chess pieces smoothly, accurately, and autonomously across a board using an underlying electromagnet grid. This involves:
1.  Simulating the physics of piece movement under applied forces.
2.  Developing effective coil activation patterns (primarily for visualization and potential hardware commands).
3.  Implementing control strategies (currently **Direct PID Force Control**) to ensure pieces reach their target squares precisely without oscillation or unintended interactions.
4.  Visualizing the system's state for debugging and analysis.
5.  Providing an interface for potential integration with physical hardware.

## Features

*   **Chess Board Simulation:** Displays a standard 8x8 chessboard with coordinate labels.
*   **Piece Representation:** Uses Unicode characters for chess pieces with defined physical properties (diameter, height, magnet strength) influencing simulation mass.
*   **Coil Grid Simulation:** Simulates a 20x20 grid of electromagnets beneath the board.
*   **Coil Activation Patterns:** Implements strategies (`directed`, `radial`, `knight`, `straight_horizontal`, `straight_vertical`) to activate coils, mainly for visualization and potential hardware output. Coil activation *does not* directly drive the simulated physics in the current PID setup.
*   **Physics Engine:** Simulates piece dynamics (position, velocity, acceleration) based on forces calculated by the PID controller. Includes velocity and acceleration limits.
*   **PID Control (Direct Force):** Implements a Proportional-Integral-Derivative (PID) controller that directly calculates the target force required for precise piece movement. This force is then applied to the piece's physics simulation.
*   **Visualization:**
    *   Displays piece positions (with optional exact coordinate dots) and movement paths.
    *   Visualizes active coils and their polarity (attraction/repulsion).
    *   Optionally displays calculated magnetic field vectors (simulated field based on coil patterns).
    *   Optionally displays a heatmap of simulated magnetic field strength beside the board.
    *   Optionally displays markers for the exact center of each square.
*   **Obstacle Avoidance (Coils):** Implemented coil masking (`_create_keep_out_mask`) to prevent activating coils directly under stationary pieces during a move (relevant for visualization/hardware).
*   **Pathfinding (Captures & Clearance):** Uses A* algorithm to calculate paths for captured pieces moving off the board and includes logic to attempt nudging blocking pieces aside for regular moves.
*   **Hardware Abstraction:** Includes an `ElectromagnetController` class to separate simulation logic from hardware communication (can run in simulation or attempt hardware mode).
*   **Interactive Controls:** Allows selecting pieces, setting targets (clicking squares sets the *center* of the square as the target float coordinates), resetting the board, toggling visualizations, adjusting simulation speed, and cycling through patterns via keyboard and mouse.
*   **Resizable Window:** The simulation window can be resized, adjusting the board, heatmap, and control panel layout.

## Project Structure
Use code with caution.
Markdown
.
├── main.py # Main entry point, argument parsing, simulation setup
├── chess_simulation.py # Core simulation logic, ChessBoard class, PID, update loop
├── chess_pieces.py # ChessPiece class definition, properties, constants
├── coil_controller.py # CoilGrid class, coil patterns, field/force sim, visualizations
├── pathfinding.py # PathFinder class (A*) for capture/clearance paths
├── visualization.py # ChessRenderer class for drawing board, pieces, UI
├── hardware_interface.py # ElectromagnetController class (hardware abstraction)
├── field_heatmap.png # Generated heatmap image (temporary)
└── README.md # This file
## Setup / Installation

1.  **Python:** Ensure you have Python 3 installed (developed with 3.11+).
2.  **Dependencies:** Install required libraries using pip:
    ```bash
    pip install pygame numpy matplotlib scipy
    ```
3.  **Hardware Libraries (Optional):** If you intend to run with physical hardware controlled via Serial/GPIO, you might need:
    ```bash
    pip install pyserial
    # On Raspberry Pi or similar for GPIO:
    # pip install RPi.GPIO
    ```
    *Note: These are **not** required to run in simulation mode.*

## Running the Simulation

Execute the main script from your terminal:

```bash
python main.py
Use code with caution.
Command-line Arguments:
--hardware: (Flag) Attempt to run in hardware control mode. Requires hardware libraries and correct setup.
--port <PORT_NAME>: Specify the serial port (e.g., COM3, /dev/ttyACM0). Default: /dev/ttyUSB0.
--baud <RATE>: Specify the serial baud rate. Default: 115200.
--dirpin <GPIO_PIN>: Specify the BCM GPIO pin number for RS485 direction control. Default: None.
--debug: (Flag) Enable extra debug output in the console (overrides the default in ChessBoard.__init__).
Example (Hardware Mode on COM4):
python main.py --hardware --port COM4
Use code with caution.
Bash
Controls
Mouse Click:
Click on a piece to select it.
Click on a target square to initiate a move (target is the center of the square).
Click the selected piece's square again to deselect.
Keyboard:
R: Reset the board to the starting position.
M: Cycle through coil activation patterns (used for visualization/hardware).
C: Toggle visualization of active coils.
F: Toggle visualization of magnetic field vectors.
P: Toggle visualization of piece movement paths.
H: Toggle display of the magnetic field strength heatmap.
D: Toggle detailed debug output in the console (useful for PID tuning).
X: Toggle visualization of square center markers.
Y: Toggle visualization of exact piece position dots and coordinates.
+/- (Plus/Minus keys): Adjust simulation speed.
Esc: Quit the simulation.
Current Status & Challenges
The simulation uses a Direct PID Force Control strategy. The PID controller calculates the necessary force based on the piece's position error (distance to the target square center) and velocity. This force is directly applied in the physics simulation (piece.apply_force). The coil activation patterns (coil_controller.py) are primarily for visualization and potential hardware commands; they do not currently influence the simulated physics forces.
Key Challenge: PID Tuning & Stability
Oscillation: As clearly shown in the debug logs, the current PID gains (pid_kp = 120.0, pid_ki = 5.0, pid_kd = 120.0, terminal_damping = 3.0) lead to severe oscillations. The piece accelerates rapidly, overshoots the target, gets driven back forcefully by the derivative (D) term, overshoots again, and repeats.
Velocity/Force Limits: The piece frequently hits the maximum velocity (max_velocity = 8.0) and the calculated PID force often exceeds reasonable limits, indicating the controller is far too aggressive.
Convergence Failure: Due to the oscillations, the piece fails to settle near the target position. The stop condition (distance_to_target < stop_threshold and np.linalg.norm(current_vel) < velocity_threshold) is unlikely to be met smoothly.
Gain Imbalance: The proportional (Kp) and derivative (Kd) gains appear too high relative to the system's simulated mass and the desired smooth movement. The integral (Ki) gain is too low to effectively counteract steady-state error before oscillations dominate. The high Kd term, intended for damping, seems to be acting as a primary driver of the reversing force causing the oscillations.
Other Areas:
Target Coordinates: Clicking a square sets the target to its exact center (e.g., (3.0, 5.0) for square (3,5)). This is correctly implemented.
Physics Model: The physics model is basic. Factors like friction are not included, which might affect real-world behavior vs. simulation. The mass calculation could be reviewed.
Coil Masking: The _create_keep_out_mask helps prevent coil visualization/commands under stationary pieces, but its effectiveness might need tuning (keep_out_radius_coils).
Future Work / Improvements
PID Tuning: Systematically tune the PID gains (Kp, Ki, Kd, terminal_damping) to achieve stable, non-oscillatory convergence to the target. This is the highest priority. (See tuning suggestions below).
Physics Model: Consider adding friction (static/dynamic) for more realism, though this will further complicate tuning.
Sensor Feedback Simulation: Simulate sensor input (e.g., camera, Hall effect) and feed that position into the PID controller instead of the perfect simulation state.
Hardware Integration: Test and refine the ElectromagnetController with actual hardware.
Advanced Control: Explore feedforward, gain scheduling (different gains for different distances/pieces), or adaptive PID techniques.
Chess Engine Integration: Connect to an engine like Stockfish for automated play.
GUI Enhancements: Improve the control panel display (e.g., PID term visualization).
Dependencies
Python 3.x
Pygame (pip install pygame)
NumPy (pip install numpy)
Matplotlib (pip install matplotlib)
SciPy (pip install scipy)
PySerial (Optional, for hardware) (pip install pyserial)
RPi.GPIO (Optional, for hardware) (pip install RPi.GPIO)
---

## PID Tuning Analysis and Suggestions

The debug output clearly shows classic symptoms of an unstable or poorly tuned PID controller, specifically **excessive oscillation driven by overly high Kp and Kd gains**.

**Observations from Logs:**

1.  **High Initial Force (Kp):** The P-term (`pid_kp * error`) is initially very large, causing rapid acceleration.
2.  **Dominant D-term (Kd):** As soon as velocity builds up, the D-term (`-pid_kd * current_vel`) becomes extremely large and negative (or positive if velocity is negative), acting as a massive braking/reversing force. This is the primary cause of the overshoot and oscillation. The current Kd (120) is far too high for damping; it's actively destabilizing the system.
3.  **Velocity Saturation:** The piece hits `max_velocity` (8.0) almost immediately and repeatedly alternates between +8.0 and -8.0. This means the controller is constantly demanding forces beyond what the physics allows, further contributing to instability.
4.  **Ineffective I-term (Ki):** The I-term accumulates very slowly (`pid_ki = 5.0`) and has negligible effect compared to the huge P and D terms during the oscillations. It cannot correct the final offset because the system never settles.
5.  **Terminal Damping Ineffective:** The piece enters the terminal zone (`distance_to_target < 0.3`) while still moving at high velocity due to the oscillations. The added damping (`terminal_damping * current_vel * terminal_factor`) is insufficient to overcome the momentum and instability generated by the main PID terms.

**Tuning Strategy:**

You need to drastically reduce the aggressiveness of the controller. The goal is smooth damping, not violent reversals.

1.  **SLASH Kp and Kd:** Start by significantly reducing both Kp and Kd.
    *   **Try:** `pid_kp = 15.0` (was 120)
    *   **Try:** `pid_kd = 20.0` (was 120)
    *   **Rationale:** Lower Kp reduces the initial "kick". Lower Kd reduces the violent reaction to velocity, allowing it to act more like damping.

2.  **Reduce or Zero Ki Initially:** While tuning Kp and Kd, set Ki very low or even to zero to isolate the effects of P and D.
    *   **Try:** `pid_ki = 0.5` (was 5.0) or even `pid_ki = 0.0` temporarily.
    *   **Rationale:** Remove the integral effect until the basic P-D response is stable.

3.  **Adjust Terminal Damping / Zone (Later):** Keep `terminal_damping = 3.0` for now, but be prepared to adjust it *after* the main oscillations are gone. You might even increase the `terminal_zone = 0.5` so it activates slightly earlier.

4.  **Observe and Iterate:**
    *   Run the simulation with these drastically reduced gains.
    *   **Expected Behavior:** The movement should be much slower and hopefully more stable (less or no oscillation). It might not reach the target perfectly or might be very sluggish.
    *   **If Stable but Slow:** Gradually increase `pid_kp` (e.g., to 20, 25, 30...) until the response speed is acceptable *without* reintroducing significant overshoot/oscillation.
    *   **If Oscillating (even mildly):** Increase `pid_kd` slightly (e.g., to 25, 30...) to add more damping. *Avoid* large increases in Kd, as that was the original problem. Find the balance between Kp (speed) and Kd (damping).
    *   **If Stable but Offset:** Once you have a stable Kp/Kd combination where the piece gets *close* to the target and settles without oscillation (even if there's a small final offset), *then* start slowly increasing `pid_ki` (e.g., 0.5 -> 1.0 -> 1.5...) This will help eliminate the steady-state error. Add Ki *very* gradually, as too much Ki can cause slow oscillations or overshoot (integral windup). Keep `pid_integral_max` to prevent extreme windup.

5.  **Consider Physics Limits:** If you still hit `max_velocity` frequently even with lower gains, consider if the limit itself is too low or if the simulated mass is too small (making the piece too easy to accelerate). You could try increasing the `mass` calculation factor in `apply_force` temporarily.

**Example Starting Point (incorporating the tries above):**

```python
# In ChessBoard __init__
self.pid_kp = 15.0
self.pid_ki = 0.5  # Start low
self.pid_kd = 20.0
self.terminal_damping = 3.0 # Keep for now
# Maybe increase zone slightly:
# terminal_zone = 0.5 # in update_move