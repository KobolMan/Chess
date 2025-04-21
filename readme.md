# Electromagnetic Chess Simulation

This project simulates a smart chessboard that uses a grid of electromagnets to control the movement of chess pieces. It includes a physics simulation for piece movement driven by calculated magnetic forces (or a PID controller) and provides visualizations of the board, pieces, coil activations, and magnetic fields.

## Goal

The primary goal is to develop and simulate the control system required to move chess pieces smoothly, accurately, and autonomously across a board using an underlying electromagnet grid. This involves:
1.  Simulating the physics of piece movement under magnetic forces.
2.  Developing effective coil activation patterns to generate desired forces.
3.  Implementing control strategies (like PID) to ensure pieces reach their target squares precisely without oscillation or unintended interactions.
4.  Visualizing the system's state for debugging and analysis.
5.  Providing an interface for potential integration with physical hardware.

## Features

*   **Chess Board Simulation:** Displays a standard 8x8 chessboard.
*   **Piece Representation:** Uses Unicode characters to represent chess pieces with defined physical properties (diameter, height, magnet strength).
*   **Coil Grid Simulation:** Simulates a 20x20 grid of electromagnets beneath the board.
*   **Coil Activation Patterns:** Implements various strategies (`directed`, `radial`, `knight`, `straight_horizontal`, `straight_vertical`) to activate coils for moving pieces.
*   **Physics Engine:** Basic simulation of piece dynamics (position, velocity, acceleration) based on applied forces.
*   **PID Control:** Implements a Proportional-Integral-Derivative (PID) controller to directly calculate the force required for precise piece movement, aiming to eliminate errors and oscillations.
*   **Visualization:**
    *   Displays piece positions and movement paths.
    *   Visualizes active coils and their polarity (attraction/repulsion).
    *   Can optionally display calculated magnetic field vectors.
    *   Can optionally display a heatmap of magnetic field strength.
*   **Obstacle Avoidance (Coils):** Implemented coil masking to prevent activating coils directly under stationary pieces during a move.
*   **Pathfinding (Captures):** Uses A* algorithm to calculate paths for captured pieces to move off the board.
*   **Hardware Abstraction:** Includes a basic `ElectromagnetController` class to separate simulation logic from hardware communication (currently runs in simulation mode).
*   **Interactive Controls:** Allows selecting pieces, setting targets, resetting the board, toggling visualizations, adjusting simulation speed, and cycling through patterns via keyboard and mouse.

## Project Structure
Use code with caution.
Markdown
.
├── main.py # Main entry point, argument parsing, simulation setup
├── chess_simulation.py # Core simulation logic, ChessBoard class, update loop
├── chess_pieces.py # ChessPiece class definition, properties, constants
├── coil_controller.py # CoilGrid class, coil patterns, field/force simulation, visualizations
├── pathfinding.py # PathFinder class (A*) for capture paths
├── visualization.py # ChessRenderer class for drawing board, pieces, UI
├── hardware_interface.py # ElectromagnetController class (hardware communication abstraction)
├── field_heatmap.png # Generated heatmap image (temporary)
└── README.md # This file
## Setup / Installation

1.  **Python:** Ensure you have Python 3 installed (developed with 3.11).
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
--hardware: (Flag) Attempt to run in hardware control mode instead of simulation. Requires hardware libraries and correct setup.
--port <PORT_NAME>: Specify the serial port for hardware communication (e.g., COM3 on Windows, /dev/ttyACM0 or /dev/ttyUSB0 on Linux). Default: /dev/ttyUSB0.
--baud <RATE>: Specify the serial baud rate. Default: 115200.
--dirpin <GPIO_PIN>: Specify the BCM GPIO pin number for RS485 direction control (if needed). Default: None.
--debug: (Flag) Enable extra debug output in the console (overrides the default in ChessBoard.__init__).
Example (Hardware Mode on COM4):
python main.py --hardware --port COM4
Use code with caution.
Bash
Controls
Mouse Click:
Click on a piece to select it.
Click on a target square to initiate a move for the selected piece.
Click the selected piece's square again to deselect.
Keyboard:
R: Reset the board to the starting position.
M: Cycle through the default coil activation patterns (directed, knight, radial) used for visualization/hardware.
C: Toggle visualization of active coils.
F: Toggle visualization of magnetic field vectors.
P: Toggle visualization of piece movement paths.
H: Toggle display of the magnetic field strength heatmap.
D: Toggle detailed debug output in the console (useful for PID tuning).
+/- (Plus/Minus keys): Adjust simulation speed.
Esc: Quit the simulation.
Current Status & Challenges
The simulation currently uses a Direct PID Force Control strategy. The PID controller calculates the necessary force based on the piece's position error and velocity, and this force is directly applied in the physics simulation. The coil activation patterns are used for visualization and potentially sending commands to hardware but do not directly drive the simulated physics.
Key Challenges & Areas for Improvement:
PID Tuning: This is the primary current challenge. The PID gains (Kp, Ki, Kd in chess_simulation.py) need careful tuning to achieve the desired performance:
Convergence: Pieces should stop precisely at the center of the target square with minimal offset before the final snap.
Overshoot: The piece should not significantly overshoot the target.
Oscillation: The piece should settle quickly without oscillating around the target.
Responsiveness: The movement should be reasonably fast.
Current Status: The direct PID force method has eliminated gross oscillations seen previously, and convergence is close, but fine-tuning is required to eliminate the small offset before the final snap and ensure smooth settling across different move types. The Ki term (integral) needs careful introduction to handle the final offset without causing instability.
Diagonal Drift: While significantly reduced by the direct PID force, slight diagonal drift during nominally straight moves can still occur if there are minor asymmetries or noise. Fine-tuning the PID might mitigate this further. The dedicated straight_horizontal and straight_vertical coil patterns help ensure the visualized/hardware command is axially aligned.
Coil Masking (_create_keep_out_mask): The current implementation prevents activating coils directly under stationary pieces. The keep_out_radius_coils parameter might need tuning to balance avoiding interference with providing enough controllable field near other pieces.
Physics Model: The current physics model is basic (mass proportional to volume, simple velocity/acceleration updates). Adding static/dynamic friction could make the simulation more realistic but also complicate PID tuning.
Coil Patterns: While PID now drives the force, the underlying patterns (directed, knight, etc.) are still used for visuals and hardware commands. Their effectiveness in generating useful fields could still be improved, especially the knight pattern.
Future Work / Improvements
Systematic PID tuning to find optimal gains for different piece types or move distances.
Implement more sophisticated physics (friction, collisions).
Refine coil activation patterns for efficiency and reduced interference.
Simulate sensor feedback (e.g., Hall effect sensors, camera) to close the loop for the PID controller based on measured position instead of simulated position.
Full hardware integration and testing using the ElectromagnetController interface.
GUI improvements (e.g., better display of PID state, captured pieces).
Integration with a chess engine (like Stockfish) for automated play.
Advanced control techniques (e.g., feedforward control, adaptive PID).
Dependencies
Python 3.x
Pygame (pip install pygame)
NumPy (pip install numpy)
Matplotlib (pip install matplotlib)
SciPy (pip install scipy)
PySerial (Optional, for hardware) (pip install pyserial)
RPi.GPIO (Optional, for hardware) (pip install RPi.GPIO)