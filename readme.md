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
*   **Selection Cancellation:** Provides easy ways to cancel current piece selection using right-click or the Escape key.
*   **Resizable Window:** The simulation window can be resized, adjusting the board, heatmap, and control panel layout with stable behavior across multiple monitors.
*   **Centered Window:** The application window appears centered on the screen.

## Recent Improvements and Bugfixes

### 1. Heatmap Visualization Fix
- **Problem:** The heatmap visualization displayed blank or had NaN values due to division by zero in magnetic field calculations.
- **Solution:** Implemented robust error handling in `update_magnetic_field()` method to prevent NaN propagation by:
  - Using a larger epsilon value (1e-5) to prevent both zero and very small divisors
  - Using `np.maximum()` to ensure distances never become too small
  - Using `np.divide()` with `out` and `where` parameters for safe division
  - Adding final NaN cleanup with `np.nan_to_num()`
- **Enhanced Heatmap Generation:**
  - Improved early detection of zero/small fields
  - Added sample debugging output to diagnose issues
  - Improved interpolation with explicit NaN handling
  - Added robust error handling and backtracking throughout the process

### 2. Optimized PID Control Parameters
- **Problem:** The PID control system had oscillation issues and didn't smoothly move pieces to target positions.
- **Solution:** Fine-tuned the PID parameters to optimal values:
  - `pid_kp = 80.0` (Proportional term) - Controls how strongly the system reacts to the current error
  - `pid_ki = 0.2` (Integral term) - Addresses accumulated error over time
  - `pid_kd = 44.0` (Derivative term) - Provides damping based on rate of change
  - These values provide smooth, stable movements without oscillation

### 3. Selection Cancellation Feature
- **Problem:** There was no easy way to cancel piece selection once started.
- **Solution:** Implemented two intuitive ways to cancel selection:
  - Right-click anywhere to cancel current selection
  - Press the Escape key to cancel selection (or exit if no selection active)
  - Properly handles cleanup of temporarily moved pieces during path clearance attempts

### 4. Window Resizing Improvements
- **Problem:** Window resizing caused issues, especially when moving between monitors with different resolutions or DPI settings.
- **Solution:** Implemented an enhanced window resize handler:
  - Added debounce mechanism to prevent handling multiple resize events in quick succession
  - Detects "major" resizes (>20% change) likely from moving between monitors
  - Implements a "settling period" after major resizes to ignore subsequent minor adjustments
  - Uses reasonable size limits to prevent renderer creation failures
  - Added comprehensive error handling with fallback resize options

### 5. Window Centering
- **Problem:** The application window appeared at the upper-left corner of the screen instead of centered.
- **Solution:** Implemented proper window centering using the SDL environment variable:
  - Added `SDL_VIDEO_CENTERED = '1'` environment setting before pygame initialization
  - Ensures the window appears centered on startup
  - Maintains position during resize operations
  - Works reliably across different platforms and monitor setups

## Project Structure
.
├── main.py                  # Main entry point, argument parsing, simulation setup
├── chess_simulation.py      # Core simulation logic, ChessBoard class, PID, update loop
├── chess_pieces.py          # ChessPiece class definition, properties, constants
├── coil_controller.py       # CoilGrid class, coil patterns, field/force sim, visualizations
├── pathfinding.py           # PathFinder class (A*) for capture/clearance paths
├── visualization.py         # ChessRenderer class for drawing board, pieces, UI
├── hardware_interface.py    # ElectromagnetController class (hardware abstraction)
├── field_heatmap.png        # Generated heatmap image (temporary)
└── README.md                # This file

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
```

### Command-line Arguments:
- `--hardware`: (Flag) Attempt to run in hardware control mode. Requires hardware libraries and correct setup.
- `--port <PORT_NAME>`: Specify the serial port (e.g., COM3, /dev/ttyACM0). Default: /dev/ttyUSB0.
- `--baud <RATE>`: Specify the serial baud rate. Default: 115200.
- `--dirpin <GPIO_PIN>`: Specify the BCM GPIO pin number for RS485 direction control. Default: None.
- `--debug`: (Flag) Enable extra debug output in the console (overrides the default in ChessBoard.__init__).

Example (Hardware Mode on COM4):
```bash
python main.py --hardware --port COM4
```

## Controls

### Mouse Controls:
- **Left Click on Piece**: Select a piece.
- **Left Click on Square**: Set target destination for selected piece.
- **Left Click on Currently Selected Piece**: Deselect the piece.
- **Right Click**: Cancel current selection.

### Keyboard Controls:
- **R**: Reset the board to the starting position and apply current PID slider values.
- **M**: Cycle through coil activation patterns (for visualization/hardware).
- **+/-**: Adjust simulation speed.
- **Esc**: Cancel selection if a piece is selected, or quit the simulation if no selection.
- **C**: Toggle visualization of active coils.
- **F**: Toggle visualization of magnetic field vectors.
- **P**: Toggle visualization of piece movement paths.
- **H**: Toggle display of the magnetic field strength heatmap.
- **D**: Toggle detailed debug output in the console (useful for PID tuning).
- **X**: Toggle visualization of square center markers.
- **Y**: Toggle visualization of exact piece position dots and coordinates.

## Dependencies
- Python 3.x
- Pygame (pip install pygame)
- NumPy (pip install numpy)
- Matplotlib (pip install matplotlib)
- SciPy (pip install scipy)
- PySerial (Optional, for hardware) (pip install pyserial)
- RPi.GPIO (Optional, for hardware) (pip install RPi.GPIO)