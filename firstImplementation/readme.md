# Smart Chessboard with Electromagnetic Control

A simulation and control system for a smart chessboard that uses a 20×20 grid of electromagnets to physically move chess pieces. The system includes advanced pathfinding for piece movement, collision avoidance, and support for both simulated and real hardware control.

## Features

- **Smart Piece Movement**: Magnetically move chess pieces using a 20×20 electromagnet grid (400 coils)
- **Advanced Pathfinding**: A* algorithm for optimal movement paths and obstacle avoidance
- **Capture Visualization**: Captured pieces follow intelligent paths off the board
- **Piece Recognition**: Pieces labeled with type and color for easy identification
- **Real-time Visualization**: See electromagnet activation patterns in real-time
- **Dual-mode Operation**: Run in simulation mode or control actual hardware
- **RS-485 Communication**: Star-connected 3-wire protocol for efficient coil control
- **Standard Dimensions**: Based on tournament chess standards (55mm squares)

## Installation

### Prerequisites

- Python 3.7+
- Pygame library
- PySerial (for hardware control)
- RPi.GPIO (for Raspberry Pi hardware control)

### Setup

1. Clone the repository
```bash
git clone https://github.com/username/smart-chessboard.git
cd smart-chessboard
```

2. Install the required packages
```bash
pip install pygame pyserial RPi.GPIO
```

3. Run the simulation
```bash
python simulation.py
```

## Usage

### Simulation Controls

- **Drag and Drop**: Move chess pieces with your mouse
- **'r' key**: Reset the board to initial position
- **'c' key**: Toggle electromagnet coil visualization
- **'p' key**: Toggle path planning visualization 
- **'h' key**: Toggle between simulation and hardware control modes

### Hardware Mode

To connect to real hardware:

1. Connect your RS-485 interface to the computer
2. Update the `com_port` parameter in `hardware_controller.py` to match your setup
3. Press the 'h' key in the simulation to activate hardware control mode

## Hardware Requirements

### Chess Pieces and Board

- **Board Size**: 440mm × 440mm (8×8 grid of 55mm squares)
- **Chess Pieces**: Standard Staunton design with ferromagnetic material in base
- **Piece Base Diameters**:
  - King: 40mm
  - Queen: 38mm
  - Bishop/Knight: 35mm
  - Rook: 33mm
  - Pawn: 29mm

### Electromagnet System

- **Grid Size**: 20×20 coils (400 total)
- **Coil Spacing**: 22mm between centers
- **Coil Diameter**: 20mm
- **Communication**: RS-485 protocol with 3-wire star connection
- **Control System**: Microcontroller with RS-485 transceiver

## Software Architecture

The project consists of two main components:

1. **Simulation Module** (`simulation.py`):
   - Game board visualization
   - Chess rules and movement logic
   - Piece capture handling
   - A* pathfinding for collision avoidance
   - Electromagnet visualization
   - User interface

2. **Hardware Controller** (`hardware_controller.py`):
   - RS-485 protocol implementation
   - Coil power management
   - Path calculation and following
   - Graceful degradation to simulation mode
   - Alternative CAN bus implementation

## How the Electromagnet System Works

The smart chessboard uses a grid of 400 electromagnets underneath the board surface to move chess pieces:

1. Each piece contains ferromagnetic material in its base
2. The system activates specific electromagnets to create a magnetic "pull"
3. A radial pattern of decreasing magnetic strength creates a gradient
4. The gradient pulls pieces toward the strongest point
5. By sequentially activating coils along a path, pieces can be moved smoothly
6. The A* algorithm calculates optimal paths that avoid collisions with other pieces

## Implementation Details

### Pathfinding Algorithm

The A* pathfinding algorithm considers:

- Physical size of each chess piece (different for pawns, kings, etc.)
- Current positions of all pieces on the board
- Shortest path to destination
- Special handling for captured pieces moving off the board

### Coil Control

Each electromagnet coil can be individually controlled with variable power levels:

- Power levels range from 0-100%
- Radial activation pattern for smooth movement
- Multiple pieces can be moved simultaneously (capturing piece and captured piece)
- Gradual power transitions prevent jerky movements

## Future Improvements

- Add chess rules enforcement
- Implement automatic gameplay against a chess engine
- Add piece recognition using sensors or computer vision
- Optimize power consumption by improving path efficiency
- Add wireless capability (Bluetooth/WiFi)
- Create a mobile app interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Chess piece dimension standards from FIDE
- A* pathfinding algorithm based on standard implementations
- RS-485 protocol design based on industrial automation standards