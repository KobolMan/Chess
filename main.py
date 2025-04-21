#!/usr/bin/env python3
# Smart Electromagnetic Chess - Main Script

print("DEBUG: Starting main.py script...") # <<< DEBUG PRINT 1

import sys
import argparse
import pygame # Keep pygame import if needed for quit/exceptions

print("DEBUG: Basic imports done.") # <<< DEBUG PRINT 2

# Import the necessary classes from your modules
# Errors during these imports might cause silent exit
try:
    from chess_simulation import ChessBoard, COIL_GRID_SIZE # Import the main simulation class
    print("DEBUG: Imported ChessBoard.") # <<< DEBUG PRINT 3
    from hardware_interface import ElectromagnetController
    print("DEBUG: Imported ElectromagnetController.") # <<< DEBUG PRINT 4 (We know this one works)
except ImportError as e:
    print(f"FATAL: Failed to import required module: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL: Unexpected error during import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def main():
    """Main entry point for the smart chess system"""
    print("DEBUG: Entered main() function.") # <<< DEBUG PRINT 5
    parser = argparse.ArgumentParser(description='Smart Electromagnetic Chess System')
    parser.add_argument('--hardware', action='store_true', help='Enable hardware control')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0', help='Serial port for hardware control')
    parser.add_argument('--baud', type=int, default=115200, help='Serial baud rate')
    parser.add_argument('--rtscts', action='store_true', help='Enable RTS/CTS hardware flow control (if supported)')
    parser.add_argument('--dirpin', type=int, default=None, help='BCM GPIO pin for RS485 direction')
    parser.add_argument('--debug', action='store_true', help='Enable extra debug output')

    print("DEBUG: Parsing arguments...") # <<< DEBUG PRINT 6
    args = parser.parse_args()
    print(f"DEBUG: Arguments parsed: {args}") # <<< DEBUG PRINT 7

    # Determine initial simulation mode based on args and serial availability
    # (hardware_interface prints its own warning if serial is missing)
    sim_mode = not args.hardware or not ElectromagnetController.SERIAL_AVAILABLE

    # --- Initialize Hardware Controller ---
    print("DEBUG: Initializing hardware controller logic...") # <<< DEBUG PRINT 8
    hardware_controller = None
    if not sim_mode:
        print(f"DEBUG: Attempting hardware mode on {args.port}...")
        try:
            hardware_controller = ElectromagnetController(
                grid_size=COIL_GRID_SIZE,
                simulation_mode=False,
                com_port=args.port,
                baud_rate=args.baud,
                rs485_dir_pin=args.dirpin
            )
            sim_mode = hardware_controller.simulation_mode # Update sim_mode if hardware init failed
        except Exception as e:
            print(f"FATAL: Unhandled error during hardware controller init: {e}")
            print("Exiting.")
            sys.exit(1)

    # Ensure controller exists even in sim mode
    if hardware_controller is None:
         print("DEBUG: Creating ElectromagnetController in simulation mode...")
         hardware_controller = ElectromagnetController(grid_size=COIL_GRID_SIZE, simulation_mode=True)
         sim_mode = True # Ensure sim_mode is True

    print(f"DEBUG: Hardware controller ready (sim_mode={sim_mode}).") # <<< DEBUG PRINT 9

    # --- Initialize Pygame and Chess Simulation ---
    try:
        print("DEBUG: Initializing Pygame...") # <<< DEBUG PRINT 10
        pygame.init()
        pygame.display.set_caption("Electromagnetic Chess")
        print("DEBUG: Pygame initialized.") # <<< DEBUG PRINT 11
    except Exception as e:
        print(f"FATAL: Error initializing Pygame: {e}")
        sys.exit(1)

    try:
        print("DEBUG: Initializing ChessBoard...") # <<< DEBUG PRINT 12
        # Determine effective debug mode for ChessBoard
        effective_debug_mode = args.debug or False # Use parsed arg or default False
        board = ChessBoard(hardware_controller=hardware_controller, debug_mode=effective_debug_mode)
        print("DEBUG: ChessBoard initialized.") # <<< DEBUG PRINT 13
    except Exception as e:
        print(f"FATAL: Error initializing ChessBoard: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for ChessBoard init errors
        pygame.quit()
        sys.exit(1)


    # --- Print Instructions ---
    print("\n===== SMART ELECTROMAGNETIC CHESS =====")
    print("- Click piece to select, click square to target.")
    print("- Keys: [R] Reset | [M] Cycle Pattern | [+/-] Speed")
    print("- Toggles: [C] Coils | [F] Field | [P] Paths | [H] Heatmap | [D] Debug")
    print(f"- Hardware Mode: {'ACTIVE' if not sim_mode else 'SIMULATED'} | [Esc] Quit")
    print("======================================\n")

    # --- Run Simulation ---
    try:
        print("DEBUG: Starting board.run()...") # <<< DEBUG PRINT 14
        board.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\n--- An Error Occurred During Simulation ---")
        import traceback
        traceback.print_exc()
        print("-------------------------------------------")
    finally:
        # --- Clean Shutdown ---
        print("Initiating shutdown...")
        if hardware_controller:
            hardware_controller.shutdown()
        pygame.quit()
        print("Simulation ended gracefully.")

if __name__ == "__main__":
    print("DEBUG: Running main block...") # <<< DEBUG PRINT 15
    main()
    print("DEBUG: main() finished.") # <<< DEBUG PRINT 16