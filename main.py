#!/usr/bin/env python3
# Smart Electromagnetic Chess - Main Script

print("DEBUG: Starting main.py script...")

import sys
import argparse
import os  # Required for SDL environment variables
import pygame

print("DEBUG: Basic imports done.")

# Import the necessary classes from your modules
try:
    # Import constants needed here (or define defaults)
    from chess_simulation import ChessBoard, COIL_GRID_SIZE, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT
    print("DEBUG: Imported ChessBoard and constants.")
    from hardware_interface import ElectromagnetController
    print("DEBUG: Imported ElectromagnetController.")
    # Ensure SERIAL_AVAILABLE is accessible if needed for initial sim_mode check
    from hardware_interface import SERIAL_AVAILABLE as HW_SERIAL_AVAILABLE
except ImportError as e:
    print(f"FATAL: Failed to import required module: {e}")
    # Attempt to provide more specific feedback
    if "chess_simulation" in str(e):
        print("-> Please ensure 'chess_simulation.py' is in the same directory.")
    elif "hardware_interface" in str(e):
         print("-> Please ensure 'hardware_interface.py' is in the same directory.")
    # Add similar checks for other imports if needed
    sys.exit(1)
except Exception as e:
    print(f"FATAL: Unexpected error during import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def main():
    """Main entry point for the smart chess system"""
    print("DEBUG: Entered main() function.")
    parser = argparse.ArgumentParser(description='Smart Electromagnetic Chess System')
    parser.add_argument('--hardware', action='store_true', help='Enable hardware control')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0', help='Serial port for hardware control')
    parser.add_argument('--baud', type=int, default=115200, help='Serial baud rate')
    parser.add_argument('--rtscts', action='store_true', help='Enable RTS/CTS hardware flow control (if supported)')
    parser.add_argument('--dirpin', type=int, default=None, help='BCM GPIO pin for RS485 direction')
    parser.add_argument('--debug', action='store_true', help='Enable extra debug output (console)')
    parser.add_argument('--no-debug-start', action='store_true', help='Start with debug output OFF')


    print("DEBUG: Parsing arguments...")
    args = parser.parse_args()
    print(f"DEBUG: Arguments parsed: {args}")

    # Determine initial simulation mode
    # Use imported HW_SERIAL_AVAILABLE for clarity
    sim_mode = not args.hardware or not HW_SERIAL_AVAILABLE

    # --- Initialize Hardware Controller ---
    print("DEBUG: Initializing hardware controller logic...")
    hardware_controller = None
    if not sim_mode:
        print(f"DEBUG: Attempting hardware mode on {args.port}...")
        try:
            hardware_controller = ElectromagnetController(
                grid_size=COIL_GRID_SIZE,
                simulation_mode=False, # Explicitly request hardware mode
                com_port=args.port,
                baud_rate=args.baud,
                rs485_dir_pin=args.dirpin
            )
            # Update sim_mode based on whether hardware init actually succeeded
            sim_mode = hardware_controller.simulation_mode
        except Exception as e:
            print(f"FATAL: Unhandled error during hardware controller init: {e}")
            print("Exiting.")
            sys.exit(1)

    # Ensure controller exists even in pure simulation mode
    if hardware_controller is None:
         print("DEBUG: Creating ElectromagnetController in simulation mode...")
         hardware_controller = ElectromagnetController(grid_size=COIL_GRID_SIZE, simulation_mode=True)
         sim_mode = True # Ensure sim_mode is True

    print(f"DEBUG: Hardware controller ready (sim_mode={sim_mode}).")

    # --- Set Initial Window Size and Center Window ---
    try:
        # Set the SDL environment variable to center the window BEFORE pygame initialization
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        print("Window set to centered mode")
        
        print("DEBUG: Initializing Pygame...")
        pygame.init()
        print("DEBUG: Pygame initialized.")
        
        display_info = pygame.display.Info()
        initial_width = display_info.current_w
        initial_height = display_info.current_h
        print(f"Detected Screen Size: {initial_width}x{initial_height}")
    except Exception as e:
        print(f"Warning: Could not get display info ({e}). Using default size.")
        initial_width = DEFAULT_WINDOW_WIDTH
        initial_height = DEFAULT_WINDOW_HEIGHT

    # --- Initialize ChessBoard ---
    try:
        print("DEBUG: Initializing ChessBoard...")
        # Set initial debug mode: ON unless --no-debug-start is specified OR --debug is explicitly false (though default is false)
        initial_debug_mode = (args.debug or not args.no_debug_start)
        board = ChessBoard(
            hardware_controller=hardware_controller,
            debug_mode=initial_debug_mode,
            initial_window_size=(initial_width, initial_height) # Pass detected size
        )
        print("DEBUG: ChessBoard initialized.")
    except Exception as e:
        print(f"FATAL: Error initializing ChessBoard: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)


    # --- Print Instructions ---
    print("\n===== SMART ELECTROMAGNETIC CHESS =====")
    print("- Click piece to select, click square to target.")
    print("- Keys: [R] Reset Board & Apply PID Slider Values")
    print("- Keys: [M] Cycle Pattern | [+/-] Speed | [Esc] Quit")
    print("- Keys: [Right-click] or [Esc] Cancel Selection")
    print("- Toggles: [C] Coils | [F] Field | [P] Paths | [H] Heatmap")
    print("- Toggles: [D] Debug | [X] Centers | [Y] Position Dots")
    print(f"- Hardware Mode: {'ACTIVE' if not sim_mode else 'SIMULATED'}")
    print("- Use Sliders to adjust PID (effective after Reset)")
    print("======================================\n")

    # --- Run Simulation ---
    try:
        print("DEBUG: Starting board.run()...")
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
    print("DEBUG: Running main block...")
    main()
    print("DEBUG: main() finished.")