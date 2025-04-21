# hardware_interface.py

import time
import numpy as np
import math

# Attempt to import hardware libraries, but don't fail if they're missing
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: PySerial library not found. Hardware control unavailable.")

try:
    # Using a generic GPIO library name placeholder
    # Replace with 'import RPi.GPIO as GPIO' or specific library if needed
    # For simulation, we don't need a real GPIO library
    # import RPi.GPIO as GPIO
    GPIO_AVAILABLE = False # Assume false unless specific import works
    # if SERIAL_AVAILABLE: # Only try GPIO if serial works (likely on RPi)
    #     try:
    #         import RPi.GPIO as GPIO
    #         GPIO_AVAILABLE = True
    #     except ImportError:
    #         print("Warning: RPi.GPIO library not found. Hardware GPIO control unavailable.")
    #         GPIO_AVAILABLE = False
except ImportError:
    GPIO_AVAILABLE = False

class ElectromagnetController:
    """Interface for controlling the physical electromagnet coil grid."""

    def __init__(self, grid_size=20, simulation_mode=True, com_port='/dev/ttyUSB0', baud_rate=115200, rs485_dir_pin=None):
        """
        Initialize the controller.

        Args:
            grid_size: Size of the coil grid (e.g., 20 for 20x20).
            simulation_mode: If True, run without attempting hardware communication.
            com_port: Serial port name (e.g., '/dev/ttyUSB0', 'COM3').
            baud_rate: Serial communication speed.
            rs485_dir_pin: GPIO pin number (BCM mode) for RS485 direction control, if needed.
        """
        self.grid_size = grid_size
        self.simulation_mode = simulation_mode or not SERIAL_AVAILABLE # Force sim if no serial
        self.com_port = com_port
        self.baud_rate = baud_rate
        self.rs485_dir_pin = rs485_dir_pin
        self.serial_conn = None
        self._gpio_initialized = False

        # Store the last sent state to optimize commands (optional)
        self.last_power_state = np.zeros((grid_size, grid_size))
        self.last_current_state = np.zeros((grid_size, grid_size))

        if not self.simulation_mode:
            self._initialize_hardware()
        else:
            print("Hardware Controller running in SIMULATION mode.")

    def _initialize_hardware(self):
        """Initialize serial and GPIO for real hardware operation."""
        print(f"Attempting hardware initialization on {self.com_port}...")
        try:
            self.serial_conn = serial.Serial(
                port=self.com_port,
                baudrate=self.baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1 # Shorter timeout
            )
            print(f"Serial port {self.com_port} opened successfully.")

            # Setup GPIO if pin provided and library available
            if self.rs485_dir_pin is not None and GPIO_AVAILABLE:
                # import RPi.GPIO as GPIO # Import here if confirmed available
                # GPIO.setmode(GPIO.BCM) # Use Broadcom pin numbering
                # GPIO.setup(self.rs485_dir_pin, GPIO.OUT)
                # GPIO.output(self.rs485_dir_pin, GPIO.LOW) # Set to receive mode initially
                # self._gpio_initialized = True
                # print(f"GPIO pin {self.rs485_dir_pin} initialized for RS485 direction.")
                print("Note: GPIO initialization logic commented out - ensure correct library and setup.")


            # Send initialization/reset command to hardware grid?
            print("Sending RESET command to hardware grid.")
            self.reset_all_coils() # Use reset to clear hardware state

        except serial.SerialException as e:
            print(f"ERROR: Could not open serial port {self.com_port}: {e}")
            print("FALLING BACK TO SIMULATION MODE.")
            self.simulation_mode = True
            if self.serial_conn:
                self.serial_conn.close()
            self.serial_conn = None
        except Exception as e:
            print(f"ERROR: Hardware initialization failed: {e}")
            print("FALLING BACK TO SIMULATION MODE.")
            self.simulation_mode = True
            if self.serial_conn:
                self.serial_conn.close()
            self.serial_conn = None
            # if self._gpio_initialized:
                # GPIO.cleanup()


    def set_coil_power(self, row, col, power, current_direction=1):
        """
        Low-level function to set a single coil's power on the hardware.

        Args:
            row, col: Coil coordinates.
            power: Power level (0-100).
            current_direction: +1 for Repel, -1 for Attract.
        """
        if self.simulation_mode:
            # In simulation mode, maybe log the intended action
            # print(f"[SIM] Set Coil ({row}, {col}): Power={power:.1f}, Dir={current_direction}")
            return

        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            print(f"Warning: Invalid coil position ({row}, {col}) for set_coil_power.")
            return

        # Send the command to the hardware via serial
        # Power needs conversion (e.g., 0-100 -> 0-255 or PWM duty cycle)
        # Current direction might map to a specific bit or value
        # *** This depends heavily on your hardware protocol ***
        power_value = np.clip(int(power * 2.55), 0, 255) # Example: 0-100 -> 0-255
        current_value = 1 if current_direction > 0 else 0 # Example: 1=repel, 0=attract

        self._send_command('SET', row, col, power_value, current_value)


    def apply_state(self, power_matrix, current_matrix):
        """
        Applies the entire desired grid state to the hardware.
        Optimized to only send commands for coils that have changed state.
        """
        if self.simulation_mode:
            # print("[SIM] Applying new state (not sending to hardware).")
            # Optionally update internal state if tracking:
            # self.last_power_state = power_matrix.copy()
            # self.last_current_state = current_matrix.copy()
            return

        commands_sent = 0
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                target_power = power_matrix[r, c]
                target_current_dir = current_matrix[r, c] # Should be +1, -1, or 0

                # Convert desired state to hardware values
                hw_power_value = np.clip(int(target_power * 2.55), 0, 255)
                hw_current_value = 1 if target_current_dir > 0 else 0 # Example

                # Compare with last sent state (implement comparison logic if optimizing)
                # Example simple optimization: Check if power changed significantly or became zero/non-zero
                # A more robust way tracks exact power/current values sent last time
                if abs(target_power - self.last_power_state[r,c]) > 1 or \
                   (target_power == 0 and self.last_power_state[r,c] != 0) or \
                   (target_power > 0 and self.last_power_state[r,c] == 0) or \
                   (target_power > 0 and np.sign(target_current_dir) != np.sign(self.last_current_state[r,c])):

                    # State changed, send update
                    self._send_command('SET', r, c, hw_power_value, hw_current_value)
                    commands_sent += 1
                    # Update last known state
                    self.last_power_state[r, c] = target_power
                    self.last_current_state[r, c] = target_current_dir
                    time.sleep(0.002) # Small delay between commands if needed

        # if commands_sent > 0:
        #     print(f"Sent {commands_sent} coil update commands to hardware.")


    def reset_all_coils(self):
        """Turns off all coils on the hardware."""
        print("Resetting all hardware coils.")
        if not self.simulation_mode:
            # Send the hardware-specific RESET command
            self._send_command('RESET', 0, 0, 0, 0) # Row/col/power/current likely ignored for RESET

        # Reset internal tracking state
        self.last_power_state.fill(0)
        self.last_current_state.fill(0)

    def _send_command(self, cmd_type, row, col, value1, value2):
        """
        Internal function to format and send a command packet via serial.
        *** Adapt this function precisely to your hardware protocol! ***
        """
        if self.simulation_mode or not self.serial_conn or not self.serial_conn.is_open:
            # print(f"[SIM] Command '{cmd_type}' Args: {row},{col},{value1},{value2}")
            return

        try:
            # --- Example Packet Formatting (Modify!) ---
            packet = bytearray()
            stx = 0x02 # Start of Text
            etx = 0x03 # End of Text

            if cmd_type == 'SET':
                # Example: [STX][AddrH][AddrL]['S'][Power][Current][Chk][ETX]
                if not (0 <= row < self.grid_size and 0 <= col < self.grid_size): return # Invalid address
                address = row * self.grid_size + col
                packet.append(stx)
                packet.extend(address.to_bytes(2, byteorder='big')) # 2-byte address
                packet.append(ord('S')) # Command byte
                packet.append(value1) # Power (0-255)
                packet.append(value2) # Current direction (0 or 1)

            elif cmd_type == 'RESET':
                # Example: [STX][0xFF][0xFF]['R'][0x00][0x00][Chk][ETX] (Broadcast Reset)
                packet.append(stx)
                packet.extend([0xFF, 0xFF]) # Broadcast address
                packet.append(ord('R'))
                packet.extend([0x00, 0x00]) # Placeholder values

            # Add more commands like 'INIT', 'GET_STATUS' as needed
            else:
                print(f"Warning: Unknown command type '{cmd_type}'")
                return

            # Calculate Checksum (Example: simple XOR)
            checksum = 0
            for byte in packet[1:]: # XOR all bytes after STX
                checksum ^= byte
            packet.append(checksum)
            packet.append(etx)
            # --- End Example Packet Formatting ---


            # Set RS485 direction to Transmit (if applicable)
            # if self._gpio_initialized:
            #     GPIO.output(self.rs485_dir_pin, GPIO.HIGH)
            #     time.sleep(0.001) # Short delay before sending

            # Send the packet
            # print(f"Sending Packet: {packet.hex()}") # Debug: print hex packet
            bytes_written = self.serial_conn.write(packet)
            self.serial_conn.flush() # Ensure data is sent

            # Set RS485 direction back to Receive (if applicable)
            # if self._gpio_initialized:
                # Allow time for transmission to complete based on packet size and baud rate
                # time_to_send = (len(packet) * 10) / self.baud_rate # 10 bits per byte (start, 8 data, stop)
                # time.sleep(time_to_send + 0.001) # Add small buffer
                # GPIO.output(self.rs485_dir_pin, GPIO.LOW)

            if bytes_written != len(packet):
                 print(f"Warning: Serial write incomplete. Sent {bytes_written}/{len(packet)} bytes.")


        except serial.SerialTimeoutException:
            print("ERROR: Serial write timeout.")
        except Exception as e:
            print(f"ERROR: Failed to send command '{cmd_type}': {e}")
            # Consider attempting to re-initialize or enter simulation mode on error
            # self.simulation_mode = True

    def shutdown(self):
        """Cleanly close hardware connections."""
        print("Shutting down hardware controller...")
        if not self.simulation_mode:
            try:
                self.reset_all_coils() # Turn off coils before closing
                time.sleep(0.1) # Allow time for reset command
                if self.serial_conn and self.serial_conn.is_open:
                    self.serial_conn.close()
                    print("Serial port closed.")
                # if self._gpio_initialized:
                    # GPIO.cleanup()
                    # print("GPIO cleaned up.")
            except Exception as e:
                print(f"Error during hardware shutdown: {e}")
        self.simulation_mode = True # Ensure it's in sim mode after shutdown
        print("Hardware controller shut down complete.")