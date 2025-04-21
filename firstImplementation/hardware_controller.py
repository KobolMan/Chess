import time
import math
try:
    # Try to import hardware-specific libraries
    # These will fail in simulation mode, which is expected
    import serial
    import RPi.GPIO as GPIO
except ImportError:
    pass

class ElectromagnetGridController:
    """
    Controller for the electromagnet grid that can operate in both simulation
    and real hardware modes. Uses RS-485 protocol for communication with the coils.
    
    In real hardware mode, it controls the physical electromagnet coils through 
    a star-connected 3-wire RS-485 bus.
    """
    
    def __init__(self, grid_size=20, simulation_mode=True, com_port='/dev/ttyUSB0'):
        """
        Initialize the electromagnet grid controller.
        
        Args:
            grid_size: Size of the coil grid (default 20x20)
            simulation_mode: If True, run in simulation mode without hardware
            com_port: Serial port for RS-485 communication (used in real mode)
        """
        self.grid_size = grid_size
        self.simulation_mode = simulation_mode
        self.com_port = com_port
        self.serial_conn = None
        
        # Initialize coil state matrix
        self.coil_states = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Initialize hardware if not in simulation mode
        if not simulation_mode:
            self._initialize_hardware()
    
    def _initialize_hardware(self):
        """Initialize the hardware components for real operation"""
        try:
            # Setup serial connection for RS-485
            self.serial_conn = serial.Serial(
                port=self.com_port,
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            
            # Setup GPIO if needed (for direction control or other functions)
            GPIO.setmode(GPIO.BCM)
            self.rs485_dir_pin = 18  # Example pin for RS485 direction control
            GPIO.setup(self.rs485_dir_pin, GPIO.OUT)
            GPIO.output(self.rs485_dir_pin, GPIO.LOW)  # Set to receive initially
            
            print(f"Hardware initialized on {self.com_port}")
            
            # Send initialization command to all coils
            self._send_command('INIT', 0, 0, 0)
            
        except Exception as e:
            print(f"Hardware initialization failed: {e}")
            self.simulation_mode = True  # Fall back to simulation mode
    
    def set_coil_power(self, row, col, power):
        """
        Set the power level for a specific coil.
        
        Args:
            row: Row index of the coil
            col: Column index of the coil
            power: Power level (0-100%)
        """
        # Validate parameters
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            print(f"Invalid coil position: ({row}, {col})")
            return
        
        if not (0 <= power <= 100):
            power = max(0, min(100, power))  # Clamp to valid range
        
        # Update internal state
        self.coil_states[row][col] = power
        
        # If not in simulation mode, send command to hardware
        if not self.simulation_mode:
            self._send_command('SET', row, col, power)
    
    def reset_all_coils(self):
        """Turn off all electromagnet coils"""
        # Reset internal state
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.coil_states[row][col] = 0
        
        # If not in simulation mode, send command to hardware
        if not self.simulation_mode:
            self._send_command('RESET', 0, 0, 0)
    
    def move_piece(self, start_row, start_col, end_row, end_col):
        """
        Move a chess piece from one position to another.
        
        Args:
            start_row, start_col: Starting position
            end_row, end_col: Ending position
        """
        print(f"Moving piece from ({start_row}, {start_col}) to ({end_row}, {end_col})")
        
        if self.simulation_mode:
            # In simulation mode, just print the path
            print("Simulation mode: piece movement path calculated")
            return
        
        # Calculate path for piece to follow (direct line for simplicity)
        path = self._calculate_path(start_row, start_col, end_row, end_col)
        
        # Move the piece along the path
        self._move_along_path(path)
    
    def move_captured_piece(self, path):
        """
        Move a captured piece along a given path.
        
        Args:
            path: List of (row, col) points defining the path
        """
        print(f"Moving captured piece along path with {len(path)} points")
        
        if self.simulation_mode:
            # In simulation mode, just print the path
            print("Simulation mode: captured piece path calculated")
            return
        
        # Move the piece along the given path
        self._move_along_path(path)
    
    def _move_along_path(self, path):
        """
        Move a piece along a specified path by activating coils sequentially.
        
        Args:
            path: List of (row, col) points defining the path
        """
        if self.simulation_mode:
            return
            
        # Activate coils along the path in sequence
        for i, (row, col) in enumerate(path):
            # Reset all coils
            self.reset_all_coils()
            
            # Activate coils around the current position
            for r in range(max(0, row-2), min(self.grid_size, row+3)):
                for c in range(max(0, col-2), min(self.grid_size, col+3)):
                    # Calculate distance from current position
                    distance = math.sqrt((r - row)**2 + (c - col)**2)
                    
                    if distance < 3:
                        # Calculate power based on distance (closer = stronger)
                        power = int(100 * (1 - distance/3))
                        self.set_coil_power(r, c, power)
            
            # Wait for the piece to move
            # Time delay should be adjusted based on piece weight, coil strength, etc.
            time.sleep(0.05)
    
    def _calculate_path(self, start_row, start_col, end_row, end_col):
        """
        Calculate a straight-line path between two points.
        
        Args:
            start_row, start_col: Starting position
            end_row, end_col: Ending position
            
        Returns:
            List of (row, col) tuples representing the path
        """
        # Calculate path using Bresenham's line algorithm
        path = []
        
        dx = abs(end_col - start_col)
        dy = abs(end_row - start_row)
        sx = 1 if start_col < end_col else -1
        sy = 1 if start_row < end_row else -1
        err = dx - dy
        
        x, y = start_col, start_row
        
        while True:
            path.append((y, x))  # Note: y = row, x = col
            
            if x == end_col and y == end_row:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return path
    
    def _send_command(self, cmd_type, row, col, value):
        """
        Send a command to the coil grid hardware using RS-485 protocol.
        
        Args:
            cmd_type: Command type ('SET', 'RESET', 'INIT')
            row, col: Coil coordinates
            value: Command value (power level for 'SET')
        """
        if self.simulation_mode or not self.serial_conn:
            return
            
        try:
            # Calculate coil address from row and column
            coil_address = row * self.grid_size + col
            
            # Prepare command packet
            if cmd_type == 'SET':
                # Set command: [STX][Address(2 bytes)][CMD='S'][Value(1 byte)][Checksum][ETX]
                packet = bytearray([0x02])  # STX
                packet.extend(coil_address.to_bytes(2, byteorder='big'))
                packet.extend(b'S')
                packet.extend([value])
            elif cmd_type == 'RESET':
                # Reset command: [STX][0xFFFF][CMD='R'][0x00][Checksum][ETX]
                packet = bytearray([0x02])  # STX
                packet.extend((0xFFFF).to_bytes(2, byteorder='big'))  # Broadcast address
                packet.extend(b'R')
                packet.extend([0])
            elif cmd_type == 'INIT':
                # Init command: [STX][0xFFFF][CMD='I'][0x00][Checksum][ETX]
                packet = bytearray([0x02])  # STX
                packet.extend((0xFFFF).to_bytes(2, byteorder='big'))  # Broadcast address
                packet.extend(b'I')
                packet.extend([0])
            
            # Calculate checksum (simple XOR of all bytes)
            checksum = 0
            for b in packet[1:]:
                checksum ^= b
            packet.extend([checksum])
            packet.extend([0x03])  # ETX
            
            # Set RS-485 direction pin to transmit mode
            GPIO.output(self.rs485_dir_pin, GPIO.HIGH)
            
            # Send the packet
            self.serial_conn.write(packet)
            self.serial_conn.flush()
            
            # Wait for transmission to complete and switch back to receive mode
            time.sleep(0.001)  # Adjust based on baud rate
            GPIO.output(self.rs485_dir_pin, GPIO.LOW)
            
        except Exception as e:
            print(f"Error sending command: {e}")
    
    def reset(self):
        """Reset the controller to initial state"""
        # Turn off all coils
        self.reset_all_coils()
        
        # Re-initialize if in hardware mode
        if not self.simulation_mode:
            self._send_command('INIT', 0, 0, 0)
    
    def shutdown(self):
        """Clean shutdown of the controller"""
        # Turn off all coils
        self.reset_all_coils()
        
        # Close serial connection if it exists
        if self.serial_conn:
            self.serial_conn.close()
            
        # Clean up GPIO if used
        if not self.simulation_mode:
            try:
                GPIO.cleanup()
            except:
                pass
        
        print("Electromagnet grid controller shut down")


# Example of a more specialized CAN bus version that could be implemented
class CANBusElectromagnetController:
    """
    Alternative implementation using CAN bus protocol instead of RS-485.
    This would be used if higher reliability or more advanced networking is needed.
    """
    
    def __init__(self, grid_size=20, simulation_mode=True):
        """Initialize the CAN bus controller"""
        self.grid_size = grid_size
        self.simulation_mode = simulation_mode
        
        # Initialize can bus
        if not simulation_mode:
            try:
                import can
                self.bus = can.interface.Bus(bustype='socketcan', 
                                             channel='can0', 
                                             bitrate=500000)
                print("CAN bus initialized")
            except Exception as e:
                print(f"CAN bus initialization failed: {e}")
                self.simulation_mode = True
    
    def set_coil_power(self, row, col, power):
        """Set power level for a specific coil using CAN bus"""
        if self.simulation_mode:
            return
            
        try:
            # Calculate coil ID from row and column
            coil_id = (row * self.grid_size + col) + 0x100
            
            # Prepare CAN message
            # ID: 0x100 + coil number
            # Data: [CMD='S', power]
            msg = can.Message(
                arbitration_id=coil_id,
                data=[ord('S'), power],
                is_extended_id=False
            )
            
            # Send the message
            self.bus.send(msg)
            
        except Exception as e:
            print(f"Error sending CAN message: {e}")
    
    def shutdown(self):
        """Clean shutdown of the CAN controller"""
        if not self.simulation_mode:
            try:
                self.bus.shutdown()
            except:
                pass


# Example usage
if __name__ == "__main__":
    # Test the controller in simulation mode
    controller = ElectromagnetGridController(simulation_mode=True)
    
    # Set a pattern of coils
    for r in range(5):
        for c in range(5):
            controller.set_coil_power(r, c, 100)
    
    # Simulate moving a piece
    controller.move_piece(0, 0, 7, 7)
    
    # Clean up
    controller.shutdown()