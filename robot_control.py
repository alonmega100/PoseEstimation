import time
import numpy as np
from numpy.ma.core import negative
from panda_py import Panda
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION CONSTANTS ---
ROBOT_IP = "172.16.0.2"
DEFAULT_SPEED_FACTOR = 0.02
# -------------------------------

class PandaController:
    """
    Encapsulates the connection and control logic for the Franka Panda robot.
    """

    # Class-level constant
    AXIS_MAP = {'x': 0, 'y': 1, 'z': 2}
    ROTATION_AXIS_MAP = {'yaw': 'z', 'pitch': 'y', 'roll': 'x'}

    def __init__(self, robot_ip: str, default_speed_factor: float = DEFAULT_SPEED_FACTOR):
        """Initializes the robot connection and motion settings."""
        self.ROBOT_IP = robot_ip
        self.speed_factor = default_speed_factor
        self.robot = self._connect_to_robot()
        print(f"[config] Current speed factor is {self.speed_factor}")

    def _connect_to_robot(self):
        """Handles the robot connection logic."""
        print(f"[robot] Attempting to connect to Panda at {self.ROBOT_IP}...")
        try:
            robot = Panda(self.ROBOT_IP)
            print(f"[robot] Connected successfully!")
            return robot
        except Exception as e:
            print(f"[error] Could not connect to robot: {e}")
            # Reraise or exit if connection is critical
            raise ConnectionError("Failed to connect to robot.")

    # --- Utility Methods ---

    def _display_pose(self, H: np.ndarray):
        """Displays the position and rotation of a 4x4 pose matrix."""
        pos = H[:3, 3]
        R_mat = H[:3, :3]

        # Convert rotation to Euler angles (ZYX = yaw,pitch,roll)
        euler_deg = R.from_matrix(R_mat).as_euler('zyx', degrees=True)

        print("\n[POSE INFO]")
        print(f"Position [m]: x={pos[0]:+.4f}, y={pos[1]:+.4f}, z={pos[2]:+.4f}")
        print(f"Rotation [deg]: yaw={euler_deg[0]:+.2f}, pitch={euler_deg[1]:+.2f}, roll={euler_deg[2]:+.2f}")
        print("-" * 60)

    def reset_position(self):
        """Moves the robot to its defined start position."""
        print("[robot] Moving to start position...")
        self.robot.move_to_start()
        self._display_pose(self.robot.get_pose())

    def _calculate_new_pose_rotation(self, T_current: np.ndarray, axis: str, delta: float) -> np.ndarray:
        """
        Calculates the new 4x4 pose matrix based on current pose and a rotation delta.
        Delta must be in degrees.
        """
        original_axis = axis
        axis = axis.lower()
        if axis not in self.ROTATION_AXIS_MAP:
            raise ValueError(f"Invalid rotation axis input: '{original_axis}'. Must be 'yaw', 'pitch', or 'roll'.")

        # 1. Extract the current rotation matrix (R_current) and translation (t_current)
        R_current_mat = T_current[:3, :3]
        t_current = T_current[:3, 3]

        # 2. Convert current rotation matrix to a Rotation object
        r_current = R.from_matrix(R_current_mat)

        # 3. Create the delta rotation (in radians for scipy's rotation)
        delta_rad = np.deg2rad(delta)
        scipy_axis = self.ROTATION_AXIS_MAP[axis]  # e.g., 'z' for yaw

        # The rotation is applied *incrementally* relative to the robot's base frame.
        # This is equivalent to applying a rotation about the base frame's x, y, or z axis.
        r_delta = R.from_rotvec(delta_rad * np.array([1 if scipy_axis == 'x' else 0,
                                                      1 if scipy_axis == 'y' else 0,
                                                      1 if scipy_axis == 'z' else 0]))

        # 4. Compose the rotations: New Rotation = Delta Rotation * Current Rotation
        # Note: Order matters! Pre-multiplying (r_delta * r_current) applies the rotation
        # relative to the world/base frame (which is what we want for direct yaw/pitch/roll control).
        r_new = r_delta * r_current

        # 5. Convert the new rotation back to a matrix
        R_new_mat = r_new.as_matrix()

        # 6. Build the new homogeneous matrix
        T_new = np.identity(4)
        T_new[:3, :3] = R_new_mat
        T_new[:3, 3] = t_current  # Keep translation the same

        print(f"Applying rotation delta {delta:+.2f} degrees to {axis}.")
        return T_new

    def _calculate_new_pose_translation(self, T_current: np.ndarray, axis: str, delta: float) -> np.ndarray:
        """
        Calculates the new 4x4 pose matrix based on current pose and a translation delta.
        """
        original_axis = axis

        # Input validation
        if axis not in self.AXIS_MAP:
            raise ValueError(f"Invalid axis input: '{original_axis}'. Must be 'x', 'y', or 'z'.")

        n = self.AXIS_MAP[axis]
        T_new = T_current.copy()

        # Apply the delta to the translational component
        print(f"Applying delta {delta:+.4f} to {axis}-axis (index {n}).")
        T_new[n, 3] += delta

        return T_new

    # --- Main Control Loop ---

    def position_control_loop(self):
        """Allows for interactive control of the robot's end-effector position and orientation."""
        print("\n--- Starting Interactive Position/Orientation Control ---")


        while True:
            try:
                # Get current pose and display it
                H_current = self.robot.get_pose()
                self._display_pose(H_current)

                # --- User Input ---
                print("Input Format: <Axis/Rotation> <Delta> (e.g., 'x 0.1' or 'yaw 10')")  # Consolidated prompt
                print("Type 'q' or 'quit' to exit this mode.")
                user_input = input("Enter command (or 'q'): ").lower().strip()
                if user_input in ["q", "quit"]:
                    break
                if user_input[0] == "-":
                    multiply_delta_by_minus_one = True
                    user_input = user_input[1:]
                else:
                    multiply_delta_by_minus_one = False
                parts = user_input.split()
                if len(parts) != 2:
                    print("[warning] Invalid format. Use '<axis> <delta>'.")
                    continue

                n_str, d_str = parts

                try:
                    delta = -1 * float(d_str) if multiply_delta_by_minus_one else float(d_str)
                    axis = n_str
                except ValueError:
                    print("[warning] Invalid delta (must be a number). Skipping move.")
                    continue

                # --- Decision and Calculation ---
                is_rotation = axis in self.ROTATION_AXIS_MAP
                is_translation = axis in self.AXIS_MAP

                if is_translation:
                    # Use the translation method
                    H_new = self._calculate_new_pose_translation(H_current, axis, delta)
                elif is_rotation:
                    # Use the new rotation method
                    H_new = self._calculate_new_pose_rotation(H_current, axis, delta)
                else:
                    raise ValueError(f"Unknown axis or rotation name: '{axis}'.")

                # --- Execute Move ---
                print("\n[robot] Moving...")
                self.robot.move_to_pose(H_new, speed_factor=self.speed_factor)  # Blocking call

                # Verify and display the actual pose after the move
                H_actual = self.robot.get_pose()
                self._display_pose(H_actual)

            except ValueError as ve:
                print(f"[warning] {ve}")
                time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n--- Position/Orientation Control Stopped. ---")
                break
            except Exception as e:
                print(f"[error] An unhandled exception occurred during movement: {e}")
                time.sleep(1)

    def set_speed_factor(self, factor: float):  # Renamed for clarity and consistency
        """Sets the motion speed factor with bounds checking."""
        if not (0.0 < factor <= 1.0):  # Check limits
            print("[warning] Speed factor must be greater than 0.0 and less than or equal to 1.0.")
            return

        before = self.speed_factor
        self.speed_factor = factor
        print(f"[config] Successfully set speed factor from {before:.2f} to {self.speed_factor:.2f}")

def main():

    try:
        controller = PandaController(ROBOT_IP)
    except ConnectionError:
        print("Application terminated due to failed robot connection.")
        return

    while True:
        print("\n--- Main Menu ---")
        print("Type 'reset' to go to start position.")
        print("Type 'pos' for interactive position control.")
        print("Type 'speed x' to change the speed factor. x between 0 and 1.")
        print("Type 'quit' or 'q' to exit.")

        command = input("Choose control function: ").lower().strip()

        if command == "reset":
            controller.reset_position()
        elif command == "pos":
            controller.position_control_loop()
        elif command.startswith("speed"):
            print("--- Attempting To Change Robot Speed Factor ---")
            parts = command.split()
            if len(parts) != 2:
                print("[warning] Invalid speed command format. Use 'speed <number>'.")
                continue
            try:
                factor = float(parts[1])
                controller.set_speed_factor(factor)
            except ValueError:
                print("[warning] Speed factor must be a valid number.")
        elif command in ["quit", "q"]:
            print("Exiting application.")
            break
        else:
            print(f"Unknown command: '{command}'. Please try again.")


if __name__ == "__main__":
    main()