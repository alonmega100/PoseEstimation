import time
import numpy as np
from panda_py import Panda
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION CONSTANTS ---
ROBOT_IP = "172.16.0.2"
DEFAULT_SPEED_FACTOR = 0.02


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
        print(f"[config] Current speed factor is {self.speed_factor:.2f}")

    def _connect_to_robot(self):
        """Handles the robot connection logic."""
        print(f"[robot] Attempting to connect to Panda at {self.ROBOT_IP}...")
        try:
            robot = Panda(self.ROBOT_IP)
            print(f"[robot] Connected successfully!")
            return robot
        except Exception as e:
            print(f"[error] Could not connect to robot: {e}")
            raise ConnectionError("Failed to connect to robot.")

    # --- Utility Methods ---

    def _display_pose(self, H: np.ndarray):
        """Displays the position and rotation of a 4x4 pose matrix."""
        pos = H[:3, 3]
        R_mat = H[:3, :3]

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
        """
        # Note: The main loop ensures 'axis' is lowercase and valid before calling this.

        # 1. Extract and Convert Rotation
        r_current = R.from_matrix(T_current[:3, :3])
        t_current = T_current[:3, 3]

        # 2. Create the delta rotation
        delta_rad = np.deg2rad(delta)
        scipy_axis = self.ROTATION_AXIS_MAP[axis]

        r_delta = R.from_rotvec(delta_rad * np.array([1 if scipy_axis == 'x' else 0,
                                                      1 if scipy_axis == 'y' else 0,
                                                      1 if scipy_axis == 'z' else 0]))

        # 3. Compose and Rebuild
        r_new = r_delta * r_current
        R_new_mat = r_new.as_matrix()

        T_new = np.identity(4)
        T_new[:3, :3] = R_new_mat
        T_new[:3, 3] = t_current

        print(f"Applying rotation delta {delta:+.2f} degrees to {axis}.")
        return T_new

    def _calculate_new_pose_translation(self, T_current: np.ndarray, axis: str, delta: float) -> np.ndarray:
        """
        Calculates the new 4x4 pose matrix based on current pose and a translation delta.
        Assumes 'axis' is a valid key in AXIS_MAP.
        """
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
                H_current = self.robot.get_pose()
                self._display_pose(H_current)

                # --- User Input Prompt ---
                print("Input Format: <Axis/Rotation> <Delta> (e.g., 'x 0.1' or 'yaw 10')")
                print("Can accept multiple pairs: 'x 0.1 yaw 5'")
                print("Type 'q' or 'quit' to exit this mode.")
                user_input = input("Enter command (or 'q'): ").lower().strip()

                if user_input in ["q", "quit"]:
                    break
                if not user_input:
                    continue

                parts = user_input.split()
                if len(parts) % 2 != 0:
                    print(f"[warning] Invalid command length ({len(parts)} parts). Commands must be in pairs.")
                    continue

                updated_H = H_current.copy()
                valid_move = True

                # --- Parse and Calculate All Commands ---
                for i in range(0, len(parts), 2):
                    n_str, d_str = parts[i], parts[i + 1]
                    axis = n_str

                    try:
                        delta = float(d_str)
                    except ValueError:
                        print(f"[warning] Delta '{d_str}' is not a valid number. Skipping move.")
                        valid_move = False
                        break

                    # Ensure axis is clean (no leading '-')
                    if axis.startswith("-"):
                        axis = axis[1:]
                        delta *= -1  # Invert delta since the sign was on the axis

                    # --- Decision and Calculation ---
                    is_rotation = axis in self.ROTATION_AXIS_MAP
                    is_translation = axis in self.AXIS_MAP

                    if is_translation:
                        updated_H = self._calculate_new_pose_translation(updated_H, axis, delta)
                    elif is_rotation:
                        updated_H = self._calculate_new_pose_rotation(updated_H, axis, delta)
                    else:
                        raise ValueError(f"Unknown axis or rotation name: '{axis}'.")

                if not valid_move:
                    continue

                # --- Execute Move ---
                print("\n[robot] Moving...")
                self.robot.move_to_pose(updated_H, speed_factor=self.speed_factor)


            except ValueError as ve:
                print(f"[warning] {ve}")
                time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n--- Position/Orientation Control Stopped. ---")
                break
            except Exception as e:
                print(f"[error] An unhandled exception occurred during movement: {e}")
                time.sleep(1)

    def set_speed_factor(self, factor: float):
        """Sets the motion speed factor with bounds checking."""
        if not (0.0 < factor <= 1.0):
            print("[warning] Speed factor must be greater than 0.0 and less than or equal to 1.0.")
            return

        before = self.speed_factor
        self.speed_factor = factor
        print(f"[config] Successfully set speed factor from {before:.2f} to {self.speed_factor:.2f}")
