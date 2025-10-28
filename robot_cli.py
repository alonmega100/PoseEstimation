import numpy as np
from parso.python.tree import String
from panda_controller import PandaController, ROBOT_IP
import time


def pos_command_to_H(command: str,
                     H_initial: np.ndarray,controller: PandaController) -> tuple[np.ndarray | None, bool]:
    """
    Parses a position command string and calculates the final 4x4 homogeneous
    transformation matrix (H_final) based on an initial pose.

    Returns: (H_final, valid_move_status)
    """
    if not command:
        return None, False

    # 1. Initialize the new pose with the current pose
    updated_H = H_initial.copy()
    valid_move = True

    # 2. Split and Validate Command Length
    parts = command.split()
    if len(parts) % 2 != 0:
        print(f"[warning] Invalid command length ({len(parts)} parts). Commands must be in pairs.")
        return None, False

    # --- Parse and Calculate All Commands ---
    for i in range(0, len(parts), 2):
        n_str, d_str = parts[i], parts[i + 1]
        axis = n_str.lower()  # Normalize axis input

        try:
            delta = float(d_str)
        except ValueError:
            print(f"[warning] Delta '{d_str}' is not a valid number. Skipping move.")
            return None, False

        # Ensure axis is clean (no leading '-')
        if axis.startswith("-"):
            axis = axis[1:]
            delta *= -1

        # --- Decision and Calculation (Access constants via controller) ---
        is_rotation = axis in controller.ROTATION_AXIS_MAP
        is_translation = axis in controller.AXIS_MAP

        if is_translation:
            updated_H = controller._calculate_new_pose_translation(updated_H, axis, delta)
        elif is_rotation:
            updated_H = controller._calculate_new_pose_rotation(updated_H, axis, delta)
        else:
            print(f"[warning] Unknown axis or rotation name: '{axis}'.")
            return None, False  # Fail immediately on unknown axis

    return updated_H, valid_move


def position_control_loop(controller: PandaController):
    """Allows for interactive control of the robot's end-effector position and orientation."""
    print("\n--- Starting Interactive Position/Orientation Control ---")

    while True:
        try:
            H_current = controller.robot.get_pose()
            controller._display_pose(H_current)

            # --- User Input Prompt ---
            print("Input Format: <Axis/Rotation> <Delta> (e.g., 'x 0.1' or 'yaw 10')")
            print("Can accept multiple pairs: 'x 0.1 yaw 5'")
            print("Type 'q' or 'quit' to exit this mode.")
            user_input = input("Enter command (or 'q'): ").lower().strip()

            if user_input in ["q", "quit"]:
                break

            updated_H, valid_move = pos_command_to_H(user_input, H_current, controller)
            if not valid_move:
                continue

            # --- Execute Move ---
            print("\n[robot] Moving...")
            controller.robot.move_to_pose(updated_H, speed_factor=controller.speed_factor)


        except ValueError as ve:
            print(f"[warning] {ve}")
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n--- Position/Orientation Control Stopped. ---")
            break
        except Exception as e:
            print(f"[error] An unhandled exception occurred during movement: {e}")
            time.sleep(1)


def main():
    try:
        controller = PandaController(ROBOT_IP)
    except ConnectionError:
        print("Application terminated due to failed robot connection.")
        return

    while True:
        print("\n--- Main Menu ---")
        print("Type 'reset' to go to start position.")
        print("Type 'cam' to start camera thread")
        print("Type 'pos' for interactive position control.")
        print("Type 'speed x' to change the speed factor (x between 0 and 1).")
        print("Type 'quit' or 'q' to exit.")

        command = input("Choose control function: ").lower().strip()

        if command == "reset":
            controller.reset_position()
        elif command == "pos":
            position_control_loop(controller)
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