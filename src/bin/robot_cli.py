import time

from src.robot.panda_controller import PandaController
from src.utils.concurrent_runner import run_concurrent_system


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

            updated_H, valid_move = controller.pos_command_to_H(user_input)
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
        controller = PandaController()
    except ConnectionError:
        print("Application terminated due to failed robot connection.")
        return

    while True:
        print("\n--- Main Menu ---")
        print("Type 'reset' to go to start position.")
        print("Type 'con' to start concurrent_runner")
        print("Type 'pos' for interactive position control.")
        print("Type 'speed x' to change the speed factor (x between 0 and 1).")
        print("Type 'quit' or 'q' to exit.")

        command = input("Choose control function: ").lower().strip()

        if command == "reset":
            controller.reset_position()
        elif command == "con":
            controller.reset_position()
            run_concurrent_system(controller)
        elif command == "cond":
            controller.reset_position()
            run_concurrent_system(controller, discard=True)
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