from panda_controller import *


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
        print("Type 'speed x' to change the speed factor (x between 0 and 1).")
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