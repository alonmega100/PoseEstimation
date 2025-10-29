import queue
import threading
import time
import numpy as np
import cv2
import sys

# Import your clean components
from panda_controller import PandaController, ROBOT_IP
from april_tag_processor import AprilTagProcessor, WORLD_TAG_ID, FRAME_W, FRAME_H, OBJ_TAG_IDS, WORLD_TAG_SIZE, \
    OBJ_TAG_SIZE

# --- 1. SHARED DATA STRUCTURE ---

# This dictionary holds the current, latest state of the system
shared_state = {
    # Pose of the Tag relative to the World Anchor (^A T_T)
    "tag_pose_A": {},

    # NEW: Store latest visualization image from each camera {serial_num: image}
    "vision_image": {},

    # Pose of the Robot End-Effector (^R T_ee)
    "robot_pose_R": np.identity(4),

    "command_queue": queue.Queue(),
    # System control flag
    "running": True
}
# Lock to ensure thread-safe access to shared_state
state_lock = threading.Lock()

# --- 2. THREAD FUNCTIONS ---
def command_writer_thread(shared_state: dict, lock: threading.Lock):
    print("Command writer thread started")
    while shared_state["running"]:
        try:
            # Note: Input is inherently blocking, which is fine for this writer thread.
            command = input("Type command (e.g., 'x 0.05'), 'l' for log, or 'q' to quit: ").strip().lower()

            if not command:
                continue

            # --- Handle Quit Command ---
            if command in ["q", "exit", "quit"]:
                with lock:
                    shared_state["running"] = False
                    # Put a dummy command to unblock other threads if needed
                    shared_state["command_queue"].put("quit_signal")
                break

            # --- Handle Log Command ---
            elif command in ["log", "l"]:
                # Log commands are put in the queue to be read by all relevant threads
                with lock:
                    shared_state["command_queue"].put("camera:log")  # General camera log
                    shared_state["command_queue"].put("robot:log") # Robot log is handled below

            # --- Handle Robot Movement Commands ---
            else:
                with lock:
                    # Prefix robot commands so the robot thread can filter them easily
                    shared_state["command_queue"].put("robot:" + command)

        except KeyboardInterrupt:
            with lock:
                shared_state["running"] = False
            break
        except Exception as e:
            print(f"[Command Writer Thread ERROR] {e}")
            time.sleep(1)

    print("[Command Writer Thread] Exiting.")


def robot_control_thread(controller: PandaController, shared_state: dict, lock: threading.Lock):
    print("\n[Robot Thread] Starting Interactive Control...")

    while shared_state["running"]:
        # Use a short timeout to prevent blocking the entire loop if the queue is empty
        command_prefix = None
        command_body = None

        try:
            # 1. Non-Blocking Command Retrieval
            with lock:
                # Use get_nowait() or a timeout to prevent blocking.
                # Since we use a loop and want to continue, get_nowait is simpler here.
                command = shared_state["command_queue"].get_nowait()

                # Split command into prefix and body
                if ":" in command:
                    command_prefix, command_body = command.split(":", 1)
                else:
                    # Handle direct signals like "quit_signal"
                    command_prefix = command
                    command_body = ""

        except queue.Empty:
            # If the queue is empty, do nothing and proceed with the rest of the loop
            pass
        except Exception as e:
            print(f"[Robot Thread ERROR] Error reading command queue: {e}")

        # 2. Process Command if found
        if command_prefix == "robot":
            move_command = command_body.strip()
            print(f"[Robot Thread] Received command: '{move_command}'")  # KEEP THIS FOR DEBUGGING

            # --- Stream current pose before executing command ---
            current_H = controller.robot.get_pose()
            with lock:
                # This is the "robot:log" action if implemented, currently merged with pose update
                shared_state["robot_pose_R"] = current_H

                # --- Execute Robot Movement ---
            controller._display_pose(current_H)
            updated_H, valid = controller.pos_command_to_H(move_command)

            if valid:
                try:
                    controller.robot.move_to_pose(updated_H)
                    print(f"[Robot Thread] Executed move: {move_command}")
                except Exception as e:
                    print(f"[Robot Thread ERROR] Failed to move robot: {e}")

        elif command_prefix == "quit_signal":
            # Graceful exit triggered by command writer
            shared_state["running"] = False
            break

        elif command_prefix is not None:
            print(f"[Robot Thread] Ignoring command prefix: {command_prefix}")

        # 3. Loop Control
        time.sleep(0.1)  # Short sleep to prevent high CPU usage

    print("[Robot Thread] Exiting.")

def vision_processing_thread(shared_state: dict, lock: threading.Lock, serial_num):
    """
    Processes camera frames and updates shared state (pose and image) without displaying.
    """
    thread_name = threading.current_thread().name
    print(f"\n[{thread_name}] Starting camera processing...")

    try:
        processor = AprilTagProcessor(
            serial=serial_num,
            world_tag_size=WORLD_TAG_SIZE,
            obj_tag_size=OBJ_TAG_SIZE,
            obj_tag_ids=OBJ_TAG_IDS,
        )
    except Exception as e:
        print(f"[Robot Thread ERROR] {e}")
    # ... (rest of error handling)

    # REMOVE cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    try:
        while shared_state["running"]:

            # 1. Process Frame
            vis_img, H0i_dict = processor.process_frame()

            # 2. Check for Log Command (MUST use try/except/put to pass non-camera commands)
            log_command = False
            try:
                # Get one command non-blocking
                with lock:
                    command = shared_state["command_queue"].get_nowait()

                if command == "camera:log":
                    log_command = True
                    # The command is consumed and processed, so it's not put back.
                elif command == "quit_signal":
                    shared_state["running"] = False
                    break
                else:
                    # PUT IT BACK for other threads (like the robot) to read!
                    with lock:
                        shared_state["command_queue"].put(command)

            except queue.Empty:
                pass  # No command in the queue
            except Exception as e:
                print(f"[{thread_name} ERROR] Error handling command: {e}")

            # 3. Update Shared State
            with lock:
                # Update Image always
                shared_state["vision_image"][serial_num] = vis_img

                # Log Pose only if the command was found
                if log_command:
                    print(f"[{thread_name}] Logging pose data.")  # For verification
                    # logger.log("[CAMERA] Logging Position")
                    shared_state["tag_pose_A"].update(H0i_dict)

            time.sleep(0.033)  # Faster sleep for consistent frame rate

    except Exception as e:
        print(f"[{thread_name} ERROR] {e}")
    finally:
        processor.release()
        print(f"[{thread_name}] Exiting.")


def run_concurrent_system(controller: PandaController):
    """Sets up and runs the dual-threaded robot and vision system."""

    # 1. Create Threads
    robot_t = threading.Thread(
        target=robot_control_thread,
        args=(controller, shared_state, state_lock),
        name="RobotControl")

    vision_t_1 = threading.Thread(
        target=vision_processing_thread,
        args=(shared_state, state_lock, "839112062097"),
        name="VisionProcessing_1"
    )

    vision_t_2 = threading.Thread(
        target=vision_processing_thread,
        args=(shared_state, state_lock, "845112070338"),
        name="VisionProcessing_2"
    )
    command_t = threading.Thread(
        target=command_writer_thread,
        args=(shared_state, state_lock),
        name="CommandWriter"
    )

    # 2. Start Threads
    robot_t.start()
    vision_t_1.start()
    vision_t_2.start()
    command_t.start()

    # 3. Main Loop: Monitor, Wait, and DISPLAY
    WIN = "Concurrent Vision Feed"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

    try:
        while shared_state["running"]:

            # --- 3a. Display Images from Threads (MUST be in main thread) ---
            with state_lock:
                images = shared_state["vision_image"]

            vis_list = [img for img in images.values() if img is not None]

            if vis_list:
                max_h = max(v.shape[0] for v in vis_list)
                vis_list_padded = [cv2.copyMakeBorder(v, 0, max_h - v.shape[0], 0, 0,
                                                      cv2.BORDER_CONSTANT, value=(0, 0, 0)) for v in vis_list]
                mosaic = cv2.hconcat(vis_list_padded)
                cv2.imshow(WIN, mosaic)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                shared_state["running"] = False


            # --- 3b. Data Check/Logging (Original Logic) ---
            with state_lock:
                tag_pose = shared_state["tag_pose_A"].get(1)
                robot_pose = shared_state["robot_pose_R"]

            if tag_pose is not None:
                # ... (rest of your logging/check logic)
                pass

            time.sleep(0.01)  # Use a very small sleep since display is now here

    except KeyboardInterrupt:
        print("\n[MAIN] Keyboard interrupt detected.")

    finally:
        # 4. Graceful Shutdown
        print("[MAIN] Shutting down concurrent processes...")
        shared_state["running"] = False

        # Wait for threads to finish cleanly
        robot_t.join()
        vision_t_1.join()
        vision_t_2.join()
        command_t.join()

        cv2.destroyAllWindows()  # Clean up all windows from the main thread

        ### log all the data into a csv to be done###
        print("[MAIN] System fully shut down.")