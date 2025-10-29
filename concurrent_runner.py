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

    # System control flag
    "running": True
}
# Lock to ensure thread-safe access to shared_state
state_lock = threading.Lock()

# --- 2. THREAD FUNCTIONS ---

def robot_control_thread(controller: PandaController, shared_state: dict, lock: threading.Lock):
    """
    Runs the interactive robot control loop and updates the latest robot pose
    to the shared data structure after every move.
    """
    print("\n[Robot Thread] Starting Interactive Control...")

    # Adapted logic from your old position_control_loop
    # NOTE: Display/Print inside threads can be messy, but we keep it for now.

    while shared_state["running"]:
        try:
            # --- Stream current pose for Vision Thread ---
            current_H = controller.robot.get_pose()
            with lock:
                shared_state["robot_pose_R"] = current_H

            # --- Interactive User Input ---
            controller._display_pose(current_H)

            # Input handling must be non-blocking or managed differently in a true thread.
            # For simplicity in this demo, we'll use a blocking input,
            # but in a real-time system, this thread should use a non-blocking queue for commands.

            print("Input Format: <Axis> <Delta> (Type 'q' to quit thread)")
            user_input = input("Enter command: ").lower().strip()

            if user_input in ["q", "quit"]:
                shared_state["running"] = False
                break


            updated_H, valid = controller.pos_command_to_H(user_input)
            if valid:
                controller.robot.move_to_pose(updated_H)
            # (PASTE YOUR EXISTING INPUT PARSING AND MOVEMENT LOGIC HERE,
            #  which uses the controller methods: _calculate_new_pose_..., move_to_pose)

            # Placeholder for movement logic:
            # if move_is_successful:
            #     controller.robot.move_to_pose(updated_H, speed_factor=controller.speed_factor)

            time.sleep(0.5)  # Prevent high CPU usage when waiting for input

        except KeyboardInterrupt:
            shared_state["running"] = False
            break
        except Exception as e:
            print(f"[Robot Thread ERROR] {e}")
            time.sleep(1)

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

            # 2. Update Shared State
            with lock:
                # Update Poses
                shared_state["tag_pose_A"].update(H0i_dict)
                # Update Image
                shared_state["vision_image"][serial_num] = vis_img



            time.sleep(0.033) # Faster sleep for consistent frame rate

    except Exception as e:
        print(f"[Robot Thread ERROR] {e}")
    finally:
        processor.release()
        # REMOVE cv2.destroyWindow(WINDOW_NAME)
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
        args=(shared_state, state_lock, "839112062097"),  # **Now a string**
        name="VisionProcessing_1"  # Changed name for clarity
    )

    vision_t_2 = threading.Thread(
        target=vision_processing_thread,
        args=(shared_state, state_lock, "845112070338"),  # **Now a string**
        name="VisionProcessing_2"  # Changed name for clarity
    )

    # 2. Start Threads
    robot_t.start()
    vision_t_1.start()
    vision_t_2.start()
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
                # Mosaic (combine) the images horizontally
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

        cv2.destroyAllWindows()  # Clean up all windows from the main thread
        print("[MAIN] System fully shut down.")