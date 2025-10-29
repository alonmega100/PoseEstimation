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
    # Stored as a dict {tag_id: 4x4 H matrix}
    "tag_pose_A": {},

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


def vision_processing_thread(shared_state: dict, lock: threading.Lock):
    """
    Continuously processes camera frames and updates the calculated tag pose
    relative to the world anchor (^A T_T) in the shared data structure.
    """
    print("\n[Vision Thread] Starting camera processing...")

    # Initialize your processors (using simplified logic for one camera in this function)
    serial_num = "839112062097"  # Use the serial of the camera that sees the target object
    try:
        processor = AprilTagProcessor(
            serial=serial_num,
            world_tag_size=WORLD_TAG_SIZE,
            obj_tag_size=OBJ_TAG_SIZE,
            obj_tag_ids=OBJ_TAG_IDS,
        )
    except Exception as e:
        print(f"[Vision Thread] Failed to open camera {serial_num}: {e}")
        shared_state["running"] = False
        return

    try:
        while shared_state["running"]:

            # 1. Process Frame (gets visualization image and H0i poses)
            vis_img, H0i_dict = processor.process_frame()

            # 2. Update Shared State
            with lock:
                # H0i_dict contains the poses relative to the World Anchor (^A T_T)
                shared_state["tag_pose_A"] = H0i_dict

                # 3. Display (Optional but useful for debugging)
            cv2.imshow("Vision Feed", vis_img)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                shared_state["running"] = False

            # Simulate a small delay for stable loop timing
            time.sleep(0.033)  # Approx 30 FPS processing rate

    except Exception as e:
        print(f"[Vision Thread ERROR] {e}")
    finally:
        processor.release()
        cv2.destroyAllWindows()
        print("[Vision Thread] Exiting.")


def run_concurrent_system(controller: PandaController):
    """Sets up and runs the dual-threaded robot and vision system."""

    # 1. Create Threads
    robot_t = threading.Thread(
        target=robot_control_thread,
        args=(controller, shared_state, state_lock),
        name="RobotControl"
    )
    vision_t = threading.Thread(
        target=vision_processing_thread,
        args=(shared_state, state_lock),
        name="VisionProcessing"
    )

    # 2. Start Threads
    robot_t.start()
    vision_t.start()

    # 3. Main Loop: Monitor and wait for exit
    try:
        while shared_state["running"]:

            # --- OPTIONAL: Alignment Check/Logging ---
            with state_lock:
                tag_pose = shared_state["tag_pose_A"].get(1)  # Get pose for Tag ID 1
                robot_pose = shared_state["robot_pose_R"]

            if tag_pose is not None:
                # Here is where you would calculate the error or use the data:
                # Aligned_Prediction = CALIBRATION_H @ tag_pose
                # print(f"Tag Pose X: {tag_pose[0, 3]:.3f} | Robot X: {robot_pose[0, 3]:.3f}")
                pass

            time.sleep(0.1)  # Check state 10 times per second

    except KeyboardInterrupt:
        print("\n[MAIN] Keyboard interrupt detected.")

    finally:
        # 4. Graceful Shutdown
        print("[MAIN] Shutting down concurrent processes...")
        shared_state["running"] = False

        # Wait for threads to finish cleanly
        robot_t.join()
        vision_t.join()
        print("[MAIN] System fully shut down.")

# --- INTEGRATION STEP ---
# In your robot_cli.py, you would now import this run_concurrent_system
# and replace the old 'pos' command logic OR add a new 'concurrent' command.