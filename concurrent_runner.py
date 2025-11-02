import threading
import time
import logging

import queue
import numpy as np
import cv2
from typing import Dict, Optional

# Import your clean components
from panda_controller import PandaController, ROBOT_IP
from april_tag_processor import (
    AprilTagProcessor, WORLD_TAG_ID, FRAME_W, FRAME_H,
    OBJ_TAG_IDS, WORLD_TAG_SIZE, OBJ_TAG_SIZE
)

# -----------------------------
# Config & Logging
# -----------------------------
CAMERA_SERIALS = ["839112062097", "845112070338"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s"
)

# -----------------------------
# Shared State (guarded by lock)
# -----------------------------
shared_state = {
    "tag_pose_A": {},                # dict[tag_id] -> 4x4 H
    "vision_image": {},              # dict[serial] -> np.ndarray (latest)
    "robot_pose_R": np.identity(4),  # 4x4 H
}
state_lock = threading.Lock()

# -----------------------------
# Command Queues
# -----------------------------
# Per-target queues prevent consumers from stealing each other's messages.
command_queues: Dict[str, queue.Queue] = {
    "robot": queue.Queue(maxsize=100),
    **{sn: queue.Queue(maxsize=100) for sn in CAMERA_SERIALS}
}

# -----------------------------
# Threads
# -----------------------------
def command_writer_thread(stop_event: threading.Event):
    """
    Reads stdin and enqueues commands to per-target queues.
    Commands:
        'q' | 'quit' | 'exit' -> stop
        'log' | 'l'           -> enqueue 'log' for all cameras + robot
        anything else         -> enqueues as robot command string
    """
    logging.info("Command writer started")
    while not stop_event.is_set():
        try:
            cmd = input("Type command (q=quit, l=log): ").strip()
            if not cmd:
                continue

            if cmd in ("q", "quit", "exit"):
                stop_event.set()
                break

            if cmd in ("l", "log"):
                # fan-out to all
                for sn in CAMERA_SERIALS:
                    try:
                        command_queues[sn].put_nowait(("log", None))
                    except queue.Full:
                        logging.warning(f"Queue full for camera {sn}; drop 'log'")
                try:
                    command_queues["robot"].put_nowait(("log", None))
                except queue.Full:
                    logging.warning("Queue full for robot; drop 'log'")
                continue

            # Everything else goes to the robot
            try:
                command_queues["robot"].put_nowait(("move", cmd))
            except queue.Full:
                logging.warning("Queue full for robot; drop 'move'")
        except KeyboardInterrupt:
            stop_event.set()
            break
        except Exception as e:
            logging.exception(f"Command writer error: {e}")
            time.sleep(0.5)

    logging.info("Command writer exiting")


def robot_control_thread(controller: PandaController, stop_event: threading.Event):
    """
    Consumes robot commands and updates shared robot pose.
    """
    logging.info("Robot control started")
    backoff = 0.1

    while not stop_event.is_set():
        try:
            # Always keep the latest pose in shared_state (best-effort)
            try:
                current_H = controller.robot.get_pose()
                with state_lock:
                    shared_state["robot_pose_R"] = current_H
            except Exception as e:
                logging.warning(f"get_pose failed: {e}")

            try:
                cmd_type, payload = command_queues["robot"].get(timeout=0.1)
            except queue.Empty:
                time.sleep(0.05)
                continue

            if cmd_type == "log":
                # Already stored current_H above
                logging.info("Robot logged current pose")
                continue

            if cmd_type == "move":
                raw = payload or ""
                # Delegate parse/execute to controller
                try:
                    controller._display_pose(current_H)
                except Exception as e:
                    logging.debug(f"_display_pose failed (non-fatal): {e}")

                try:
                    updated_H, valid = controller.pos_command_to_H(raw)
                except Exception as e:
                    logging.error(f"Failed to parse command '{raw}': {e}")
                    valid = False
                    updated_H = None

                if valid and updated_H is not None:
                    try:
                        controller.robot.move_to_pose(updated_H)
                        # Update shared pose after motion
                        current_H = controller.robot.get_pose()
                        with state_lock:
                            shared_state["robot_pose_R"] = current_H
                    except Exception as e:
                        logging.error(f"move_to_pose failed: {e}")
                else:
                    logging.warning(f"Ignored invalid robot command: {raw}")

            # small successful-iteration sleep
            time.sleep(0.05)
            backoff = 0.1  # reset backoff after success

        except KeyboardInterrupt:
            stop_event.set()
            break
        except Exception as e:
            logging.exception(f"Robot thread error: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 2.0)

    logging.info("Robot control exiting")


def vision_processing_thread(serial_num: str, stop_event: threading.Event):
    """
    Reads frames, updates latest image; consumes 'log' commands to snapshot poses.
    """
    logging.info(f"Vision processing started for {serial_num}")
    processor: Optional[AprilTagProcessor] = None

    try:
        try:
            processor = AprilTagProcessor(
                serial=serial_num,
                world_tag_size=WORLD_TAG_SIZE,
                obj_tag_size=OBJ_TAG_SIZE,
                obj_tag_ids=OBJ_TAG_IDS,
            )
        except Exception as e:
            logging.exception(f"Failed to init AprilTagProcessor[{serial_num}]: {e}")
            return  # Exit thread if camera cannot initialize

        backoff = 0.01
        while not stop_event.is_set():
            try:
                vis_img, H0i_dict = processor.process_frame()
                # Update latest image
                with state_lock:
                    shared_state["vision_image"][serial_num] = vis_img

                # Drain all pending commands for this camera (keep only the last effect)
                drained = []
                while True:
                    try:
                        drained.append(command_queues[serial_num].get_nowait())
                    except queue.Empty:
                        break

                for (cmd_type, payload) in drained:
                    if cmd_type == "log":
                        # Merge/overwrite tag poses
                        with state_lock:
                            shared_state["tag_pose_A"].update(H0i_dict)
                        logging.info(f"Camera {serial_num}: logged {len(H0i_dict)} tag poses")

                # Frame pacing
                time.sleep(0.01)
                backoff = 0.01

            except KeyboardInterrupt:
                stop_event.set()
                break
            except Exception as e:
                logging.warning(f"Vision[{serial_num}] frame error: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 0.2)  # cap backoff for vision

    finally:
        if processor is not None:
            try:
                processor.release()
            except Exception as e:
                logging.debug(f"Release failed for {serial_num}: {e}")
        logging.info(f"Vision processing exiting for {serial_num}")


# -----------------------------
# Main orchestration
# -----------------------------
def run_concurrent_system(controller: PandaController):
    stop_event = threading.Event()

    # Threads
    robot_t = threading.Thread(
        target=robot_control_thread,
        name="RobotControl",
        args=(controller, stop_event),
        daemon=True
    )
    vision_threads = [
        threading.Thread(
            target=vision_processing_thread,
            name=f"Vision_{sn}",
            args=(sn, stop_event),
            daemon=True
        )
        for sn in CAMERA_SERIALS
    ]
    command_t = threading.Thread(
        target=command_writer_thread,
        name="CommandWriter",
        args=(stop_event,),
        daemon=True
    )

    # Start
    robot_t.start()
    for t in vision_threads:
        t.start()
    command_t.start()

    # UI in main thread
    WIN = "Concurrent Vision Feed"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

    try:
        while not stop_event.is_set():
            # Gather latest images (non-blocking)
            with state_lock:
                images = [shared_state["vision_image"].get(sn) for sn in CAMERA_SERIALS]

            images = [img for img in images if img is not None]
            if images:
                max_h = max(img.shape[0] for img in images)
                padded = [
                    cv2.copyMakeBorder(
                        img, 0, max_h - img.shape[0], 0, 0,
                        cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    ) for img in images
                ]
                mosaic = cv2.hconcat(padded)
                cv2.imshow(WIN, mosaic)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                stop_event.set()
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt in main; stopping...")
        stop_event.set()
    except Exception as e:
        logging.exception(f"Main loop error: {e}")
        stop_event.set()
    finally:
        logging.info("Shutting down...")
        # Give threads a moment to observe stop_event
        for _ in range(50):
            if not any(t.is_alive() for t in ([robot_t] + vision_threads + [command_t])):
                break
            time.sleep(0.02)

        # Join politely
        robot_t.join(timeout=1.0)
        for t in vision_threads:
            t.join(timeout=1.0)
        command_t.join(timeout=1.0)

        cv2.destroyAllWindows()
        logging.info("System fully shut down.")


# -----------------------------
# Example entry (uncomment when integrating)
# -----------------------------
# if __name__ == "__main__":
#     ctrl = PandaController(ROBOT_IP)
#     run_concurrent_system(ctrl)
