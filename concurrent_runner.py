import threading
import time
import logging
import csv
import datetime
import queue
import numpy as np
import cv2
from typing import Dict, Optional

from panda_controller import PandaController, ROBOT_IP
from april_tag_processor import (
    AprilTagProcessor, WORLD_TAG_ID, FRAME_W, FRAME_H,
    OBJ_TAG_IDS, WORLD_TAG_SIZE, OBJ_TAG_SIZE
)
import json
from tools import matrix_to_flat_dict, is_4x4_matrix

POSE_COLS = [f"pose_{r}{c}" for r in range(4) for c in range(4)]

# -----------------------------
# Config
# -----------------------------
CAMERA_SERIALS = ["839112062097", "845112070338"]
TARGET_LOG_HZ = 30.0
LOG_INTERVAL = 1.0 / TARGET_LOG_HZ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s"
)

# -----------------------------
# Shared state (global-ish)
# -----------------------------
shared_state = {
    "tag_pose_A": {},
    "vision_image": {},
    "robot_pose_R": np.identity(4),
}
state_lock = threading.Lock()

# command queues
command_queues: Dict[str, queue.Queue] = {
    "robot": queue.Queue(maxsize=100),
    **{sn: queue.Queue(maxsize=100) for sn in CAMERA_SERIALS}
}


# -----------------------------
# Threads
# -----------------------------
def command_writer_thread(stop_event: threading.Event):
    """
    Reads stdin and enqueues commands.
    (No 'd' anymore, because discard is decided beforehand.)
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
                # user-triggered one-shot log
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

            # everything else goes to the robot as move
            try:
                command_queues["robot"].put_nowait(("move", cmd))
            except queue.Full:
                logging.warning("Queue full for robot; drop 'move'")

        except KeyboardInterrupt:
            stop_event.set()
            break
        except Exception as e:
            logging.exception(f"Command writer error: {e}")
            time.sleep(0.01)

    logging.info("Command writer exiting")


def robot_control_thread(
    controller: PandaController,
    stop_event: threading.Event,
    log_event,              # function
    discard: bool
):
    logging.info("Robot control started")
    while not stop_event.is_set():
        try:
            got_log_request = False
            move_cmds = []

            # drain queue
            while True:
                try:
                    cmd_type, payload = command_queues["robot"].get_nowait()
                except queue.Empty:
                    break

                if cmd_type == "log":
                    got_log_request = True
                elif cmd_type == "move":
                    move_cmds.append(payload)

            # execute moves
            for raw in move_cmds:
                raw = raw or ""
                try:
                    updated_H, valid = controller.pos_command_to_H(raw)
                except Exception as e:
                    logging.error(f"Failed to parse command '{raw}': {e}")
                    valid = False
                    updated_H = None

                if valid and updated_H is not None:
                    try:
                        controller.robot.move_to_pose(updated_H)
                    except Exception as e:
                        logging.error(f"move_to_pose failed: {e}")

            # log robot pose on request (only if not discarding)
            if got_log_request and not discard:
                try:
                    current_H = controller.robot.get_pose()
                    with state_lock:
                        shared_state["robot_pose_R"] = current_H
                    log_event("robot", "pose_snapshot", {"pose": current_H.tolist()})
                    logging.debug("Robot logged current pose")
                except Exception as e:
                    logging.warning(f"get_pose failed: {e}")

            if not got_log_request and not move_cmds:
                time.sleep(0.001)

        except KeyboardInterrupt:
            stop_event.set()
            break
        except Exception as e:
            logging.exception(f"Robot thread error: {e}")
            time.sleep(0.005)

    logging.info("Robot control exiting")


def vision_processing_thread(
    serial_num: str,
    stop_event: threading.Event,
    log_event,              # function
    discard: bool
):
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
            return

        backoff = 0.01
        while not stop_event.is_set():
            try:
                vis_img, H0i_dict = processor.process_frame()
                # update latest image
                with state_lock:
                    shared_state["vision_image"][serial_num] = vis_img

                # --- normalize tag poses to world (from this camera) ---
                # processor.process_frame() is already giving you tag poses relative to WORLD_TAG
                # BUT we still guard here in case world tag is missing in this frame.
                if WORLD_TAG_ID in H0i_dict:
                    # in your processor, H0i_dict[tag] = H_world_tag already, but let's keep the shape
                    # if later you change the processor to return camera poses, this block is still useful
                    normalized_dict = {}
                    for tag_id, H_world_tag in H0i_dict.items():
                        # optional flip check in WORLD frame:
                        z_world = H_world_tag[:3, 2]
                        if z_world[2] < 0:  # tag is upside-down relative to world Z
                            H_world_tag = H_world_tag.copy()
                            H_world_tag[:3, :3] = H_world_tag[:3, :3] @ np.diag([1, -1, -1])
                        normalized_dict[tag_id] = H_world_tag
                    H0i_dict = normalized_dict
                # else: leave H0i_dict as-is (no world tag this frame)

                # handle log commands for this camera
                drained = []
                while True:
                    try:
                        drained.append(command_queues[serial_num].get_nowait())
                    except queue.Empty:
                        break

                for (cmd_type, payload) in drained:
                    if cmd_type == "log" and not discard:
                        log_event(serial_num, "tag_pose_snapshot", {"pose": H0i_dict})
                        with state_lock:
                            # store the world-aligned, flip-corrected poses
                            shared_state["tag_pose_A"].update(H0i_dict)
                        logging.debug(f"Camera {serial_num}: logged {len(H0i_dict)} tag poses")

                time.sleep(0.01)
                backoff = 0.01

            except KeyboardInterrupt:
                stop_event.set()
                break
            except Exception as e:
                logging.warning(f"Vision[{serial_num}] frame error: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 0.2)

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
def run_concurrent_system(controller: PandaController, discard: bool = False):
    """
    discard=False  -> normal: logs are collected and saved
    discard=True   -> no periodic log commands, logger becomes no-op, and nothing is saved
    """
    stop_event = threading.Event()

    # per-run log buffer (not global anymore)
    run_logs = []
    log_lock = threading.Lock()

    # closure that respects discard
    def log_event(source: str, event: str, data: dict = None):
        if discard:
            return
        entry = {
            "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "source": source,
            "event": event,
            "data": data or {},
        }
        with log_lock:
            run_logs.append(entry)

    # Threads
    robot_t = threading.Thread(
        target=robot_control_thread,
        name="RobotControl",
        args=(controller, stop_event, log_event, discard),
        daemon=True
    )
    vision_threads = [
        threading.Thread(
            target=vision_processing_thread,
            name=f"Vision_{sn}",
            args=(sn, stop_event, log_event, discard),
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

    # UI / main loop
    WIN = "Concurrent Vision Feed"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    next_log_time = time.perf_counter()

    try:
        while not stop_event.is_set():
            now = time.perf_counter()

            # periodic logging only if not discarding
            if not discard and now >= next_log_time:
                while next_log_time <= now:
                    next_log_time += LOG_INTERVAL

                # record the request
                log_event("command", "log_request", {"targets": ["robot"] + CAMERA_SERIALS})

                # fan out to threads
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

            # show images
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
        # wait for threads
        for _ in range(50):
            if not any(t.is_alive() for t in ([robot_t] + vision_threads + [command_t])):
                break
            time.sleep(0.01)

        robot_t.join(timeout=1.0)
        for t in vision_threads:
            t.join(timeout=1.0)
        command_t.join(timeout=1.0)

        # save only if not discarding
        if not discard:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"CSV/session_log_{timestamp}.csv"

            rows_to_write = []
            with log_lock:
                for entry in run_logs:
                    ts = entry.get("timestamp")
                    src = entry.get("source")
                    ev = entry.get("event")
                    data = entry.get("data", {})

                    # 1) data = {"pose": 4x4}
                    if isinstance(data, dict) and "pose" in data and is_4x4_matrix(data["pose"]):
                        flat_pose = matrix_to_flat_dict("pose", data["pose"])
                        rows_to_write.append({
                            "timestamp": ts,
                            "source": src,
                            "event": ev,
                            **flat_pose
                        })
                        continue

                    # 2) data = {"pose": {tag_id: 4x4 or np.array, ...}}
                    if isinstance(data, dict) and "pose" in data and isinstance(data["pose"], dict):
                        for tag_id, mat in data["pose"].items():
                            if is_4x4_matrix(mat):
                                flat_pose = matrix_to_flat_dict("pose", mat)
                                rows_to_write.append({
                                    "timestamp": ts,
                                    "source": src,
                                    "event": ev,
                                    "tag_id": str(tag_id),
                                    **flat_pose
                                })
                            else:
                                if isinstance(mat, np.ndarray):
                                    mat = mat.tolist()
                                rows_to_write.append({
                                    "timestamp": ts,
                                    "source": src,
                                    "event": ev,
                                    "tag_id": str(tag_id),
                                    "raw_data": json.dumps(mat)
                                })
                        continue

                    # 3) everything else
                    safe_data = data
                    if isinstance(safe_data, np.ndarray):
                        safe_data = safe_data.tolist()
                    rows_to_write.append({
                        "timestamp": ts,
                        "source": src,
                        "event": ev,
                        "raw_data": json.dumps(safe_data) if safe_data else ""
                    })

            fieldnames = ["timestamp", "source", "event", "tag_id", "raw_data"] + POSE_COLS
            try:
                with open(log_filename, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows_to_write:
                        writer.writerow(row)
                logging.info(f"Saved {len(rows_to_write)} log rows to {log_filename}")
            except Exception as e:
                logging.error(f"Failed to save run logs: {e}")

        cv2.destroyAllWindows()
        logging.info("System fully shut down.")
