import threading
import time
import sys, select
import logging
import csv
import queue
import numpy as np
import cv2
from typing import Dict, Optional
import os
import datetime
import json

from src.robot.panda_controller import PandaController
from src.vision.april_tag_processor import AprilTagProcessor
from src.vision.vision_display import VisionDisplay
from src.utils.tools import matrix_to_flat_dict, is_4x4_matrix, list_of_movements_generator, make_serializable
from src.utils.hdf5_writer import HDF5Writer
from src.utils.config import OBJ_TAG_IDS, CAMERA_SERIALS, NUM_OF_COMMANDS_TO_GENERATE
from src.imu.imu_reader import IMUReader

# -------------------------------------------------
# Config
# -------------------------------------------------

TARGET_LOG_HZ = 30.0  # robot logger rate
LOG_INTERVAL = 1.0 / TARGET_LOG_HZ

POSE_COLS = [f"pose_{r}{c}" for r in range(4) for c in range(4)]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s"
)

# -------------------------------------------------
# Shared state
# -------------------------------------------------
shared_state = {
    "tag_pose_A": {},
    "vision_image": {},
    "tag_pose_by_cam": {},
    "robot_pose_R": np.identity(4),
}
state_lock = threading.Lock()

# command queues
command_queues: Dict[str, queue.Queue] = {
    "robot": queue.Queue(maxsize=100),
    **{sn: queue.Queue(maxsize=100) for sn in CAMERA_SERIALS}
}


# -------------------------------------------------
# Command thread (stdin)
# -------------------------------------------------

def command_writer_thread(
        stop_event: threading.Event,
        no_more_commands_event: threading.Event,
):
    logging.info("Command writer started")
    list_of_movements = list_of_movements_generator(NUM_OF_COMMANDS_TO_GENERATE)
    # list_of_movements = ['y 0.2 z -0.1', 'x 0.1 y -0.2'] # Override for repeating experiment

    print(list_of_movements)
    while not stop_event.is_set():
        try:
            print("Type command (q=quit): ")
            i, o, e = select.select([sys.stdin], [], [], 2)

            if i:
                cmd = sys.stdin.readline().strip()
            else:
                try:
                    cmd = list_of_movements.pop(0).strip()
                except IndexError:
                    print("List of commands is empty. No more commands will be sent.")
                    no_more_commands_event.set()
                    break

            print("Command received: {}".format(cmd))

            if not cmd:
                continue

            if cmd in ("q", "quit", "exit"):
                stop_event.set()
                break

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


# -------------------------------------------------
# Robot MOVE thread (blocking moves only)
# -------------------------------------------------
def robot_move_thread(
        controller: PandaController,
        stop_event: threading.Event,
        no_more_commands_event: threading.Event,
):
    logging.info("Robot MOVE thread started")

    updated_H, valid = controller.pos_command_to_H("yaw 20 -z 0.15")
    controller.robot.move_to_pose(updated_H, speed_factor=controller.speed_factor)

    while not stop_event.is_set():
        try:
            move_cmds = []
            while True:
                try:
                    cmd_type, payload = command_queues["robot"].get_nowait()
                except queue.Empty:
                    break

                if cmd_type == "move":
                    move_cmds.append(payload)

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
                        controller.robot.move_to_pose(updated_H, speed_factor=controller.speed_factor)
                    except Exception as e:
                        logging.error(f"move_to_pose failed: {e}")

            if not move_cmds:
                time.sleep(0.001)

            if no_more_commands_event.is_set() and command_queues["robot"].empty():
                logging.info("Robot MOVE thread: finished all commands, signaling global stop.")
                stop_event.set()
                break
        except KeyboardInterrupt:
            stop_event.set()
            break
        except Exception as e:
            logging.exception(f"Robot MOVE thread error: {e}")
            time.sleep(0.005)

    logging.info("Robot MOVE thread exiting")


# -------------------------------------------------
# Robot LOGGER thread
# -------------------------------------------------
def robot_logger_thread(
        controller: PandaController,
        stop_event: threading.Event,
        log_event,
        discard: bool,
        writer: Optional["HDF5Writer"],
        hz: float = 30.0,
):
    logging.info("Robot LOGGER thread started")
    period = 1.0 / hz

    while not stop_event.is_set():
        start_t = time.time()
        try:
            if hasattr(controller, "get_state_raw"):
                state = controller.get_state_raw()
            else:
                state = controller.robot.get_state()

            H = controller.robot.get_pose()
            t = time.time()
            if writer is not None:
                try:
                    writer.add_robot_data(state.q, state.dq, np.zeros(6), state.tau_J, H, t)
                except Exception as e:
                    logging.error(f"Robot logger: failed to add robot data: {e}")

            if not discard:
                def to_list(x):
                    if isinstance(x, np.ndarray): return x.tolist()
                    if hasattr(x, "tolist"): return x.tolist()
                    return list(x)

                log_event(
                    "robot",
                    "pose_snapshot",
                    {
                        "pose": H.tolist(),
                        "q": to_list(state.q),
                        "dq": to_list(state.dq),
                        "tau_J": to_list(state.tau_J),
                    },
                )

            with state_lock:
                shared_state["robot_pose_R"] = H

        except Exception as e:
            logging.warning(f"Robot logger: read failed: {e}")

        dt = time.time() - start_t
        if dt < period:
            time.sleep(period - dt)

    logging.info("Robot LOGGER thread exiting")


# -------------------------------------------------
# Vision (AprilTag) threads
# -------------------------------------------------
def vision_processing_thread(
        serial_num: str,
        stop_event: threading.Event,
        log_event,
        discard: bool
):
    logging.info(f"Vision processing started for {serial_num}")
    processor: Optional[AprilTagProcessor] = None

    try:
        try:
            processor = AprilTagProcessor(serial_num)
        except Exception as e:
            logging.exception(f"Failed to init AprilTagProcessor[{serial_num}]: {e}")
            return

        backoff = 0.01
        while not stop_event.is_set():
            try:
                vis_img, Hci_dict = processor.process_frame()
                with state_lock:
                    shared_state["vision_image"][serial_num] = vis_img
                    shared_state["tag_pose_by_cam"][serial_num] = Hci_dict

                if not discard:
                    log_event(serial_num, "tag_pose_snapshot", {"pose": Hci_dict})
                    with state_lock:
                        shared_state["tag_pose_A"].update(Hci_dict)

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
            except Exception:
                pass
        logging.info(f"Vision processing exiting for {serial_num}")


# -------------------------------------------------
# IMU thread (CLEARED: Raw logging only)
# -------------------------------------------------
def imu_thread(
        stop_event: threading.Event,
        log_event,
        discard: bool,
        imu_reader: IMUReader,
):
    """
    Poll IMUReader.get_latest() and log samples.
    No drift correction, no position snapping, no gravity cleaning logic here.
    """
    if imu_reader is None:
        logging.warning("IMU thread: imu_reader is None, exiting immediately")
        return

    logging.info("IMU thread started (Logging Yaw, Pitch, Roll, Accel)")
    backoff = 0.01

    try:
        while not stop_event.is_set():
            try:
                # Log latest IMU sample for CSV / HDF5
                sample = imu_reader.get_latest()
                if sample is not None and not discard:
                    log_event("imu", "imu_sample", sample)

                time.sleep(0.01)
                backoff = 0.01
            except KeyboardInterrupt:
                stop_event.set()
                break
            except Exception as e:
                logging.warning(f"IMU thread error: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 0.5)
    finally:
        logging.info("IMU thread exiting")


# -------------------------------------------------
# Main orchestrator
# -------------------------------------------------
def run_concurrent_system(controller: PandaController, discard: bool = False):
    stop_event = threading.Event()
    no_more_commands_event = threading.Event()

    # --- IMU setup ---
    imu_reader = None
    imu_t = None
    try:
        imu_reader = IMUReader(rate_hz=50.0)
        imu_reader.start()
    except Exception as e:
        logging.warning(f"Failed to start IMUReader: {e}")

    os.makedirs("../../data/DATA", exist_ok=True)
    os.makedirs("../../data/CSV", exist_ok=True)
    writer = None
    if not discard:
        writer = HDF5Writer("../../data/DATA/session.h5", "session")
        writer.start()
        run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        writer.add_run(run_id, force=0, dx=0, dy=0, angle=0)
        writer.start_writing()
        writer.to_file_enabled = True

    run_logs = []
    log_lock = threading.Lock()

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

    # start robot threads
    robot_move_t = threading.Thread(
        target=robot_move_thread,
        name="RobotMove",
        args=(controller, stop_event, no_more_commands_event),
        daemon=True,
    )
    robot_log_t = threading.Thread(
        target=robot_logger_thread,
        name="RobotLogger",
        args=(controller, stop_event, log_event, discard, writer, TARGET_LOG_HZ),
        daemon=True,
    )

    # vision threads
    vision_threads = [
        threading.Thread(
            target=vision_processing_thread,
            name=f"Vision_{sn}",
            args=(sn, stop_event, log_event, discard),
            daemon=True
        )
        for sn in CAMERA_SERIALS
    ]

    # command thread
    command_t = threading.Thread(
        target=command_writer_thread,
        name="CommandWriter",
        args=(stop_event, no_more_commands_event),
        daemon=True
    )

    # start all
    robot_move_t.start()
    robot_log_t.start()

    # IMU thread - Updated signature (no correction_interval)
    if imu_reader is not None:
        imu_t = threading.Thread(
            target=imu_thread,
            name="IMU",
            args=(stop_event, log_event, discard, imu_reader),
            daemon=True,
        )
        imu_t.start()

    for t in vision_threads:
        t.start()
    command_t.start()

    WIN = "Concurrent Vision Feed"
    display = VisionDisplay(window_title=WIN)

    try:
        while not stop_event.is_set():
            with state_lock:
                images = {sn: shared_state["vision_image"].get(sn) for sn in CAMERA_SERIALS}
                tag_by_cam = {sn: shared_state.get("tag_pose_by_cam", {}).get(sn, {}) for sn in CAMERA_SERIALS}

            for sn, img in images.items():
                if img is not None:
                    tags = tag_by_cam.get(sn) or {}
                    overlay = [f"tags: {len(tags)}"]
                    display.update_frame(sn, img, overlay_text=overlay)

            if not display.show_mosaic():
                stop_event.set()
                break

            time.sleep(0.01)

    finally:
        logging.info("Shutting down...")
        stop_event.set()

        if display:
            display.cleanup()

        robot_move_t.join(timeout=2.0)
        robot_log_t.join(timeout=2.0)

        for t in vision_threads:
            t.join(timeout=5.0)

        command_t.join(timeout=1.0)

        if imu_t is not None:
            imu_t.join(timeout=2.0)
        if imu_reader is not None:
            try:
                imu_reader.stop()
            except Exception as e:
                logging.error(f"Failed to stop IMUReader: {e}")

        # write CSV
        if not discard:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"data/CSV/session_log_{timestamp}.csv"

            rows_to_write = []
            with log_lock:
                for entry in run_logs:
                    ts = entry.get("timestamp")
                    src = entry.get("source")
                    ev = entry.get("event")
                    data = entry.get("data", {})

                    if src == "robot" and isinstance(data, dict) and "pose" in data:
                        flat_pose = matrix_to_flat_dict("pose", data["pose"])
                        extras = data.copy()
                        extras.pop("pose", None)
                        rows_to_write.append({
                            "timestamp": ts,
                            "source": src,
                            "event": ev,
                            "raw_data": json.dumps(extras),
                            **flat_pose
                        })
                        continue

                    if isinstance(data, dict) and "pose" in data and isinstance(data["pose"], dict):
                        for tag_id, mat in data["pose"].items():
                            mat = make_serializable(mat)
                            mat = mat[0]
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
                                print(f"[WARNING] Unusable data {mat}")
                        continue

                    # IMU samples -> LOG ORIENTATION AND ACCELERATION ONLY
                    if src == "imu" and isinstance(data, dict):
                        yaw = data.get("yaw")
                        pitch = data.get("pitch")
                        roll = data.get("roll")

                        # Trying to find acceleration.
                        # Assumes key is 'accel'. Adjust this if your dict uses 'accel_m_s2' or similar.
                        acc = data.get("accel")
                        if acc is None:
                            # Fallback: try separate keys if tuple not found
                            acc = (data.get("acc_x"), data.get("acc_y"), data.get("acc_z"))

                        rows_to_write.append({
                            "timestamp": ts,
                            "source": src,
                            "event": ev,
                            "raw_data": json.dumps(data),
                            "yaw": float(yaw) if yaw is not None else "",
                            "pitch": float(pitch) if pitch is not None else "",
                            "roll": float(roll) if roll is not None else "",
                            "acc_x": float(acc[0]) if acc and acc[0] is not None else "",
                            "acc_y": float(acc[1]) if acc and acc[1] is not None else "",
                            "acc_z": float(acc[2]) if acc and acc[2] is not None else "",
                        })
                    else:
                        safe_data = data
                        if isinstance(safe_data, np.ndarray):
                            safe_data = safe_data.tolist()
                        rows_to_write.append({
                            "timestamp": ts,
                            "source": src,
                            "event": ev,
                            "raw_data": json.dumps(safe_data) if safe_data else ""
                        })

            # UPDATED IMU COLUMNS

            fieldnames = ["timestamp", "source", "event", "tag_id", "raw_data","yaw", "pitch", "roll", "acc_x", "acc_y", "acc_z"] + POSE_COLS
            try:
                with open(log_filename, "w", newline="") as f:
                    csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
                    csv_writer.writeheader()
                    for row in rows_to_write:
                        csv_writer.writerow(row)
                logging.info(f"Saved {len(rows_to_write)} log rows to {log_filename}")
            except Exception as e:
                logging.error(f"Failed to save run logs: {e}")

        if writer is not None:
            try:
                writer.stop()
            except Exception as e:
                logging.error(f"Failed to stop HDF5 writer: {e}")

        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception as e:
            logging.warning(f"OpenCV cleanup failed (non-fatal): {e}")