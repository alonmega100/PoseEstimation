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

from src.robot.panda_controller import PandaController, DEFAULT_SPEED_FACTOR
from src.vision.april_tag_processor import AprilTagProcessor
from src.utils.tools import matrix_to_flat_dict, is_4x4_matrix, list_of_movements_generator
from src.utils.hdf5_writer import HDF5Writer
from src.utils.config import OBJ_TAG_IDS, WORLD_TAG_SIZE, OBJ_TAG_SIZE, CAMERA_SERIALS
from src.imu.imu_reader import IMUReader


# -------------------------------------------------
# Config
# -------------------------------------------------


IMU_CORRECTION_INTERVAL = 3.0   # seconds; how often to "snap" IMU to camera tag
IMU_CORRECTION_ZERO_VEL = True  # zero IMU velocity at correction
IMU_CORRECTION_TAG_ID = 2  # which tag to use as the ground-truth position

TARGET_LOG_HZ = 30.0     # robot logger rate
LOG_INTERVAL = 1.0 / TARGET_LOG_HZ
NUM_OF_COMMANDS_TO_GENERATE = 20
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
    ###### Using the following line to override the command queue to get a reapeating experiment  ######
    # list_of_movements = ['y 0.2 z -0.1', 'x 0.1 y -0.2', 'y 0.2', 'x -0.1', 'x 0.1', 'y -0.2', 'x -0.1', 'x 0.1', 'y 0.2', 'y -0.1', 'x -0.2', 'y -0.1', 'y 0.2', 'x 0.1 y -0.2', 'x 0.1 y 0.1', 'z 0.1', 'y 0.1 z -0.1', 'y -0.1 z 0.1', 'x -0.1', 'x -0.1 y -0.1']
    ######   ######
    print(list_of_movements)
    # list_of_movements = ["yaw 30 -z 0.05", "y 0.1 x 0.1"]
    while not stop_event.is_set():
        try:
            print("Type command (q=quit): ")

            i, o, e = select.select([sys.stdin], [], [], 2)

            if i:
                # User typed something
                cmd = sys.stdin.readline().strip()
            else:
                # Auto mode: take from pre-generated list
                try:
                    cmd = list_of_movements.pop(0).strip()
                except IndexError:
                    # No more commands to send -> tell the robot thread
                    print("List of commands is empty. No more commands will be sent.")
                    no_more_commands_event.set()
                    # We don't set stop_event here; robot thread will stop
                    # once it has finished all queued moves.
                    break

            print("Command received: {}".format(cmd))

            if not cmd:
                continue

            # Manual abort (user typed q/quit/exit)
            if cmd in ("q", "quit", "exit"):
                stop_event.set()
                break


            # everything else goes to the robot as move
            try:
                print("putting the toopel", cmd)

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
#    --- Manual Initialization ---

    updated_H, valid = controller.pos_command_to_H("yaw 20 -z 0.15")

    controller.robot.move_to_pose(updated_H)

#  --- End of Initialization ---
    while not stop_event.is_set():
        try:
            move_cmds = []
            # drain robot queue
            while True:
                try:
                    cmd_type, payload = command_queues["robot"].get_nowait()
                    print("got this gem:", cmd_type, payload)
                except queue.Empty:
                    break

                if cmd_type == "move":
                    move_cmds.append(payload)

            # run moves (may block)
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
                        controller.robot.move_to_pose(updated_H, speed_factor=DEFAULT_SPEED_FACTOR)
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
# Robot LOGGER thread (runs at fixed rate, dumps everything)
# -------------------------------------------------
def robot_logger_thread(
    controller: PandaController,
    stop_event: threading.Event,
    log_event,
    discard: bool,
    writer: Optional["HDF5Writer"],  # may be None when discard=True
    hz: float = 30.0,
):
    logging.info("Robot LOGGER thread started")
    period = 1.0 / hz

    while not stop_event.is_set():
        start_t = time.time()
        try:
            # grab robot state regardless of motion
            if hasattr(controller, "get_state_raw"):
                state = controller.get_state_raw()
            else:
                state = controller.robot.get_state()

            H = controller.robot.get_pose()
            t = time.time()
            if writer is not None:
            # ---- 1) push to HDF5
                try:
                    writer.add_robot_data(
                        state.q,
                        state.dq,
                        np.zeros(6),
                        state.tau_J,
                        H,
                        t
                    )
                except Exception as e:
                    logging.error(f"Robot logger: failed to add robot data: {e}")

            # ---- 2) push to CSV (make everything jsonable)
            if not discard:
                def to_list(x):
                    if isinstance(x, np.ndarray):
                        return x.tolist()
                    if hasattr(x, "tolist"):
                        return x.tolist()
                    return list(x)  # assume iterable

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

            # update shared pose
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
#   log every frame
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
                with state_lock:
                    shared_state["vision_image"][serial_num] = vis_img


                # log this frame’s tag poses (even if identical)
                if not discard:
                    log_event(serial_num, "tag_pose_snapshot", {"pose": H0i_dict})
                    with state_lock:
                        shared_state["tag_pose_A"].update(H0i_dict)

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

def imu_thread(
    stop_event: threading.Event,
    log_event,
    discard: bool,
    imu_reader: IMUReader,
    correction_interval: float = IMU_CORRECTION_INTERVAL,
):
    """
    Poll IMUReader.get_latest(), log samples, and periodically snap the
    integrated IMU position to the current OBJECT TAG pose estimated by
    the cameras.

    The world tag still defines the global origin for the cameras, but
    the IMU "follows" the object tag: every `correction_interval` seconds
    we take the latest object-tag pose from `shared_state["tag_pose_A"]`
    and set the IMU position to that translation.
    """
    # Defensive check: if imu_reader is None, exit gracefully
    if imu_reader is None:
        logging.warning("IMU thread: imu_reader is None, exiting immediately")
        return
    
    logging.info("IMU thread started")
    backoff = 0.01
    last_correction_wall = time.time()

    # helper to fetch the latest object-tag pose from shared_state
    def _get_object_tag_position():
        with state_lock:
            tag_dict = dict(shared_state.get("tag_pose_A", {}))

        if not tag_dict:
            return None

        # OBJ_TAG_IDS may be a list/tuple or a single id
        if isinstance(OBJ_TAG_IDS, (list, tuple, set)):
            candidate_ids = list(OBJ_TAG_IDS)
        else:
            candidate_ids = [OBJ_TAG_IDS]

        H_obj = None
        for cid in candidate_ids:
            # try both raw key and stringified key
            if cid in tag_dict:
                H_obj = tag_dict[cid]
                break
            scid = str(cid)
            if scid in tag_dict:
                H_obj = tag_dict[scid]
                break

        if H_obj is None or not is_4x4_matrix(H_obj):
            return None

        H = np.array(H_obj, dtype=float)
        return H[:3, 3]  # (x,y,z)

    try:
        while not stop_event.is_set():
            try:
                # 1) Log latest IMU sample for CSV / HDF5
                sample = imu_reader.get_latest()
                if sample is not None and not discard:
                    log_event("imu", "imu_sample", sample)

                # 2) Periodic drift correction using object tag pose
                if correction_interval is not None and correction_interval > 0.0:
                    now = time.time()
                    if now - last_correction_wall >= correction_interval:
                        pos_obj = _get_object_tag_position()
                        if pos_obj is not None:
                            # Optionally keep current velocity, or zero it.
                            if IMU_CORRECTION_ZERO_VEL:
                                vel = (0.0, 0.0, 0.0)
                            else:
                                vel = None
                                if sample is not None:
                                    vel = sample.get("vel_m_s")
                                if vel is None:
                                    vel = (0.0, 0.0, 0.0)

                            imu_reader.set_position(pos_obj, vel)
                            logging.debug(f"IMU correction: snapping to object tag at {pos_obj}")
                            last_correction_wall = now

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

    # make dirs
    os.makedirs("../../data/DATA", exist_ok=True)
    os.makedirs("../../data/CSV", exist_ok=True)
    writer = None
    if not discard:

        # HDF5 writer
        writer = HDF5Writer("../../data/DATA/session.h5", "session")
        writer.start()
        run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        writer.add_run(run_id, force=0, dx=0, dy=0, angle=0)

        # you said: "if a vision thread is running -> we get a camera log every frame"
        # so we just keep writer on
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

    # vision threads (now log every frame)
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
    # IMU thread (only if imu_reader was successfully initialized)
    if imu_reader is not None:
        imu_t = threading.Thread(
            target=imu_thread,
            name="IMU",
            args=(stop_event, log_event, discard, imu_reader, IMU_CORRECTION_INTERVAL),
            daemon=True,
        )
        imu_t.start()
    
    for t in vision_threads:
        t.start()
    command_t.start()

    WIN = "Concurrent Vision Feed"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

    try:
        while not stop_event.is_set():
            # show combined camera view
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

    finally:
        logging.info("Shutting down...")
        stop_event.set()
        # STEP 1: Wait for robot threads (usually fast)
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

                    # robot rows with full state
                    if src == "robot" and isinstance(data, dict) and "pose" in data:
                        flat_pose = matrix_to_flat_dict("pose", data["pose"])
                        # dump the rest to raw_data
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

                    # camera: multi-tag dict
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

                    # single pose (rare here)
                    if isinstance(data, dict) and "pose" in data and is_4x4_matrix(data["pose"]):
                        flat_pose = matrix_to_flat_dict("pose", data["pose"])
                        rows_to_write.append({
                            "timestamp": ts,
                            "source": src,
                            "event": ev,
                            **flat_pose
                        })
                        continue

                    # IMU samples -> flatten key fields for CSV
                    if src == "imu" and isinstance(data, dict):
                        pos = data.get("pos_m") or (None, None, None)
                        vel = data.get("vel_m_s") or (None, None, None)
                        yaw = data.get("yaw_deg")
                        pitch = data.get("pitch_deg")
                        roll = data.get("roll_deg")
                        rows_to_write.append({
                            "timestamp": ts,
                            "source": src,
                            "event": ev,
                            "raw_data": json.dumps(data),
                            "imu_x": float(pos[0]) if pos and pos[0] is not None else "",
                            "imu_y": float(pos[1]) if pos and pos[1] is not None else "",
                            "imu_z": float(pos[2]) if pos and pos[2] is not None else "",
                            "imu_yaw_deg": float(yaw) if yaw is not None else "",
                            "imu_pitch_deg": float(pitch) if pitch is not None else "",
                            "imu_roll_deg": float(roll) if roll is not None else "",
                            "imu_vx": float(vel[0]) if vel and vel[0] is not None else "",
                            "imu_vy": float(vel[1]) if vel and vel[1] is not None else "",
                            "imu_vz": float(vel[2]) if vel and vel[2] is not None else "",
                        })
                    else:
                        # fallback
                        safe_data = data
                        if isinstance(safe_data, np.ndarray):
                            safe_data = safe_data.tolist()
                        rows_to_write.append({
                            "timestamp": ts,
                            "source": src,
                            "event": ev,
                            "raw_data": json.dumps(safe_data) if safe_data else ""
                        })

            # add imu flat columns for easier analysis
            IMU_COLS = [
                "imu_x", "imu_y", "imu_z",
                "imu_yaw_deg", "imu_pitch_deg", "imu_roll_deg",
                "imu_vx", "imu_vy", "imu_vz",
            ]
            fieldnames = ["timestamp", "source", "event", "tag_id", "raw_data"] + IMU_COLS + POSE_COLS
            try:
                with open(log_filename, "w", newline="") as f:
                    csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
                    csv_writer.writeheader()
                    for row in rows_to_write:
                        csv_writer.writerow(row)
                logging.info(f"Saved {len(rows_to_write)} log rows to {log_filename}")
            except Exception as e:
                logging.error(f"Failed to save run logs: {e}")

        # close HDF5
        if writer is not None:
            try:
                writer.stop()
            except Exception as e:
                logging.error(f"Failed to stop HDF5 writer: {e}")
        
        # Clean up OpenCV windows safely
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Give OpenCV a moment to process the destroy event
        except Exception as e:
            logging.warning(f"OpenCV cleanup failed (non-fatal): {e}")