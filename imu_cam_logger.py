#!/usr/bin/env python3
import threading
import time
import logging
import csv
import queue
import numpy as np
import cv2
from typing import Dict, Optional
import os
import datetime
import json

from april_tag_processor import AprilTagProcessor
from imu_reader import IMUReader
from tools import matrix_to_flat_dict, is_4x4_matrix
from config import (
    WORLD_TAG_ID, FRAME_W, FRAME_H,
    OBJ_TAG_IDS, WORLD_TAG_SIZE, OBJ_TAG_SIZE,
    CAMERA_SERIALS,
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

TARGET_LOG_HZ = 30.0
LOG_INTERVAL = 1.0 / TARGET_LOG_HZ
POSE_COLS = [f"pose_{r}{c}" for r in range(4) for c in range(4)]

# ---------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------

shared_state = {
    "tag_pose_A": {},        # latest tag poses per tag id (world frame)
    "vision_image": {},      # latest annotated frame per camera
}
state_lock = threading.Lock()

# ---------------------------------------------------------------------
# Vision thread: one per camera
# ---------------------------------------------------------------------

def vision_processing_thread(
    serial_num: str,
    stop_event: threading.Event,
    log_event,
    discard: bool
):
    """
    For a single camera:
    - runs AprilTagProcessor
    - pushes tag poses via log_event
    - updates shared_state["vision_image"] and ["tag_pose_A"]
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
            return

        backoff = 0.01
        while not stop_event.is_set():
            try:
                vis_img, H0i_dict = processor.process_frame()

                # Update images and tag poses for visualization / fusion
                with state_lock:
                    shared_state["vision_image"][serial_num] = vis_img
                    shared_state["tag_pose_A"].update(H0i_dict)

                # Log this frame’s tag poses (even if identical)
                if not discard:
                    log_event(serial_num, "tag_pose_snapshot", {"pose": H0i_dict})

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

# ---------------------------------------------------------------------
# IMU thread
# ---------------------------------------------------------------------

def imu_thread(
    stop_event: threading.Event,
    log_event,
    discard: bool,
    imu_reader: IMUReader,
):
    """Poll IMUReader.get_latest() and push samples to the run log via log_event."""
    logging.info("IMU thread started")
    backoff = 0.01
    try:
        while not stop_event.is_set():
            try:
                sample = imu_reader.get_latest()
                if sample is not None and not discard:
                    # push the full sample dict so CSV writer can serialize it
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

# ---------------------------------------------------------------------
# Main orchestrator (IMU + cameras, no robot)
# ---------------------------------------------------------------------

def run_imu_camera_session(discard: bool = False):
    """
    Start IMU + cameras, log everything to memory, then dump to CSV when done.
    Press 'q' or ESC in the OpenCV window to stop.
    """
    stop_event = threading.Event()

    # --- IMU setup ---
    imu_reader = None
    imu_t = None
    try:
        imu_reader = IMUReader(rate_hz=50.0)
        imu_reader.start()
    except Exception as e:
        logging.warning(f"Failed to start IMUReader: {e}")

    # make dirs
    os.makedirs("CSV", exist_ok=True)

    # run log (in memory)
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

    # start vision threads
    vision_threads = [
        threading.Thread(
            target=vision_processing_thread,
            name=f"Vision[{sn}]",
            args=(sn, stop_event, log_event, discard),
            daemon=True,
        )
        for sn in CAMERA_SERIALS
    ]

    # start IMU thread
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

    # --- UI loop (camera mosaic) ---
    WIN = "IMU + Camera Logger"
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

        for t in vision_threads:
            t.join(timeout=1.0)

        if imu_t is not None:
            imu_t.join(timeout=1.0)
        if imu_reader is not None:
            try:
                imu_reader.stop()
            except Exception as e:
                logging.error(f"Failed to stop IMUReader: {e}")

        # -----------------------------
        # write CSV
        # -----------------------------
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
                                # store raw pose if not 4x4
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

                        # NEW: body-frame and world-frame accelerations
                        acc_body = data.get("acc_body") or (None, None, None)
                        acc_world = data.get("acc_world_m_s2") or (None, None, None)

                        rows_to_write.append({
                            "timestamp": ts,
                            "source": src,
                            "event": ev,
                            "raw_data": json.dumps(data),

                            # position (world)
                            "imu_x": float(pos[0]) if pos and pos[0] is not None else "",
                            "imu_y": float(pos[1]) if pos and pos[1] is not None else "",
                            "imu_z": float(pos[2]) if pos and pos[2] is not None else "",

                            # orientation
                            "imu_yaw_deg":   float(yaw)   if yaw   is not None else "",
                            "imu_pitch_deg": float(pitch) if pitch is not None else "",
                            "imu_roll_deg":  float(roll)  if roll  is not None else "",

                            # velocity (world)
                            "imu_vx": float(vel[0]) if vel and vel[0] is not None else "",
                            "imu_vy": float(vel[1]) if vel and vel[1] is not None else "",
                            "imu_vz": float(vel[2]) if vel and vel[2] is not None else "",

                            # NEW: accelerations (body frame)
                            "imu_ax_body": float(acc_body[0]) if acc_body and acc_body[0] is not None else "",
                            "imu_ay_body": float(acc_body[1]) if acc_body and acc_body[1] is not None else "",
                            "imu_az_body": float(acc_body[2]) if acc_body and acc_body[2] is not None else "",

                            # NEW: accelerations (world frame, after your processing)
                            "imu_ax_world": float(acc_world[0]) if acc_world and acc_world[0] is not None else "",
                            "imu_ay_world": float(acc_world[1]) if acc_world and acc_world[1] is not None else "",
                            "imu_az_world": float(acc_world[2]) if acc_world and acc_world[2] is not None else "",
                        })
                        continue

                    # fallback: just dump whatever data we have
                    safe_data = data
                    try:
                        json.dumps(data)
                    except TypeError:
                        safe_data = str(data)
                    rows_to_write.append({
                        "timestamp": ts,
                        "source": src,
                        "event": ev,
                        "raw_data": json.dumps(safe_data) if safe_data else ""
                    })

            # add imu flat columns for easier analysis
            # add imu flat columns for easier analysis
            IMU_COLS = [
                "imu_x", "imu_y", "imu_z",
                "imu_yaw_deg", "imu_pitch_deg", "imu_roll_deg",
                "imu_vx", "imu_vy", "imu_vz",
                # NEW accel columns:
                "imu_ax_body", "imu_ay_body", "imu_az_body",
                "imu_ax_world", "imu_ay_world", "imu_az_world",
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

# ---------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s"
    )
    run_imu_camera_session(discard=False)
