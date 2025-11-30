#!/usr/bin/env python3
"""
Analyze camera performance by comparing poses from different cameras
and against the robot, after transforming everything into the robot base frame.

For each CSV, we can:
  1. Transform camera-frame tag poses to robot base frame using hand-eye transforms.
  2. Time-align samples between sensors using nearest-neighbor in time with a tolerance.
  3. Compute error vectors and statistics on ||p_A - p_B|| (in *millimeters*).

Assumptions about the CSV (matches your concurrent logger):
-----------------------------------------------------------
Columns:
    timestamp  : ISO string or numeric time
    source     : 'robot' for robot pose rows, camera serial for camera rows
    event      : e.g. 'pose_snapshot' for robot, something like 'tag_pose' for cameras
    tag_id     : AprilTag ID for camera rows (robot rows may have NaN here)
    pose_ij    : entries of a 4x4 pose matrix, row i col j, i,j in {0,1,2,3}.

For cameras:
    The 4x4 matrix encodes the tag pose in the camera frame (T_cam_tag).
For robot:
    The 4x4 matrix encodes the robot/tool pose in the robot base frame (T_base_robot).

Hand–eye calibration gives:
    T_base_cam  (cam->robot-base)  from separate .npz files.

Then we compute:
    T_base_tag_from_cam = T_base_cam @ T_cam_tag
    p_base_tag_from_cam = translation of T_base_tag_from_cam

    T_base_robot        = matrix from robot row
    p_base_robot        = translation of T_base_robot

All distances are reported in millimeters.
"""

import argparse
import glob
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import DEFAULT_TRANSFORM_PATH, DEFAULT_TRANSFORM_DIR, DEFAULT_CSV_DIR
from tools import H_to_xyzrpy_ZYX, rot_geodesic_angle_deg
import numpy as _np
MM = 1000.0  # (meters -> millimeters)




# ---------------------------------------------------------------------
# Transform loading
# ---------------------------------------------------------------------
def load_cam_to_robot_transforms(
    transform_file: Optional[str],
    transform_dir: str,
    cam_ids_in_csv: List[str],
) -> Dict[str, np.ndarray]:
    """
    Load cam->robot transforms for the cameras present in the CSV.

    Supports:
      1) New style: per-camera files in a directory:
         DATA/hand_eye/cam_<CAMID>_to_robot_transform.npz
         with keys: R (3x3), t (3,), (and possibly stats, source).

      2) Old style: a single .npz where each key is a cam id and each value is
         a 4x4 homogeneous matrix.

      3) A single .npz with R (3x3), t (3,) only -> treated as a global transform
         for any camera that doesn't already have a specific one.
    """
    cam_ids_in_csv = [str(c) for c in cam_ids_in_csv]
    transforms: Dict[str, np.ndarray] = {}

    # ---- 1) Per-camera files in transform_dir (preferred) ----
    for cam_id in cam_ids_in_csv:
        fname = f"cam_{cam_id}_to_robot_transform.npz"
        fpath = os.path.join(transform_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            data = np.load(fpath, allow_pickle=True)
        except Exception as e:
            print(f"[WARN] Failed to load per-camera transform for {cam_id} ({fpath}): {e}")
            continue

        T = None
        # New-style: R (3x3) and t (3,)
        if "R" in data.files and "t" in data.files:
            R = np.asarray(data["R"], dtype=float)
            t = np.asarray(data["t"], dtype=float).reshape(3)
            if R.shape == (3, 3) and t.shape == (3,):
                T = np.eye(4, dtype=float)
                T[:3, :3] = R
                T[:3, 3] = t
            else:
                print(f"[WARN] Invalid R/t shapes in {fpath}: R {R.shape}, t {t.shape}")
        else:
            # Fallback: look for a 4x4 matrix inside
            for key in data.files:
                arr = np.asarray(data[key])
                if arr.shape == (4, 4):
                    T = arr
                    break

        if T is not None:
            transforms[cam_id] = T
            print(f"[INFO] Using per-camera transform for {cam_id} from {fpath}")
        else:
            print(f"[WARN] No valid transform found in {fpath}")

    # If we already have everything for this CSV, we can skip reading a global file
    if transform_file is None and len(transforms) == len(cam_ids_in_csv):
        print(f"[INFO] Loaded {len(transforms)} per-camera transforms from directory {transform_dir}")
        return transforms

    # ---- 2) Optional: old/global transform file ----
    if transform_file is None:
        return transforms

    if not os.path.exists(transform_file):
        print(f"[WARN] Transform file not found: {transform_file}")
        return transforms

    try:
        data = np.load(transform_file, allow_pickle=True)
    except Exception as e:
        print(f"[WARN] Failed to load transform file {transform_file}: {e}")
        return transforms

    # First try "old" multi-4x4 style
    multi_count = 0
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.shape == (4, 4):
            transforms[str(key)] = arr
            multi_count += 1

    if multi_count > 0:
        print(f"[INFO] Loaded {multi_count} 4x4 transforms from {transform_file}")
        return transforms

    # Then try single global R,t
    if "R" in data.files and "t" in data.files:
        R = np.asarray(data["R"], dtype=float)
        t = np.asarray(data["t"], dtype=float).reshape(3)
        if R.shape == (3, 3) and t.shape == (3,):
            T_global = np.eye(4, dtype=float)
            T_global[:3, :3] = R
            T_global[:3, 3] = t
            # Assign to any camera that doesn't already have a per-camera transform
            missing = [cid for cid in cam_ids_in_csv if cid not in transforms]
            for cid in missing:
                transforms[cid] = T_global
            print(
                f"[INFO] Loaded global R,t transform from {transform_file} "
                f"and applied to {len(missing)} cameras without per-camera files."
            )
        else:
            print(f"[WARN] Invalid global R/t shapes in {transform_file}: R {R.shape}, t {t.shape}")
    else:
        print(f"[WARN] No usable transforms found in {transform_file} (no 4x4, no R/t).")

    if not transforms:
        raise ValueError(
            f"No valid transforms found for cameras {cam_ids_in_csv}. "
            f"Check {transform_dir} and transform file {transform_file}."
        )

    return transforms


# ---------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------
POSE_PREFIX = "pose_"


def pose_row_to_matrix(row: pd.Series, prefix: str = POSE_PREFIX) -> np.ndarray:
    """
    Build a 4x4 pose matrix from a row with columns:
        pose_00, pose_01, ..., pose_33
    """
    T = np.eye(4, dtype=float)
    for r in range(4):
        for c in range(4):
            key = f"{prefix}{r}{c}"
            T[r, c] = float(row[key])
    return T


def add_camera_base_columns(df_cam: pd.DataFrame, cam_id: str, T_cam_to_robot: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    For a camera dataframe, compute the tag position in the robot base frame
    using the full 4x4 pose matrices.

    Assumes:
        T_cam_tag (from CSV) and T_base_cam (from hand-eye) so:
        T_base_tag = T_base_cam @ T_cam_tag
    """
    if cam_id not in T_cam_to_robot:
        raise KeyError(f"No transform found for camera {cam_id}")

    T_base_cam = T_cam_to_robot[cam_id]

    xs = []
    ys = []
    zs = []
    for _, row in df_cam.iterrows():
        T_cam_tag = pose_row_to_matrix(row)
        T_base_tag = T_base_cam @ T_cam_tag
        p = T_base_tag[:3, 3]
        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2])

    df_cam = df_cam.copy()
    df_cam["x_base"] = xs
    df_cam["y_base"] = ys
    df_cam["z_base"] = zs
    return df_cam


def add_robot_base_columns(df_robot: pd.DataFrame) -> pd.DataFrame:
    """
    For robot rows, the pose matrix is assumed to already be in the robot base frame:
        T_base_robot (base->tool or base->whatever).
    We just extract the translation.
    """
    xs = []
    ys = []
    zs = []
    for _, row in df_robot.iterrows():
        T_base_robot = pose_row_to_matrix(row)
        p = T_base_robot[:3, 3]
        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2])

    df_robot = df_robot.copy()
    df_robot["x_base"] = xs
    df_robot["y_base"] = ys
    df_robot["z_base"] = zs
    return df_robot


# ---------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------
def analyze_camera_pair(
    df: pd.DataFrame,
    cam_a: str,
    cam_b: str,
    T_cam_to_robot: Dict[str, np.ndarray],
    time_tol: float,
    time_col: str,
    cam_col: str,
    tag_col: str,
    tag_id_filter: Optional[int] = None,
) -> None:
    """
    For a given DataFrame and pair of cameras, compute position error statistics
    in the robot base frame, per tag, and print metrics in millimeters.
    """
    df_pair = df.copy()

    # Optional: filter by tag_id
    if tag_id_filter is not None:
        df_pair = df_pair[df_pair[tag_col] == tag_id_filter]

    # Filter rows for each camera
    df_a = df_pair[df_pair[cam_col] == cam_a].copy()
    df_b = df_pair[df_pair[cam_col] == cam_b].copy()

    if df_a.empty or df_b.empty:
        print(f"[WARN] No data for camera pair ({cam_a}, {cam_b}) in this CSV.")
        return

    # Add robot-base positions for each camera
    df_a = add_camera_base_columns(df_a, cam_a, T_cam_to_robot)
    df_b = add_camera_base_columns(df_b, cam_b, T_cam_to_robot)

    # Sort by time for merge_asof
    df_a = df_a.sort_values(time_col)
    df_b = df_b.sort_values(time_col)

    # We'll align per-tag
    tags_a = set(df_a[tag_col].dropna().unique())
    tags_b = set(df_b[tag_col].dropna().unique())
    common_tags = sorted(tags_a & tags_b)

    if not common_tags:
        print(f"[WARN] No common tag_ids for cameras {cam_a} and {cam_b}.")
        return

    all_errors = []

    for tag_id in common_tags:
        sub_a = df_a[df_a[tag_col] == tag_id].copy()
        sub_b = df_b[df_b[tag_col] == tag_id].copy()
        if sub_a.empty or sub_b.empty:
            continue

        # Decide tolerance type based on time dtype
        if np.issubdtype(sub_a[time_col].dtype, np.datetime64):
            tol = pd.Timedelta(seconds=time_tol)
        else:
            tol = time_tol

        merged = pd.merge_asof(
            sub_a.sort_values(time_col),
            sub_b.sort_values(time_col),
            on=time_col,
            direction="nearest",
            tolerance=tol,
            suffixes=("_a", "_b"),
        )

        merged = merged.dropna(subset=["x_base_b", "y_base_b", "z_base_b"])
        if merged.empty:
            continue

        pts_a = merged[["x_base_a", "y_base_a", "z_base_a"]].to_numpy()
        pts_b = merged[["x_base_b", "y_base_b", "z_base_b"]].to_numpy()

        diffs = pts_a - pts_b
        dists = np.linalg.norm(diffs, axis=1)  # meters
        all_errors.append(dists)

    if not all_errors:
        print(f"[WARN] No aligned samples for cameras {cam_a} and {cam_b}.")
        return

    all_errors = np.concatenate(all_errors)

    mse = float(np.mean(all_errors ** 2))
    rmse = float(np.sqrt(mse))
    mean_err = float(np.mean(all_errors))
    median_err = float(np.median(all_errors))
    p95 = float(np.percentile(all_errors, 95))

    # Convert to millimeters for reporting
    mse_mm2 = mse * (MM ** 2)
    rmse_mm = rmse * MM
    mean_err_mm = mean_err * MM
    median_err_mm = median_err * MM
    p95_mm = p95 * MM

    print(f"\n=== Camera pair: {cam_a} vs {cam_b} ===")
    if tag_id_filter is not None:
        print(f"Tag filter: {tag_id_filter}")
    print(f"Aligned samples : {len(all_errors)}")
    print(f"Mean |Δp|       : {mean_err_mm:.2f} mm")
    print(f"Median |Δp|     : {median_err_mm:.2f} mm")
    print(f"95th percentile : {p95_mm:.2f} mm")
    print(f"MSE             : {mse_mm2:.2f} mm^2")
    print(f"RMSE            : {rmse_mm:.2f} mm")


def analyze_camera_vs_robot(
    df: pd.DataFrame,
    cam_id: str,
    T_cam_to_robot: Dict[str, np.ndarray],
    time_tol: float,
    time_col: str,
    cam_col: str,
    tag_col: str,
    robot_source_name: str = "robot",
    tag_id_filter: Optional[int] = None,
) -> None:
    """
    Compare a camera's tag-based pose estimates against the robot's reported pose.

    For each matched time pair (camera tag observation, robot pose snapshot),
    we compute:
        p_base_tag_from_cam (from camera + hand-eye)
        p_base_robot        (from robot 4x4 matrix)
    and then statistics on ||p_cam - p_robot||, in millimeters.
    """
    df_all = df.copy()

    # Optional: filter by tag_id for camera
    df_cam = df_all[df_all[cam_col] == cam_id].copy()
    if tag_id_filter is not None:
        df_cam = df_cam[df_cam[tag_col] == tag_id_filter]

    df_robot = df_all[df_all[cam_col] == robot_source_name].copy()

    if df_cam.empty:
        print(f"[WARN] No camera data for {cam_id} in this CSV.")
        return
    if df_robot.empty:
        print(f"[WARN] No robot data (source == '{robot_source_name}') in this CSV.")
        return

    # Compute base-frame positions
    df_cam = add_camera_base_columns(df_cam, cam_id, T_cam_to_robot)
    df_robot = add_robot_base_columns(df_robot)

    df_cam = df_cam.sort_values(time_col)
    df_robot = df_robot.sort_values(time_col)

    # Decide tolerance type based on time dtype
    if np.issubdtype(df_cam[time_col].dtype, np.datetime64):
        tol = pd.Timedelta(seconds=time_tol)
    else:
        tol = time_tol

    merged = pd.merge_asof(
        df_cam,
        df_robot,
        on=time_col,
        direction="nearest",
        tolerance=tol,
        suffixes=("_cam", "_robot"),
    )

    merged = merged.dropna(subset=["x_base_robot", "y_base_robot", "z_base_robot"])
    if merged.empty:
        print(f"[WARN] No aligned samples for camera {cam_id} vs robot within tolerance.")
        return

    pts_cam = merged[["x_base_cam", "y_base_cam", "z_base_cam"]].to_numpy()
    pts_robot = merged[["x_base_robot", "y_base_robot", "z_base_robot"]].to_numpy()

    diffs = pts_cam - pts_robot
    dists = np.linalg.norm(diffs, axis=1)  # meters

    mse = float(np.mean(dists ** 2))
    rmse = float(np.sqrt(mse))
    mean_err = float(np.mean(dists))
    median_err = float(np.median(dists))
    p95 = float(np.percentile(dists, 95))

    # Convert to millimeters for reporting
    mse_mm2 = mse * (MM ** 2)
    rmse_mm = rmse * MM
    mean_err_mm = mean_err * MM
    median_err_mm = median_err * MM
    p95_mm = p95 * MM

    print(f"\n=== Camera vs Robot: {cam_id} vs {robot_source_name} ===")
    if tag_id_filter is not None:
        print(f"Tag filter (camera side): {tag_id_filter}")
    print(f"Aligned samples : {len(dists)}")
    print(f"Mean |Δp|       : {mean_err_mm:.2f} mm")
    print(f"Median |Δp|     : {median_err_mm:.2f} mm")
    print(f"95th percentile : {p95_mm:.2f} mm")
    print(f"MSE             : {mse_mm2:.2f} mm^2")
    print(f"RMSE            : {rmse_mm:.2f} mm")


def analyze_imu_vs_robot(
    df: pd.DataFrame,
    imu_source: str,
    robot_source_name: str = "robot",
    time_tol: float = 0.02,
    time_col: str = "timestamp",
):
    """
    Compare IMU-integrated position/yaw/pitch/roll against robot pose snapshots.

    Expects IMU rows to have columns: imu_x, imu_y, imu_z, imu_yaw_deg, imu_pitch_deg, imu_roll_deg
    (these are produced by the concurrent runner CSV writer).
    """
    if imu_source not in set(df["source"].astype(str).unique()):
        print(f"[INFO] No IMU rows with source '{imu_source}' in CSV; skipping IMU vs robot analysis.")
        return

    df_imu = df[df["source"] == imu_source].copy()
    df_robot = df[df["source"] == robot_source_name].copy()

    if df_imu.empty:
        print("[WARN] No IMU rows found; skipping IMU analysis.")
        return
    if df_robot.empty:
        print("[WARN] No robot rows found; cannot compare IMU to robot.")
        return

    # convert times if necessary
    if not np.issubdtype(df[time_col].dtype, np.number):
        try:
            df_imu[time_col] = pd.to_datetime(df_imu[time_col])
            df_robot[time_col] = pd.to_datetime(df_robot[time_col])
            tol = pd.Timedelta(seconds=time_tol)
        except Exception:
            tol = time_tol
    else:
        tol = time_tol

    # pick relevant columns and drop rows with missing pos
    imu_cols = ["imu_x", "imu_y", "imu_z", "imu_yaw_deg", "imu_pitch_deg", "imu_roll_deg"]
    for c in imu_cols:
        if c not in df_imu.columns:
            print(f"[WARN] IMU column {c} not found in CSV; skipping IMU analysis.")
            return

    imu_sel = df_imu[[time_col] + imu_cols].dropna()
    robot_sel = df_robot[[time_col] + [f"{POSE_PREFIX}{r}{c}" for r in range(4) for c in range(4)]].dropna()

    if imu_sel.empty or robot_sel.empty:
        print("[WARN] Not enough IMU or robot data for analysis.")
        return

    # convert robot pose rows to positions and yaw/pitch/roll (ZYX)
    robot_sel = robot_sel.copy()
    robot_sel["x_base"] = robot_sel.apply(lambda row: pose_row_to_matrix(row)[0,3], axis=1)
    robot_sel["y_base"] = robot_sel.apply(lambda row: pose_row_to_matrix(row)[1,3], axis=1)
    robot_sel["z_base"] = robot_sel.apply(lambda row: pose_row_to_matrix(row)[2,3], axis=1)
    # extract robot RPY (ZYX) and convert to degrees to match IMU degrees
    robot_sel[["r_roll", "r_pitch", "r_yaw"]] = robot_sel.apply(
        lambda row: pd.Series(_np.degrees(H_to_xyzrpy_ZYX(pose_row_to_matrix(row))[3:6])), axis=1
    )

    # sort and merge asof
    imu_sorted = imu_sel.sort_values(time_col)
    robot_sorted = robot_sel.sort_values(time_col)

    merged = pd.merge_asof(
        imu_sorted,
        robot_sorted,
        on=time_col,
        direction="nearest",
        tolerance=tol,
    )

    merged = merged.dropna(subset=["x_base"])
    if merged.empty:
        print("[WARN] No aligned IMU<->robot samples within tolerance.")
        return

    imu_pts = merged[["imu_x", "imu_y", "imu_z"]].to_numpy(dtype=float)
    robot_pts = merged[["x_base", "y_base", "z_base"]].to_numpy(dtype=float)
    pos_errs = np.linalg.norm(imu_pts - robot_pts, axis=1)

    # orientation errors in degrees (yaw/pitch/roll separately and norm)
    # IMU RPY columns are named imu_yaw_deg, imu_pitch_deg, imu_roll_deg in CSV.
    # Build arrays in robot order: roll, pitch, yaw (degrees)
    imu_rpy = merged[["imu_roll_deg", "imu_pitch_deg", "imu_yaw_deg"]].to_numpy(dtype=float)
    rob_rpy = merged[["r_roll", "r_pitch", "r_yaw"]].to_numpy(dtype=float)

    # per-axis Euler diffs (wrap to [-180,180])
    def wrap_deg(d):
        return (d + 180.0) % 360.0 - 180.0

    rpy_diff = wrap_deg(imu_rpy - rob_rpy)
    rpy_abs = np.abs(rpy_diff)

    def stats(vals):
        return float(np.mean(vals)), float(np.median(vals)), float(np.percentile(vals, 95)), float(np.sqrt(np.mean(vals ** 2)))

    mean_pos, med_pos, p95_pos, rmse_pos = stats(pos_errs)
    print("\n=== IMU vs Robot ===")
    print(f"Aligned samples: {len(pos_errs)}")
    print(f"Position error (m): mean={mean_pos:.4f}, median={med_pos:.4f}, p95={p95_pos:.4f}, rmse={rmse_pos:.4f}")
    print("Orientation absolute error per-axis (deg):")
    for i, name in enumerate(["roll", "pitch", "yaw"]):
        mean_o, med_o, p95_o, rmse_o = stats(rpy_abs[:, i])
        print(f"  {name}: mean={mean_o:.2f}, median={med_o:.2f}, p95={p95_o:.2f}, rmse={rmse_o:.2f}")

    # Also compute geodesic rotation error per-sample using full rotation matrices
    def rpy_to_R_deg(yaw_deg, pitch_deg, roll_deg):
        y = np.radians(yaw_deg); p = np.radians(pitch_deg); r = np.radians(roll_deg)
        cy, sy = np.cos(y), np.sin(y)
        cp, sp = np.cos(p), np.sin(p)
        cr, sr = np.cos(r), np.sin(r)
        Rz = np.array([[cy, -sy, 0.0],[sy, cy, 0.0],[0.0,0.0,1.0]])
        Ry = np.array([[cp, 0.0, sp],[0.0,1.0,0.0],[-sp,0.0,cp]])
        Rx = np.array([[1.0,0.0,0.0],[0.0,cr,-sr],[0.0,sr,cr]])
        return Rz @ Ry @ Rx

    geo_angles = []
    for _, row in merged.iterrows():
        try:
            R_robot = pose_row_to_matrix(row)[:3,:3]
            R_imu = rpy_to_R_deg(row.get("imu_yaw_deg"), row.get("imu_pitch_deg"), row.get("imu_roll_deg"))
            ang = rot_geodesic_angle_deg(R_robot, R_imu)
            geo_angles.append(ang)
        except Exception:
            geo_angles.append(float('nan'))

    geo = np.array(geo_angles)
    geo = geo[~np.isnan(geo)]
    if geo.size:
        mean_g, med_g, p95_g, rmse_g = stats(geo)
        print(f"Geodesic rotation error (deg): mean={mean_g:.2f}, median={med_g:.2f}, p95={p95_g:.2f}, rmse={rmse_g:.2f}")


# ---------------------------------------------------------------------
# CSV selection
# ---------------------------------------------------------------------
def choose_csv_interactively(csv_dir: str) -> str:
    """
    Mimic your 'find rot tran' behaviour:
    list CSV/session_log_*.csv, sort, print, choose index, Enter = last.
    """
    pattern = os.path.join(csv_dir, "session_log_*.csv")
    files = [os.path.basename(f) for f in glob.glob(pattern)]
    if not files:
        raise FileNotFoundError(f"No CSV files found with pattern {pattern}")

    files = sorted(files)
    print("Available CSV files in", csv_dir)
    for i, name in enumerate(files):
        print(f"  {i}: {name}")

    num = input(
        "File to load? 0 for the first and so on...\n"
        "Press Enter for the last one\n :"
    )
    if not num.strip():
        idx = -1
    else:
        idx = int(num)

    chosen = files[idx]
    print("You chose", chosen)
    return os.path.join(csv_dir, chosen)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze camera performance (camera vs camera, and camera vs robot) in robot base frame."
    )
    parser.add_argument(
        "--transform",
        default=DEFAULT_TRANSFORM_PATH,
        help=(
            "Optional path to a cam->robot transform .npz.\n"
            " - Old style: multiple 4x4 matrices keyed by camera id.\n"
            " - New style: single R (3x3), t (3,) used as global fallback.\n"
            "If omitted, only per-camera files in --transform-dir are used."
        ),
    )
    parser.add_argument(
        "--transform-dir",
        default=DEFAULT_TRANSFORM_DIR,
        help=(
            "Directory with per-camera transforms named "
            "cam_<CAMID>_to_robot_transform.npz (default: DATA/hand_eye)"
        ),
    )
    parser.add_argument("--csv", help="Path to a specific CSV file (if omitted, select from CSV/)")
    parser.add_argument(
        "--csv-dir",
        default=DEFAULT_CSV_DIR,
        help=f"Directory with CSV/session_log_*.csv (default: {DEFAULT_CSV_DIR})",
    )

    parser.add_argument(
        "--time-col",
        default="timestamp",
        help="Time column name (ISO string or float seconds, default: 'timestamp')",
    )
    parser.add_argument(
        "--cam-col",
        default="source",
        help="Sensor/camera ID column name (default: 'source')",
    )
    parser.add_argument(
        "--tag-col",
        default="tag_id",
        help="Tag ID column name (default: 'tag_id')",
    )
    parser.add_argument(
        "--time-tol",
        type=float,
        default=0.02,
        help="Max time difference (seconds) for matching samples (default: 0.02)",
    )
    parser.add_argument(
        "--tag-id",
        type=int,
        help="Optional tag_id to filter on (camera side; default: use all tags)",
    )
    parser.add_argument(
        "--robot-source-name",
        default="robot",
        help="Value of 'source' column that denotes robot pose rows (default: 'robot')",
    )

    args = parser.parse_args()

    # ---------------------------
    # Choose ONE CSV
    # ---------------------------
    if args.csv:
        csv_file = args.csv
    else:
        csv_file = choose_csv_interactively(args.csv_dir)

    print("\n##############################")
    print(f"[INFO] Processing CSV: {csv_file}")
    print("##############################")

    # ---------------------------
    # Load CSV
    # ---------------------------
    df = pd.read_csv(csv_file)

    # Convert timestamp to datetime if it's not numeric
    if args.time_col in df.columns:
        if not np.issubdtype(df[args.time_col].dtype, np.number):
            try:
                df[args.time_col] = pd.to_datetime(df[args.time_col])
            except Exception as e:
                print(f"[WARN] Failed to parse {args.time_col} as datetime: {e}")

    # Check required columns
    pose_cols = [f"{POSE_PREFIX}{r}{c}" for r in range(4) for c in range(4)]
    required_cols = [args.time_col, args.cam_col, args.tag_col] + pose_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] CSV {csv_file} missing required columns: {missing}")
        print("        Expected pose_ij columns like 'pose_00', 'pose_01', ..., 'pose_33'.")
        return

    # All camera IDs (everything except the robot)
    all_sources = set(df[args.cam_col].astype(str).unique())
    robot_name = args.robot_source_name
    cam_ids_in_csv = sorted(s for s in all_sources if s != robot_name)

    if not cam_ids_in_csv:
        print(f"[ERROR] No camera sources found in CSV (sources: {sorted(all_sources)})")
        return

    try:
        T_cam_to_robot = load_cam_to_robot_transforms(
            args.transform, args.transform_dir, cam_ids_in_csv
        )
    except Exception as e:
        print(f"[ERROR] Could not load transforms for {csv_file}: {e}")
        return

    # Keep only cameras that actually have a transform
    cam_ids = [cid for cid in cam_ids_in_csv if cid in T_cam_to_robot]

    if len(cam_ids) < 1:
        print(f"[ERROR] No cameras with valid transforms were found. Cameras in CSV: {cam_ids_in_csv}")
        return

    # ---------------------------
    # Camera-vs-camera analysis
    # ---------------------------
    if len(cam_ids) >= 2:
        for i in range(len(cam_ids)):
            for j in range(i + 1, len(cam_ids)):
                cam_a = cam_ids[i]
                cam_b = cam_ids[j]
                analyze_camera_pair(
                    df,
                    cam_a,
                    cam_b,
                    T_cam_to_robot,
                    time_tol=args.time_tol,
                    time_col=args.time_col,
                    cam_col=args.cam_col,
                    tag_col=args.tag_col,
                    tag_id_filter=args.tag_id,
                )
    else:
        print("[INFO] Only one camera with a valid transform; skipping camera-to-camera comparison.")

    # ---------------------------
    # Camera-vs-robot analysis
    # ---------------------------
    has_robot = (df[args.cam_col] == robot_name).any()
    if has_robot:
        for cam_id in cam_ids:
            analyze_camera_vs_robot(
                df,
                cam_id,
                T_cam_to_robot,
                time_tol=args.time_tol,
                time_col=args.time_col,
                cam_col=args.cam_col,
                tag_col=args.tag_col,
                robot_source_name=robot_name,
                tag_id_filter=args.tag_id,
            )
    else:
        print(f"[INFO] No robot rows found (source == '{robot_name}'); skipping camera-to-robot comparison.")

    # IMU vs robot analysis (if IMU rows present)
    imu_source_candidates = [s for s in set(df[args.cam_col].astype(str).unique()) if s.lower().startswith("imu") or s == "imu"]
    if imu_source_candidates:
        imu_source = imu_source_candidates[0]
        analyze_imu_vs_robot(df, imu_source, robot_source_name=robot_name, time_tol=args.time_tol, time_col=args.time_col)


if __name__ == "__main__":
    main()
