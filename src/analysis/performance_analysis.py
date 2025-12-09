#!/usr/bin/env python3
"""
Analyze camera performance by comparing poses from different cameras
and against the robot, after transforming everything into the robot base frame.

For each CSV, we can:
  1. Transform camera-frame tag poses to robot base frame using hand-eye transforms.
  2. Time-align samples between sensors using nearest-neighbor in time with a tolerance.
  3. Compute error vectors and statistics on ||p_A - p_B|| (in *millimeters*).
"""

import numpy as np
import pandas as pd
import argparse
import glob
import os
from typing import Dict, List, Optional

# We assume these tools exist in your utils; if not, you might need to adjust imports
from src.utils.tools import H_to_xyzrpy_ZYX, rot_geodesic_angle_deg

# ---------------------------------------------------------------------
# CONFIG: Paths relative to project root
# ---------------------------------------------------------------------
DEFAULT_CSV_DIR = "data/CSV"
DEFAULT_TRANSFORM_DIR = "data/DATA/hand_eye"
# Fallback global transform if per-camera one isn't found
DEFAULT_TRANSFORM_PATH = "data/DATA/hand_eye/cam_to_robot_transform.npz"

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
    """
    cam_ids_in_csv = [str(c) for c in cam_ids_in_csv]
    transforms: Dict[str, np.ndarray] = {}

    # Ensure dir exists before listing
    if not os.path.exists(transform_dir):
        print(f"[WARN] Transform directory not found: {transform_dir}")
    else:
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
        # Only warn if the user explicitly provided a path or if we found NOTHING
        if not transforms:
            print(f"[WARN] Global transform file not found: {transform_file}")
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
            if missing:
                print(
                    f"[INFO] Loaded global R,t transform from {transform_file} "
                    f"and applied to {len(missing)} cameras without per-camera files."
                )
        else:
            print(f"[WARN] Invalid global R/t shapes in {transform_file}: R {R.shape}, t {t.shape}")
    else:
        print(f"[WARN] No usable transforms found in {transform_file} (no 4x4, no R/t).")

    if not transforms:
        print(
            f"[ERROR] No valid transforms found for cameras {cam_ids_in_csv}. "
            f"Check {transform_dir} or ensure a global transform exists."
        )
        raise ValueError("No transforms found")

    return transforms


def load_imu_to_robot_transform(
        imu_source: str,
        transform_dir: str,
) -> Optional[np.ndarray]:
    """
    Load imu->robot transform:
        DATA/hand_eye/imu_<IMUSRC>_to_robot_transform.npz
    """
    fname = f"imu_{imu_source}_to_robot_transform.npz"
    fpath = os.path.join(transform_dir, fname)
    if not os.path.exists(fpath):
        print(f"[INFO] No IMU transform file found for '{imu_source}' in {transform_dir}")
        return None

    try:
        data = np.load(fpath, allow_pickle=True)
    except Exception as e:
        print(f"[WARN] Failed to load IMU transform {fpath}: {e}")
        return None

    if "R" not in data.files:
        print(f"[WARN] IMU transform {fpath} missing R; ignoring.")
        return None

    R = np.asarray(data["R"], dtype=float)
    if R.shape != (3, 3):
        print(f"[WARN] IMU transform {fpath} has invalid R shape: {R.shape}")
        return None

    if "t" in data.files:
        t = np.asarray(data["t"], dtype=float).reshape(-1)
        if t.shape != (3,):
            print(f"[WARN] IMU transform {fpath} has invalid t shape: {t.shape}, using zeros.")
            t = np.zeros(3, dtype=float)
    else:
        t = np.zeros(3, dtype=float)

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    print(f"[INFO] Loaded IMU transform for '{imu_source}' from {fpath}")
    return T


# ---------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------
POSE_PREFIX = "pose_"


def pose_row_to_matrix(row: pd.Series, prefix: str = POSE_PREFIX) -> np.ndarray:
    T = np.eye(4, dtype=float)
    for r in range(4):
        for c in range(4):
            key = f"{prefix}{r}{c}"
            T[r, c] = float(row[key])
    return T


def add_camera_base_columns(df_cam: pd.DataFrame, cam_id: str, T_cam_to_robot: Dict[str, np.ndarray]) -> pd.DataFrame:
    if cam_id not in T_cam_to_robot:
        raise KeyError(f"No transform found for camera {cam_id}")

    T_base_cam = T_cam_to_robot[cam_id]

    xs, ys, zs = [], [], []
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
    xs, ys, zs = [], [], []
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
    df_pair = df.copy()

    if tag_id_filter is not None:
        df_pair = df_pair[df_pair[tag_col] == tag_id_filter]

    df_a = df_pair[df_pair[cam_col] == cam_a].copy()
    df_b = df_pair[df_pair[cam_col] == cam_b].copy()

    if df_a.empty or df_b.empty:
        print(f"[WARN] No data for camera pair ({cam_a}, {cam_b}) in this CSV.")
        return

    df_a = add_camera_base_columns(df_a, cam_a, T_cam_to_robot)
    df_b = add_camera_base_columns(df_b, cam_b, T_cam_to_robot)

    df_a = df_a.sort_values(time_col)
    df_b = df_b.sort_values(time_col)

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
    df_all = df.copy()

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

    df_cam = add_camera_base_columns(df_cam, cam_id, T_cam_to_robot)
    df_robot = add_robot_base_columns(df_robot)

    df_cam = df_cam.sort_values(time_col)
    df_robot = df_robot.sort_values(time_col)

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
    dists = np.linalg.norm(diffs, axis=1)

    mse = float(np.mean(dists ** 2))
    rmse = float(np.sqrt(mse))
    mean_err = float(np.mean(dists))
    median_err = float(np.median(dists))
    p95 = float(np.percentile(dists, 95))

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
        transform_dir: str = DEFAULT_TRANSFORM_DIR,
):
    if imu_source not in set(df["source"].astype(str).unique()):
        return

    df_imu = df[df["source"] == imu_source].copy()
    df_robot = df[df["source"] == robot_source_name].copy()

    if df_imu.empty or df_robot.empty:
        return

    if not np.issubdtype(df[time_col].dtype, np.number):
        try:
            df_imu[time_col] = pd.to_datetime(df_imu[time_col])
            df_robot[time_col] = pd.to_datetime(df_robot[time_col])
            tol = pd.Timedelta(seconds=time_tol)
        except Exception:
            tol = time_tol
    else:
        tol = time_tol

    imu_cols = ["imu_x", "imu_y", "imu_z", "imu_yaw_deg", "imu_pitch_deg", "imu_roll_deg"]
    for c in imu_cols:
        if c not in df_imu.columns:
            return

    imu_sel = df_imu[[time_col] + imu_cols].dropna()
    robot_sel = df_robot[[time_col] + [f"{POSE_PREFIX}{r}{c}" for r in range(4) for c in range(4)]].dropna()

    if imu_sel.empty or robot_sel.empty:
        return

    robot_sel = robot_sel.copy()
    robot_sel["x_base"] = robot_sel.apply(lambda row: pose_row_to_matrix(row)[0, 3], axis=1)
    robot_sel["y_base"] = robot_sel.apply(lambda row: pose_row_to_matrix(row)[1, 3], axis=1)
    robot_sel["z_base"] = robot_sel.apply(lambda row: pose_row_to_matrix(row)[2, 3], axis=1)
    robot_sel[["r_roll", "r_pitch", "r_yaw"]] = robot_sel.apply(
        lambda row: pd.Series(np.degrees(H_to_xyzrpy_ZYX(pose_row_to_matrix(row))[3:6])), axis=1
    )

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
        return

    T_imu_cal = load_imu_to_robot_transform(imu_source, transform_dir)
    R_cal = T_imu_cal[:3, :3] if T_imu_cal is not None else None

    # --- Position errors ---
    imu_pts = merged[["imu_x", "imu_y", "imu_z"]].to_numpy(dtype=float)
    robot_pts = merged[["x_base", "y_base", "z_base"]].to_numpy(dtype=float)

    if T_imu_cal is not None:
        R_imu = T_imu_cal[:3, :3]
        t_imu = T_imu_cal[:3, 3]
        imu_pts_aligned = (R_imu @ imu_pts.T).T + t_imu
    else:
        imu_pts_aligned = imu_pts

    pos_errs = np.linalg.norm(imu_pts_aligned - robot_pts, axis=1)

    def stats(vals):
        vals = np.asarray(vals, dtype=float)
        return (
            float(np.mean(vals)),
            float(np.median(vals)),
            float(np.percentile(vals, 95)),
            float(np.sqrt(np.mean(vals ** 2))),
        )

    mean_pos_m, med_pos_m, p95_pos_m, rmse_pos_m = stats(pos_errs)
    mean_pos_mm = mean_pos_m * MM
    med_pos_mm = med_pos_m * MM
    p95_pos_mm = p95_pos_m * MM
    rmse_pos_mm = rmse_pos_m * MM
    mse_pos_mm2 = (rmse_pos_m ** 2) * (MM ** 2)

    print("\n=== IMU vs Robot (position) ===")
    print(f"Aligned samples : {len(pos_errs)}")
    print(f"Mean |Δp|       : {mean_pos_mm:.2f} mm")
    print(f"Median |Δp|     : {med_pos_mm:.2f} mm")
    print(f"95th percentile : {p95_pos_mm:.2f} mm")
    print(f"MSE             : {mse_pos_mm2:.2f} mm^2")
    print(f"RMSE            : {rmse_pos_mm:.2f} mm")

    # --- Geodesic rotation error (deg) ---
    def rpy_to_R_deg(yaw_deg, pitch_deg, roll_deg):
        y = np.radians(yaw_deg)
        p = np.radians(pitch_deg)
        r = np.radians(roll_deg)
        cy, sy = np.cos(y), np.sin(y)
        cp, sp = np.cos(p), np.sin(p)
        cr, sr = np.cos(r), np.sin(r)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        return Rz @ Ry @ Rx

    geo_angles = []
    for _, row in merged.iterrows():
        try:
            R_robot = pose_row_to_matrix(row)[:3, :3]
            R_imu_raw = rpy_to_R_deg(
                row.get("imu_yaw_deg"),
                row.get("imu_pitch_deg"),
                row.get("imu_roll_deg"),
            )
            if R_cal is not None:
                R_imu = R_cal @ R_imu_raw
            else:
                R_imu = R_imu_raw
            ang = rot_geodesic_angle_deg(R_robot, R_imu)
            geo_angles.append(ang)
        except Exception:
            geo_angles.append(float("nan"))

    geo = np.array(geo_angles)
    geo = geo[~np.isnan(geo)]
    if geo.size:
        mean_g, med_g, p95_g, rmse_g = stats(geo)
        print("\n" + "=" * 72)
        print("=== IMU Orientation vs Robot (Geodesic Angle, Degrees) ===")
        print("=" * 72)
        if R_cal is not None:
            print("Using calibrated IMU→Robot rotation (R_cal).")
        else:
            print("No IMU→Robot calibration found — using raw IMU orientation.")

        print(f"\nAligned samples : {geo.size}")
        print(f"Mean error      : {mean_g:.3f}°")
        print(f"Median error    : {med_g:.3f}°")
        print(f"95th percentile : {p95_g:.3f}°")
        print(f"RMS error       : {rmse_g:.3f}°")
        print("=" * 72)


# ---------------------------------------------------------------------
# CSV selection
# ---------------------------------------------------------------------
def choose_csv_interactively(csv_dir: str) -> str:
    # Ensure directory exists before globbing
    if not os.path.exists(csv_dir):
        print(f"[ERROR] CSV directory not found: {os.path.abspath(csv_dir)}")
        # Fallback to local if data/CSV fails? Or just crash gracefully
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    pattern = os.path.join(csv_dir, "*.csv")
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
    # UPDATED: Defaults are now relative to the project root
    parser.add_argument(
        "--transform",
        default=DEFAULT_TRANSFORM_PATH,
        help="Optional path to a cam->robot transform .npz."
    )
    parser.add_argument(
        "--transform-dir",
        default=DEFAULT_TRANSFORM_DIR,
        help="Directory with per-camera transforms"
    )
    parser.add_argument("--csv", help="Path to a specific CSV file")
    parser.add_argument(
        "--csv-dir",
        default=DEFAULT_CSV_DIR,
        help="Directory with CSV/session_log_*.csv"
    )

    parser.add_argument("--time-col", default="timestamp")
    parser.add_argument("--cam-col", default="source")
    parser.add_argument("--tag-col", default="tag_id")
    parser.add_argument("--time-tol", type=float, default=0.02)
    parser.add_argument("--tag-id", type=int)
    parser.add_argument("--robot-source-name", default="robot")

    args = parser.parse_args()

    # ---------------------------
    # Choose ONE CSV
    # ---------------------------
    if args.csv:
        csv_file = args.csv
    else:
        try:
            csv_file = choose_csv_interactively(args.csv_dir)
        except Exception as e:
            print(f"[CRITICAL] {e}")
            return

    print("\n##############################")
    print(f"[INFO] Processing CSV: {csv_file}")
    print("##############################")

    # ---------------------------
    # Load CSV
    # ---------------------------
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV {csv_file}: {e}")
        return

    # Convert timestamp
    if args.time_col in df.columns:
        if not np.issubdtype(df[args.time_col].dtype, np.number):
            try:
                df[args.time_col] = pd.to_datetime(df[args.time_col])
            except Exception:
                pass

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
        # We can still attempt IMU analysis below even if cams fail
    else:
        # Camera-vs-camera
        if len(cam_ids) >= 2:
            for i in range(len(cam_ids)):
                for j in range(i + 1, len(cam_ids)):
                    analyze_camera_pair(
                        df, cam_ids[i], cam_ids[j],
                        T_cam_to_robot,
                        time_tol=args.time_tol, time_col=args.time_col,
                        cam_col=args.cam_col, tag_col=args.tag_col,
                        tag_id_filter=args.tag_id,
                    )

        # Camera-vs-robot
        has_robot = (df[args.cam_col] == robot_name).any()
        if has_robot:
            for cam_id in cam_ids:
                analyze_camera_vs_robot(
                    df, cam_id,
                    T_cam_to_robot,
                    time_tol=args.time_tol, time_col=args.time_col,
                    cam_col=args.cam_col, tag_col=args.tag_col,
                    robot_source_name=robot_name,
                    tag_id_filter=args.tag_id,
                )

    # IMU vs robot analysis
    imu_source_candidates = [
        s for s in set(df[args.cam_col].astype(str).unique())
        if s.lower().startswith("imu") or s == "imu"
    ]
    if imu_source_candidates:
        imu_source = imu_source_candidates[0]
        analyze_imu_vs_robot(
            df,
            imu_source,
            robot_source_name=robot_name,
            time_tol=args.time_tol,
            time_col=args.time_col,
            transform_dir=args.transform_dir,
        )


if __name__ == "__main__":
    main()