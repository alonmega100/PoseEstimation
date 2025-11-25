#!/usr/bin/env python3
"""
Analyze camera performance by comparing positions from different cameras
after transforming them into the robot base frame.

For each CSV, and for each pair of cameras, we:
  1. Transform camera-frame points to robot base frame using hand-eye transforms.
  2. Time-align samples between the two cameras (per tag_id) using nearest-neighbor
     in time with a tolerance.
  3. Compute error vectors and MSE / stats of ||p_camA - p_camB||.

Usage examples
--------------
# Use defaults (CSV dir, transform dir, column names)
python camera_performance_analysis.py

# Analyze a specific CSV file
python camera_performance_analysis.py --csv CSV/session_log_0001.csv

# Change time tolerance to 20 ms and focus on tag 1
python camera_performance_analysis.py --time-tol 0.02 --tag-id 1

# Use a specific transform directory
python camera_performance_analysis.py --transform-dir DATA/hand_eye
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd


DEFAULT_TRANSFORM_PATH = None  # optional global file (for backwards compat)
DEFAULT_TRANSFORM_DIR = "DATA/hand_eye"
DEFAULT_CSV_DIR = "CSV"


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
# CSV handling (still here but we won't use it in the default path)
# ---------------------------------------------------------------------
def find_csv_files(csv_arg: Optional[str], csv_dir: str) -> List[str]:
    """
    If csv_arg is given, return [csv_arg].
    Otherwise, find all CSV/session_log_*.csv in csv_dir.
    (Kept for backwards compat; main() now uses "find_rot_tran"-style selection.)
    """
    if csv_arg:
        if not os.path.exists(csv_arg):
            raise FileNotFoundError(f"CSV file not found: {csv_arg}")
        return [csv_arg]

    pattern = os.path.join(csv_dir, "session_log_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found with pattern: {pattern}")
    print(f"[INFO] Found {len(files)} CSV files in {csv_dir}")
    return files


def apply_transform(T: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """
    Apply 4x4 homogeneous transform T to Nx3 points in camera frame.
    T maps cam -> base (i.e., p_base = T @ p_cam_h).
    """
    assert pts_cam.shape[1] == 3
    N = pts_cam.shape[0]
    homo = np.concatenate([pts_cam, np.ones((N, 1))], axis=1)  # (N,4)
    pts_base_h = homo @ T.T  # (N,4)
    return pts_base_h[:, :3]


# ---------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------
def analyze_pair(
    df: pd.DataFrame,
    cam_a: str,
    cam_b: str,
    T_cam_to_robot: Dict[str, np.ndarray],
    time_tol: float,
    time_col: str,
    cam_col: str,
    tag_col: str,
    cam_x_col: str,
    cam_y_col: str,
    cam_z_col: str,
    tag_id_filter: Optional[int] = None,
) -> None:
    """
    For a given DataFrame and pair of cameras, compute MSE between their
    robot-frame positions for temporally aligned samples.
    """
    df_pair = df.copy()

    # Optional: filter by tag_id
    if tag_id_filter is not None:
        df_pair = df_pair[df_pair[tag_col] == tag_id_filter]

    # Filter rows for cam_a and cam_b
    df_a = df_pair[df_pair[cam_col] == cam_a].copy()
    df_b = df_pair[df_pair[cam_col] == cam_b].copy()

    if df_a.empty or df_b.empty:
        print(f"[WARN] No data for camera pair ({cam_a}, {cam_b}) in this CSV.")
        return

    # Sort by time for merge_asof
    df_a = df_a.sort_values(time_col)
    df_b = df_b.sort_values(time_col)

    # Transform to robot frame
    if cam_a not in T_cam_to_robot or cam_b not in T_cam_to_robot:
        print(f"[WARN] Missing transform for one of ({cam_a}, {cam_b}), skipping.")
        return

    T_a = T_cam_to_robot[cam_a]
    T_b = T_cam_to_robot[cam_b]

    # We'll merge per-tag if tags exist
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

        # Align by time, per tag
        merged = pd.merge_asof(
            sub_a.sort_values(time_col),
            sub_b.sort_values(time_col),
            on=time_col,
            direction="nearest",
            tolerance=tol,
            suffixes=("_a", "_b"),
        )

        merged = merged.dropna(subset=[f"{cam_x_col}_b", f"{cam_y_col}_b", f"{cam_z_col}_b"])
        if merged.empty:
            continue

        # Extract camera-frame positions
        pts_a_cam = merged[[f"{cam_x_col}_a", f"{cam_y_col}_a", f"{cam_z_col}_a"]].to_numpy()
        pts_b_cam = merged[[f"{cam_x_col}_b", f"{cam_y_col}_b", f"{cam_z_col}_b"]].to_numpy()

        # Transform to robot base frame
        pts_a_base = apply_transform(T_a, pts_a_cam)
        pts_b_base = apply_transform(T_b, pts_b_cam)

        # Error vectors and norms
        diffs = pts_a_base - pts_b_base       # (N,3)
        dists = np.linalg.norm(diffs, axis=1) # (N,)

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

    print(f"\n=== Camera pair: {cam_a} vs {cam_b} ===")
    if tag_id_filter is not None:
        print(f"Tag filter: {tag_id_filter}")
    print(f"Aligned samples: {len(all_errors)}")
    print(f"Mean |Δp|   : {mean_err:.4f} m")
    print(f"Median |Δp| : {median_err:.4f} m")
    print(f"95th |Δp|   : {p95:.4f} m")
    print(f"MSE        : {mse:.6f} m^2")
    print(f"RMSE       : {rmse:.6f} m\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze camera performance by comparing robot-frame positions between cameras."
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
    parser.add_argument("--csv", help="Path to a specific CSV file (if omitted, use selection from CSV/)")
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
    # NEW DEFAULTS for your concurrent logger CSV:
    parser.add_argument(
        "--cam-col",
        default="source",
        help="Camera ID column name (default: 'source' – matching your CSV)",
    )
    parser.add_argument(
        "--tag-col",
        default="tag_id",
        help="Tag ID column name (default: 'tag_id')",
    )
    # We use the translation part of the 4x4 pose matrix: pose_03, pose_13, pose_23
    parser.add_argument(
        "--cam-x-col",
        default="pose_03",
        help="Camera-frame X column name (default: 'pose_03')",
    )
    parser.add_argument(
        "--cam-y-col",
        default="pose_13",
        help="Camera-frame Y column name (default: 'pose_13')",
    )
    parser.add_argument(
        "--cam-z-col",
        default="pose_23",
        help="Camera-frame Z column name (default: 'pose_23')",
    )

    parser.add_argument(
        "--time-tol",
        type=float,
        default=0.02,
        help="Max time difference (seconds) for matching points between cameras (default: 0.02)",
    )
    parser.add_argument(
        "--tag-id",
        type=int,
        help="Optional tag_id to filter on (default: use all tags)",
    )

    args = parser.parse_args()

    # ---------------------------
    # Choose ONE CSV, like find_rot_tran
    # ---------------------------
    if args.csv:
        csv_files = [args.csv]
    else:
        # Mimic the behavior from the "find rot tran" script:
        # list CSV/, sort, print, ask for index, Enter = last.
        csv_dir = args.csv_dir
        files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
        if not files:
            raise FileNotFoundError(f"No CSV files found in {csv_dir}")
        sorted_files = sorted(files)
        print("Available CSV files in CSV/:")
        for i, name in enumerate(sorted_files):
            print(f"  {i}: {name}")
        num = input(
            "File to load? 0 for the first and so on...\n"
            " Press Enter for the last one\n :"
        )
        if not num:
            num = -1
        num = int(num)
        chosen = sorted_files[num]
        print("You chose", chosen)
        csv_files = [os.path.join(csv_dir, chosen)]

    # Now process ONLY this one CSV
    for csv_file in csv_files:
        print(f"\n##############################")
        print(f"[INFO] Processing CSV: {csv_file}")
        print(f"##############################")

        df = pd.read_csv(csv_file)

        # Convert timestamp to datetime if it's not numeric
        if args.time_col in df.columns:
            if not np.issubdtype(df[args.time_col].dtype, np.number):
                try:
                    df[args.time_col] = pd.to_datetime(df[args.time_col])
                except Exception as e:
                    print(f"[WARN] Failed to parse {args.time_col} as datetime: {e}")

        required_cols = [
            args.time_col,
            args.cam_col,
            args.tag_col,
            args.cam_x_col,
            args.cam_y_col,
            args.cam_z_col,
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"[ERROR] CSV {csv_file} missing required columns: {missing}")
            print(
                "        Use --time-col / --cam-col / --tag-col / "
                "--cam-x-col / --cam-y-col / --cam-z-col to adapt."
            )
            continue

        # All cameras appearing in this CSV
        cam_ids_in_csv = sorted(set(df[args.cam_col].astype(str).unique()))

        try:
            T_cam_to_robot = load_cam_to_robot_transforms(
                args.transform, args.transform_dir, cam_ids_in_csv
            )
        except Exception as e:
            print(f"[ERROR] Could not load transforms for {csv_file}: {e}")
            continue

        # Keep only cameras that have a transform
        cam_ids = [cid for cid in cam_ids_in_csv if cid in T_cam_to_robot]

        if len(cam_ids) < 2:
            print(f"[WARN] Need at least 2 cameras with transforms; found {cam_ids}")
            continue

        # Analyze every pair (i < j)
        for i in range(len(cam_ids)):
            for j in range(i + 1, len(cam_ids)):
                cam_a = cam_ids[i]
                cam_b = cam_ids[j]
                analyze_pair(
                    df,
                    cam_a,
                    cam_b,
                    T_cam_to_robot,
                    time_tol=args.time_tol,
                    time_col=args.time_col,
                    cam_col=args.cam_col,
                    tag_col=args.tag_col,
                    cam_x_col=args.cam_x_col,
                    cam_y_col=args.cam_y_col,
                    cam_z_col=args.cam_z_col,
                    tag_id_filter=args.tag_id,
                )


if __name__ == "__main__":
    main()
