#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse

from src.utils.tools import (
    load_cam_to_robot_transforms,
    choose_csv_interactively,
    pose_row_to_matrix,
    rot_geodesic_angle_deg,
)
from src.utils.config import DEFAULT_TRANSFORM_DIR, DEFAULT_CSV_DIR, CAMERA_SERIALS

MM = 1000.0


def inv_T(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 homogeneous transform."""
    return np.linalg.inv(T)


def add_camera_base_columns(df_cam: pd.DataFrame, cam_id: str, T_cam_to_robot: dict) -> pd.DataFrame:
    """Add x_base/y_base/z_base columns by transforming each tag pose into the robot/base frame."""
    T_base_cam = T_cam_to_robot[cam_id]
    xyz = []
    for _, row in df_cam.iterrows():
        T_cam_tag = pose_row_to_matrix(row)
        p = (T_base_cam @ T_cam_tag)[:3, 3]
        xyz.append(p)
    xyz = np.asarray(xyz)
    df_cam = df_cam.copy()
    df_cam["x_base"], df_cam["y_base"], df_cam["z_base"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return df_cam


def add_camera_base_rotation_columns(df_cam: pd.DataFrame, cam_id: str, T_cam_to_robot: dict) -> pd.DataFrame:
    """Add R_base_* columns (flattened 3x3) for optional cam-to-cam rotation comparisons."""
    T_base_cam = T_cam_to_robot[cam_id]
    Rs = []
    for _, row in df_cam.iterrows():
        T_cam_tag = pose_row_to_matrix(row)
        R = (T_base_cam @ T_cam_tag)[:3, :3]
        Rs.append(R.reshape(-1))
    Rs = np.asarray(Rs)
    df_cam = df_cam.copy()
    for i in range(9):
        df_cam[f"R_base_{i}"] = Rs[:, i]
    return df_cam


def analyze_camera_vs_robot(df: pd.DataFrame, cam_id: str, T_cam_to_robot: dict, time_tol: float, robot_name: str = "robot") -> None:
    df_cam = df[df["source"] == cam_id].copy()
    df_rob = df[df["source"] == robot_name].copy()

    if df_cam.empty or df_rob.empty:
        return

    # Transform Camera to Base
    df_cam = add_camera_base_columns(df_cam, cam_id, T_cam_to_robot)

    # Robot is already Base (extract x,y,z from pose matrix)
    xyz_r = []
    for _, row in df_rob.iterrows():
        xyz_r.append(pose_row_to_matrix(row)[:3, 3])
    xyz_r = np.asarray(xyz_r)
    df_rob["x_base_rob"], df_rob["y_base_rob"], df_rob["z_base_rob"] = xyz_r[:, 0], xyz_r[:, 1], xyz_r[:, 2]

    # Merge
    tol = pd.Timedelta(seconds=time_tol)
    merged = pd.merge_asof(
        df_cam.sort_values("timestamp"),
        df_rob.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=tol,
    ).dropna(subset=["x_base", "x_base_rob"])

    if merged.empty:
        return

    # Error
    p_cam = merged[["x_base", "y_base", "z_base"]].to_numpy()
    p_rob = merged[["x_base_rob", "y_base_rob", "z_base_rob"]].to_numpy()
    dists = np.linalg.norm(p_cam - p_rob, axis=1) * MM

    print(f"\n=== {cam_id} vs Robot ===")
    print(f"Samples: {len(dists)}")
    print(f"Mean Error: {np.mean(dists):.2f} mm")
    print(f"Std Error:  {np.std(dists):.2f} mm")
    print(f"RMSE:       {np.sqrt(np.mean(dists ** 2)):.2f} mm")


def analyze_camera_vs_camera(df: pd.DataFrame, cam_a: str, cam_b: str, T_cam_to_robot: dict, time_tol: float) -> None:
    """Cam-to-cam comparison AFTER transforming both cams into the same (robot/base) frame."""
    df_a = df[df["source"] == cam_a].copy()
    df_b = df[df["source"] == cam_b].copy()
    if df_a.empty or df_b.empty:
        print(f"[WARN] Cam-to-cam skipped: missing data for {cam_a if df_a.empty else ''}{' and ' if df_a.empty and df_b.empty else ''}{cam_b if df_b.empty else ''}")
        return

    # Transform both to base: position + (optional) rotation
    df_a = add_camera_base_columns(df_a, cam_a, T_cam_to_robot)
    df_b = add_camera_base_columns(df_b, cam_b, T_cam_to_robot)
    df_a = add_camera_base_rotation_columns(df_a, cam_a, T_cam_to_robot)
    df_b = add_camera_base_rotation_columns(df_b, cam_b, T_cam_to_robot)

    tol = pd.Timedelta(seconds=time_tol)
    merged = pd.merge_asof(
        df_a.sort_values("timestamp"),
        df_b.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=tol,
        suffixes=("_a", "_b"),
    )

    # Ensure we have successful matches
    merged = merged.dropna(subset=["x_base_a", "x_base_b"])
    if merged.empty:
        print(f"[WARN] Cam-to-cam: no timestamp matches within {time_tol}s for {cam_a} <-> {cam_b}")
        return

    # Position error in base
    p_a = merged[["x_base_a", "y_base_a", "z_base_a"]].to_numpy()
    p_b = merged[["x_base_b", "y_base_b", "z_base_b"]].to_numpy()
    dists_mm = np.linalg.norm(p_a - p_b, axis=1) * MM

    # Rotation error in base (geodesic angle)
    angs_deg = []
    for _, row in merged.iterrows():
        Ra = np.array([row[f"R_base_{i}_a"] for i in range(9)]).reshape(3, 3)
        Rb = np.array([row[f"R_base_{i}_b"] for i in range(9)]).reshape(3, 3)
        angs_deg.append(rot_geodesic_angle_deg(Ra, Rb))
    angs_deg = np.asarray(angs_deg)

    print(f"\n=== {cam_a} vs {cam_b} (cam-to-cam in Base, after transform) ===")
    print(f"Samples: {len(dists_mm)}")
    print(f"Mean Position Δ: {np.mean(dists_mm):.2f} mm")
    print(f"Std  Position Δ: {np.std(dists_mm):.2f} mm")
    print(f"RMSE Position Δ: {np.sqrt(np.mean(dists_mm ** 2)):.2f} mm")
    print(f"Mean Rotation Δ: {np.mean(angs_deg):.3f} deg")
    print(f"Std  Rotation Δ: {np.std(angs_deg):.3f} deg")
    print(f"RMSE Rotation Δ: {np.sqrt(np.mean(angs_deg ** 2)):.3f} deg")


def analyze_camera_vs_camera_pre_transform(
    df: pd.DataFrame,
    cam_a: str,
    cam_b: str,
    T_cam_to_robot: dict,
    time_tol: float,
) -> None:
    """Cam-to-cam comparison in camera frames (pre-transform).

    We use the known extrinsics to map cam_a tag poses into cam_b frame, then compare
    to cam_b's observed tag pose (position + rotation).
    """
    df_a = df[df["source"] == cam_a].copy()
    df_b = df[df["source"] == cam_b].copy()
    if df_a.empty or df_b.empty:
        print(
            f"[WARN] Pre-transform cam-to-cam skipped: missing data for {cam_a if df_a.empty else ''}"
            f"{' and ' if df_a.empty and df_b.empty else ''}{cam_b if df_b.empty else ''}"
        )
        return

    # Relative transform taking points/poses in cam_a frame into cam_b frame.
    # T_base_camX is actually T_camX_to_robot (i.e., T_base_camX).
    T_base_a = T_cam_to_robot[cam_a]
    T_base_b = T_cam_to_robot[cam_b]
    T_b_a = inv_T(T_base_b) @ T_base_a

    tol = pd.Timedelta(seconds=time_tol)
    merged = pd.merge_asof(
        df_a.sort_values("timestamp"),
        df_b.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=tol,
        suffixes=("_a", "_b"),
    )

    # Ensure we have successful matches
    merged = merged.dropna(subset=["pose_00_a", "pose_00_b"])
    if merged.empty:
        print(f"[WARN] Pre-transform cam-to-cam: no timestamp matches within {time_tol}s for {cam_a} <-> {cam_b}")
        return

    dists_mm = []
    angs_deg = []
    for _, row in merged.iterrows():
        T_a_tag = pose_row_to_matrix(row, prefix="pose_") if "pose_00_a" not in row else pose_row_to_matrix(row, prefix="pose_")
        # The merged row contains both _a and _b columns, but pose_row_to_matrix expects exact keys.
        # Build tag transforms explicitly from suffixed columns.
        def row_to_T(r, suf):
            H = np.eye(4, dtype=float)
            for rr in range(4):
                for cc in range(4):
                    key = f"pose_{rr}{cc}{suf}"
                    if key in r:
                        H[rr, cc] = float(r[key])
            return H

        T_a_tag = row_to_T(row, "_a")
        T_b_tag_obs = row_to_T(row, "_b")
        T_b_tag_pred = T_b_a @ T_a_tag

        p_pred = T_b_tag_pred[:3, 3]
        p_obs = T_b_tag_obs[:3, 3]
        dists_mm.append(float(np.linalg.norm(p_pred - p_obs) * MM))

        R_pred = T_b_tag_pred[:3, :3]
        R_obs = T_b_tag_obs[:3, :3]
        angs_deg.append(float(rot_geodesic_angle_deg(R_pred, R_obs)))

    dists_mm = np.asarray(dists_mm)
    angs_deg = np.asarray(angs_deg)

    print(f"\n=== {cam_a} vs {cam_b} (cam-to-cam in Camera frames, pre-transform) ===")
    print(f"Samples: {len(dists_mm)}")
    print(f"Mean Position Δ: {np.mean(dists_mm):.2f} mm")
    print(f"Std  Position Δ: {np.std(dists_mm):.2f} mm")
    print(f"RMSE Position Δ: {np.sqrt(np.mean(dists_mm ** 2)):.2f} mm")
    print(f"Mean Rotation Δ: {np.mean(angs_deg):.3f} deg")
    print(f"Std  Rotation Δ: {np.std(angs_deg):.3f} deg")
    print(f"RMSE Rotation Δ: {np.sqrt(np.mean(angs_deg ** 2)):.3f} deg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to CSV")
    parser.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)
    parser.add_argument("--transform-dir", default=DEFAULT_TRANSFORM_DIR)
    parser.add_argument("--time-tol", type=float, default=0.03)
    args = parser.parse_args()

    csv_path = args.csv if args.csv else choose_csv_interactively(args.csv_dir)

    print(f"Analyzing {csv_path}...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    cam_ids_in_csv = [s for s in df["source"].unique() if s != "robot" and not str(s).startswith("imu")]

    # Load transforms using shared tool
    transforms = load_cam_to_robot_transforms(args.transform_dir)

    # Per-camera vs robot
    for cid in cam_ids_in_csv:
        if cid in transforms:
            analyze_camera_vs_robot(df, cid, transforms, args.time_tol)
        else:
            print(f"[WARN] No transform for {cid}")

    # Cam-to-cam check (ONLY if CAMERA_SERIALS contains exactly 2 cameras)
    if len(CAMERA_SERIALS) == 2:
        cam_a, cam_b = CAMERA_SERIALS[0], CAMERA_SERIALS[1]

        missing = [c for c in (cam_a, cam_b) if c not in transforms]
        if missing:
            print(f"[WARN] Cam-to-cam skipped: missing transform(s) for {missing}")
            return

        missing_csv = [c for c in (cam_a, cam_b) if c not in cam_ids_in_csv]
        if missing_csv:
            print(f"[WARN] Cam-to-cam skipped: missing camera source(s) in CSV: {missing_csv}")
            return

        # Pre-transform cam-to-cam (compare in camera frames using relative extrinsics)
        analyze_camera_vs_camera_pre_transform(df, cam_a, cam_b, transforms, args.time_tol)

        # After-transform cam-to-cam (compare in base frame)
        analyze_camera_vs_camera(df, cam_a, cam_b, transforms, args.time_tol)
    else:
        print(f"[INFO] Cam-to-cam check skipped: CAMERA_SERIALS must contain exactly 2 cameras (got {len(CAMERA_SERIALS)}).")


if __name__ == "__main__":
    main()