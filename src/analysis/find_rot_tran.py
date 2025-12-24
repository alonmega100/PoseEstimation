import numpy as np
import pandas as pd
import argparse
import os
import sys
from src.utils.tools import (
    ransac_rigid_transform, rigid_transform, nearest_merge,
    choose_csv_interactively, rpy_to_R_deg, rot_geodesic_angle_deg,
    pose_row_to_matrix
)
from src.utils.config import WORLD_TAG_ID, CAMERA_SERIALS

# Config
DEFAULT_CSV_DIR = "data/CSV"
SAVE_DIR = "data/DATA/hand_eye"
TIME_TOL = pd.Timedelta('30ms')


def pick_xyz(df_like, suffix=""):
    return df_like[[f"pose_03{suffix}", f"pose_13{suffix}", f"pose_23{suffix}"]].to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to CSV file")
    parser.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)
    args = parser.parse_args()

    # Load CSV
    if args.csv:
        csv_path = args.csv
    else:
        try:
            csv_path = choose_csv_interactively(args.csv_dir)
        except Exception as e:
            print(e)
            sys.exit(1)

    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    robot = df[df['source'] == 'robot']

    # ----------------------------------------
    # 1. Camera <-> Robot Calibration
    # ----------------------------------------
    all_sources = sorted(df["source"].astype(str).unique())
    cam_sources = [s for s in all_sources if s != "robot" and not s.lower().startswith("imu")]

    for src in cam_sources:
        cam_df = df[df["source"] == src]
        # Filter for world tag snapshots
        cam_tag = cam_df[(cam_df["event"] == "tag_pose_snapshot") & (cam_df["tag_id"] == WORLD_TAG_ID)]

        if cam_tag.empty:
            print(f"[{src}] No tag snapshots found for ID {WORLD_TAG_ID}")
            continue

        # Merge
        m = nearest_merge(cam_tag, robot, tol=TIME_TOL, suffix_cam="_cam", suffix_rob="_rob")
        m = m.dropna(subset=["pose_03_rob"])

        if m.empty:
            print(f"[{src}] No time-aligned matches found.")
            continue

        A = pick_xyz(m, "_cam")
        B = pick_xyz(m, "_rob")

        try:
            R, t, stats, _ = ransac_rigid_transform(A, B, random_state=42)
            print(f"\n=== {src} -> Robot Result ===")
            print(f"R=\n{R}\nt={t}")
            print(f"Stats: {stats}")

            save_path = os.path.join(SAVE_DIR, f"cam_{src}_to_robot_transform.npz")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, R=R, t=t, stats=stats, source=src)
            print(f"Saved to {save_path}")
        except Exception as e:
            print(f"[{src}] RANSAC failed: {e}")

    # ----------------------------------------
    # 2. IMU <-> Robot Orientation
    # ----------------------------------------
    imu_sources = [s for s in all_sources if s.lower().startswith("imu")]
    for src in imu_sources:
        imu_df = df[df["source"] == src]
        m = nearest_merge(imu_df, robot, tol=TIME_TOL, suffix_rob="_rob")

        # Drop invalid rows immediately
        req_cols = ["imu_yaw_deg", "imu_pitch_deg", "imu_roll_deg", "pose_00_rob"]
        if not set(req_cols).issubset(m.columns):
            continue

        m = m.dropna(subset=req_cols)

        if m.empty:
            print(f"[{src}] No valid IMU-Robot aligned samples.")
            continue

        # Build Rotation Lists
        R_rob_list, R_imu_list = [], []
        for _, row in m.iterrows():
            try:
                Rr = pose_row_to_matrix(row, "pose_")[:3, :3]  # Robot
                Ri = rpy_to_R_deg(row["imu_yaw_deg"], row["imu_pitch_deg"], row["imu_roll_deg"])

                # Check for NaNs/Infs in matrices
                if not (np.isfinite(Rr).all() and np.isfinite(Ri).all()):
                    continue

                R_rob_list.append(Rr)
                R_imu_list.append(Ri)
            except Exception:
                continue

        if not R_rob_list:
            print(f"[{src}] No valid rotation matrices could be constructed.")
            continue

        # Procrustes for Rotation
        M = np.zeros((3, 3))
        for Rr, Ri in zip(R_rob_list, R_imu_list):
            M += Rr @ Ri.T

        # Validate M before SVD
        if not np.isfinite(M).all():
            print(f"[{src}] Error: Accumulated matrix M contains NaNs or Infs. Skipping.")
            continue

        try:
            U, _, Vt = np.linalg.svd(M)
        except np.linalg.LinAlgError as e:
            print(f"[{src}] SVD failed to converge: {e}")
            continue

        R_cal = U @ Vt
        if np.linalg.det(R_cal) < 0:
            U[:, -1] *= -1
            R_cal = U @ Vt

        print(f"\n=== IMU {src} -> Robot Orientation ===")
        print(f"R_cal=\n{R_cal}")

        save_path = os.path.join(SAVE_DIR, f"imu_{src}_to_robot_transform.npz")
        np.savez(save_path, R=R_cal, t=np.zeros(3), source=src)
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()