#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import os
from src.utils.tools import (
    load_cam_to_robot_transforms, load_imu_to_robot_transform,
    choose_csv_interactively, pose_row_to_matrix,
    rot_geodesic_angle_deg, rpy_to_R_deg
)
from src.utils.config import DEFAULT_TRANSFORM_DIR, DEFAULT_CSV_DIR

MM = 1000.0


def add_camera_base_columns(df_cam, cam_id, T_cam_to_robot):
    T_base_cam = T_cam_to_robot[cam_id]
    xyz = []
    for _, row in df_cam.iterrows():
        T_cam_tag = pose_row_to_matrix(row)
        p = (T_base_cam @ T_cam_tag)[:3, 3]
        xyz.append(p)
    xyz = np.array(xyz)
    df_cam = df_cam.copy()
    df_cam["x_base"], df_cam["y_base"], df_cam["z_base"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return df_cam


def analyze_camera_vs_robot(df, cam_id, T_cam_to_robot, time_tol, robot_name="robot"):
    df_cam = df[df["source"] == cam_id].copy()
    df_rob = df[df["source"] == robot_name].copy()

    if df_cam.empty or df_rob.empty: return

    # Transform Camera to Base
    df_cam = add_camera_base_columns(df_cam, cam_id, T_cam_to_robot)

    # Robot is already Base (extract x,y,z from pose matrix)
    xyz_r = []
    for _, row in df_rob.iterrows():
        xyz_r.append(pose_row_to_matrix(row)[:3, 3])
    xyz_r = np.array(xyz_r)
    df_rob["x_base_rob"], df_rob["y_base_rob"], df_rob["z_base_rob"] = xyz_r[:, 0], xyz_r[:, 1], xyz_r[:, 2]

    # Merge
    tol = pd.Timedelta(seconds=time_tol)
    merged = pd.merge_asof(
        df_cam.sort_values("timestamp"), df_rob.sort_values("timestamp"),
        on="timestamp", direction="nearest", tolerance=tol
    ).dropna(subset=["x_base", "x_base_rob"])

    if merged.empty: return

    # Error
    p_cam = merged[["x_base", "y_base", "z_base"]].to_numpy()
    p_rob = merged[["x_base_rob", "y_base_rob", "z_base_rob"]].to_numpy()
    dists = np.linalg.norm(p_cam - p_rob, axis=1) * MM

    print(f"\n=== {cam_id} vs Robot ===")
    print(f"Samples: {len(dists)}")
    print(f"Mean Error: {np.mean(dists):.2f} mm")
    print(f"RMSE:       {np.sqrt(np.mean(dists ** 2)):.2f} mm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to CSV")
    parser.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)
    parser.add_argument("--transform-dir", default=DEFAULT_TRANSFORM_DIR)
    parser.add_argument("--time-tol", type=float, default=0.03)
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        csv_path = choose_csv_interactively(args.csv_dir)

    print(f"Analyzing {csv_path}...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    cam_ids = [s for s in df["source"].unique() if s != "robot" and not str(s).startswith("imu")]

    # Load Transforms using Shared Tool
    transforms = load_cam_to_robot_transforms(None, args.transform_dir, cam_ids)

    for cid in cam_ids:
        if cid in transforms:
            analyze_camera_vs_robot(df, cid, transforms, args.time_tol)
        else:
            print(f"[WARN] No transform for {cid}")


if __name__ == "__main__":
    main()