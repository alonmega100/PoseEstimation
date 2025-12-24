#!/usr/bin/env python3
import csv
import argparse
import numpy as np
import plotly.graph_objects as go
import os
from src.utils.tools import (
    load_cam_to_robot_transforms, choose_csv_interactively,
    moving_average, find_latest_csv
)


def extract_points(csv_path):
    points = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("source", "").strip()
            ts = row.get("timestamp")

            # IMU
            if src == "imu" and row.get("imu_x"):
                points.append({
                    "kind": "imu", "source": src, "timestamp": ts,
                    "x": float(row["imu_x"]), "y": float(row["imu_y"]), "z": float(row["imu_z"])
                })
            # Pose (Robot or Camera)
            elif "pose_03" in row and row["pose_03"]:
                kind = "robot" if src == "robot" else "camera"
                points.append({
                    "kind": kind, "source": src, "timestamp": ts,
                    "x": float(row["pose_03"]), "y": float(row["pose_13"]), "z": float(row["pose_23"])
                })
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--transform-dir", default="data/DATA/hand_eye")
    parser.add_argument("--imu-smoothing", type=int, default=1)
    args = parser.parse_args()

    # 1. Select CSV
    if args.csv:
        csv_path = args.csv
    else:
        # Fallback to latest auto or interactive?
        # For plotting, users often want the last run.
        csv_path = find_latest_csv("data/CSV") or choose_csv_interactively("data/CSV")

    print(f"Plotting: {csv_path}")
    points = extract_points(csv_path)

    # 2. Smooth IMU
    if args.imu_smoothing > 1:
        imus = [p for p in points if p["kind"] == "imu"]
        if imus:
            smooth_x = moving_average([p["x"] for p in imus], args.imu_smoothing)
            smooth_y = moving_average([p["y"] for p in imus], args.imu_smoothing)
            smooth_z = moving_average([p["z"] for p in imus], args.imu_smoothing)
            for i, p in enumerate(imus):
                p["x"], p["y"], p["z"] = smooth_x[i], smooth_y[i], smooth_z[i]

    # 3. Transform Cameras
    cams = [p for p in points if p["kind"] == "camera"]
    cam_ids = list(set(p["source"] for p in cams))
    transforms = load_cam_to_robot_transforms(None, args.transform_dir, cam_ids)

    aligned_points = []
    for p in cams:
        if p["source"] in transforms:
            T = transforms[p["source"]]
            v = np.array([p["x"], p["y"], p["z"], 1.0])
            v_new = T @ v
            new_p = p.copy()
            new_p["kind"] = "camera_aligned"
            new_p["x"], new_p["y"], new_p["z"] = v_new[0], v_new[1], v_new[2]
            aligned_points.append(new_p)

    points.extend(aligned_points)

    # 4. Plot (Simplified)
    fig = go.Figure()

    def add_trace(kind, color, size=3):
        subset = [p for p in points if p["kind"] == kind]
        if not subset: return
        # Group by source
        sources = set(p["source"] for p in subset)
        for src in sources:
            data = [p for p in subset if p["source"] == src]
            fig.add_trace(go.Scatter3d(
                x=[p["x"] for p in data], y=[p["y"] for p in data], z=[p["z"] for p in data],
                mode="markers", marker=dict(size=size, color=color),
                name=f"{kind} ({src})"
            ))

    add_trace("robot", "black", 4)
    add_trace("camera", "red", 2)
    add_trace("camera_aligned", "green", 3)
    add_trace("imu", "blue", 2)

    fig.update_layout(scene=dict(aspectmode="data"), title=f"Session: {os.path.basename(csv_path)}")
    fig.show()


if __name__ == "__main__":
    main()