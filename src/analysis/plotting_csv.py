#!/usr/bin/env python3
import csv
import argparse
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
import os
import itertools
from src.utils.tools import (
    load_cam_to_robot_transforms, choose_csv_interactively,
    moving_average, find_latest_csv
)


def extract_points(csv_path):
    points = []
    # Sort CSV by timestamp to ensure lines connect in order
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Simple sort by timestamp string (ISO format sorts correctly)
    rows.sort(key=lambda r: r.get("timestamp", ""))

    for row in rows:
        src = row.get("source", "").strip()
        ts = row.get("timestamp", "")

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
    cam_ids = sorted(list(set(p["source"] for p in cams)))
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

    # 4. Plot Setup
    fig = go.Figure()

    # Generate distinct colors for each camera source
    color_cycle = itertools.cycle(pc.qualitative.Dark24)
    source_colors = {cid: next(color_cycle) for cid in cam_ids}
    source_colors["robot"] = "black"
    source_colors["imu"] = "blue"

    # Helper to add traces
    def add_group(kind_filter, name_suffix="", line_style="solid", opacity=1.0, symbol="circle", size=3):
        subset = [p for p in points if p["kind"] == kind_filter]
        if not subset: return

        unique_sources = sorted(list(set(p["source"] for p in subset)))

        for src in unique_sources:
            data = [p for p in subset if p["source"] == src]

            # Use assigned color, fallback to grey if unknown
            color = source_colors.get(src, "grey")

            fig.add_trace(go.Scatter3d(
                x=[p["x"] for p in data],
                y=[p["y"] for p in data],
                z=[p["z"] for p in data],
                mode="lines+markers",
                marker=dict(size=size, symbol=symbol, color=color, opacity=opacity),
                line=dict(color=color, dash=line_style, width=2 if line_style == "solid" else 1),
                name=f"{src} {name_suffix}",
                hovertext=[f"Time: {p['timestamp']}<br>Src: {src}" for p in data],
                hoverinfo="text+x+y+z"
            ))

    # --- Plot Layers ---

    # 1. Robot (Ground Truth) - Thick solid lines
    add_group("robot", "(Robot)", size=3)

    # 2. Raw Camera - Dashed, smaller, slightly transparent (Ghost)
    add_group("camera", "(Raw)", line_style="dash", opacity=1, size=3, symbol="circle")

    # 3. Aligned Camera - Solid, matching color to Raw
    add_group("camera_aligned", "(Aligned)", size=3, symbol="circle")

    # 4. IMU
    add_group("imu", "(IMU)", size=2)

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)"
        ),
        title=f"Session Analysis: {os.path.basename(csv_path)}",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    fig.show()


if __name__ == "__main__":
    main()