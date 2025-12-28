#!/usr/bin/env python3
import csv
import argparse
import numpy as np
import plotly.graph_objects as go
import os
from src.utils.config import DEFAULT_TRANSFORM_DIR
from src.utils.tools import (
    load_cam_to_robot_transforms, choose_csv_interactively,
    moving_average, find_latest_csv
)


# -----------------------------------------------------------
# Visual Style Helpers (From your original script)
# -----------------------------------------------------------
def get_color(i):
    # The exact color cycle from your original script
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
              "#bcbd22", "#17becf"]
    return colors[i % len(colors)]


def extract_points(csv_path):
    points = []
    # Read all rows first to sort by timestamp (ensures lines connect correctly)
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # ISO timestamp strings sort correctly
    rows.sort(key=lambda r: r.get("timestamp", ""))

    for i, row in enumerate(rows):
        src = row.get("source", "").strip()
        ts = row.get("timestamp", "")

        # Helper dict structure
        p = {
            "timestamp": ts,
            "source": src,
            "_idx": i  # Keep original index for stability
        }

        # IMU
        if src == "imu" and row.get("imu_x"):
            try:
                p.update({
                    "kind": "imu",
                    "x": float(row["imu_x"]), "y": float(row["imu_y"]), "z": float(row["imu_z"])
                })
                points.append(p)
            except ValueError:
                pass

        # Pose (Robot or Camera)
        elif "pose_03" in row and row["pose_03"]:
            kind = "robot" if src == "robot" else "camera"
            try:
                p.update({
                    "kind": kind,
                    "x": float(row["pose_03"]), "y": float(row["pose_13"]), "z": float(row["pose_23"])
                })
                points.append(p)
            except ValueError:
                pass

    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--transform-dir", default=DEFAULT_TRANSFORM_DIR)
    parser.add_argument("--imu-smoothing", type=int, default=1)
    parser.add_argument("--only-aligned", action="store_true", help="Hide raw camera traces")
    # Toggle lines (defaults from your script)
    parser.add_argument("--no-robot-lines", action="store_true")
    parser.add_argument("--no-camera-lines", action="store_true")
    args = parser.parse_args()

    # 1. Select CSV
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = find_latest_csv("data/CSV") or choose_csv_interactively("data/CSV")

    print(f"Plotting: {csv_path}")
    points = extract_points(csv_path)
    #
    # # 2. Smooth IMU
    # if args.imu_smoothing > 1:
    #     imus = [p for p in points if p["kind"] == "imu"]
    #     if imus:
    #         smooth_x = moving_average([p["x"] for p in imus], args.imu_smoothing)
    #         smooth_y = moving_average([p["y"] for p in imus], args.imu_smoothing)
    #         smooth_z = moving_average([p["z"] for p in imus], args.imu_smoothing)
    #         for i, p in enumerate(imus):
    #             p["x"], p["y"], p["z"] = smooth_x[i], smooth_y[i], smooth_z[i]

    # 3. Transform Cameras
    # Note: We filter out non-camera points later based on --only-aligned if needed
    cams = [p for p in points if p["kind"] == "camera"]
    transforms = load_cam_to_robot_transforms(args.transform_dir)


    aligned_points = []
    for p in cams:
        # print("for p in cams")
        if p["source"] in transforms:

            T = transforms[p["source"]]
            v = np.array([p["x"], p["y"], p["z"], 1.0])
            v_new = T @ v
            new_p = p.copy()
            new_p["kind"] = "camera_aligned"
            new_p["x"], new_p["y"], new_p["z"] = v_new[0], v_new[1], v_new[2]
            aligned_points.append(new_p)

    # Extend points list
    points.extend(aligned_points)

    # 4. Plot (Replicating exact visual style)
    fig = go.Figure()

    # --- Helper to add traces with your specific visual logic ---
    def add_traces_for_kind(kind_filter, base_label, offset=0, size=3, connect_lines=True):
        subset = [p for p in points if p["kind"] == kind_filter]
        if not subset: return

        # Group by source to draw separate lines
        # Preserving the order of appearance or simple sort
        unique_sources = sorted(list(set(p["source"] for p in subset)))

        for i, src in enumerate(unique_sources):
            # VISUAL MATCH: Use the specific color cycle with offset
            color = get_color(i + offset)

            data = [p for p in subset if p["source"] == src]
            print("printing data once")


            # VISUAL MATCH: specific hovertemplate from your script
            fig.add_trace(go.Scatter3d(
                x=[p["x"] for p in data],
                y=[p["y"] for p in data],
                z=[p["z"] for p in data],
                mode="markers+lines" if connect_lines else "markers",
                marker=dict(size=size, color=color),
                line=dict(width=2, color=color),  # Solid lines, matching color
                name=f"{base_label} {src}",
                customdata=[p["timestamp"] for p in data],
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>t=%{customdata}<extra>%{fullData.name}</extra>"
            ))

    # --- 1. Robot ---
    # Your script implicitly gave Robot the default first color (blue) or explicit logic?
    # Your script used: fig.add_trace(..., name="Robot") without explicit color -> Plotly default blue.
    # We will replicate that or use explicit black if you prefer distinction.
    # Based on "THE SAME VISUALS", your script likely had Robot as Blue (#1f77b4).
    robots = [p for p in points if p["kind"] == "robot"]
    if robots:
        fig.add_trace(go.Scatter3d(
            x=[p["x"] for p in robots],
            y=[p["y"] for p in robots],
            z=[p["z"] for p in robots],
            mode="markers+lines" if not args.no_robot_lines else "markers",
            marker=dict(size=4), line=dict(width=2), name="Robot",
            customdata=[p["timestamp"] for p in robots],
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>t=%{customdata}<extra>Robot</extra>"
        ))

    # --- 2. Raw Cameras ---
    if not args.only_aligned:
        # Offset 0, Size 3
        add_traces_for_kind("camera", "Camera", offset=0, size=3, connect_lines=not args.no_camera_lines)

    # --- 3. Aligned Cameras ---

    add_traces_for_kind("camera_aligned", "Camera (aligned)", offset=5, size=4, connect_lines=not args.no_camera_lines)

    # Save and show
    html_path = os.path.splitext(csv_path)[0] + ".html"
    fig.write_html(html_path)
    print(f"Saved plot to {html_path}")
    fig.show()


if __name__ == "__main__":
    main()