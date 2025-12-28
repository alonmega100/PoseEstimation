#!/usr/bin/env python3
import argparse
import csv
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from src.utils.tools import moving_average, H_to_xyzrpy_ZYX, pose_row_to_matrix, find_latest_csv

MOVING_AVG_WINDOW = 200  # number of samples in the moving average window


# ---------------------------------------------------------------
# Load IMU rows
# ---------------------------------------------------------------
def load_imu_from_csv(csv_path):
    t = []
    ax, ay, az = [], [], []
    yaw, pitch, roll = [], [], []

    # Robust reading using csv.DictReader
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("source") != "imu":
                continue

            try:
                # --- 1. Extract JSON Raw Data (if available) ---
                raw = row.get("raw_data") or ""
                data = {}
                if raw:
                    try:
                        data = json.loads(raw)
                    except:
                        pass

                # --- 2. Get Timing ---
                # Prefer 't_sec' from raw data (relative time), fallback to empty
                t_val = None
                if "t_sec" in data:
                    t_val = float(data["t_sec"])
                elif row.get("timestamp"):
                    # Fallback logic if needed, but usually we rely on t_sec for IMU plotting
                    pass

                if t_val is None:
                    continue

                # --- 3. Get Orientation ---
                # Try flattened columns first
                r_yaw = row.get("yaw")
                r_pitch = row.get("pitch")
                r_roll = row.get("roll")

                # If empty string, try JSON keys
                y = float(r_yaw) if r_yaw and r_yaw != "" else data.get("yaw")
                p = float(r_pitch) if r_pitch and r_pitch != "" else data.get("pitch")
                r = float(r_roll) if r_roll and r_roll != "" else data.get("roll")

                # --- 4. Get Acceleration ---
                # Try flattened columns first
                r_ax = row.get("acc_x")
                r_ay = row.get("acc_y")
                r_az = row.get("acc_z")

                if r_ax and r_ax != "":
                    cx, cy, cz = float(r_ax), float(r_ay), float(r_az)
                # Fallback: check 'accel' tuple in JSON
                elif "accel" in data and data["accel"]:
                    cx, cy, cz = data["accel"]
                # Fallback: check separate keys in JSON
                elif "acc_x" in data:
                    cx, cy, cz = float(data["acc_x"]), float(data["acc_y"]), float(data["acc_z"])
                else:
                    # No accel data available
                    cx, cy, cz = float("nan"), float("nan"), float("nan")

                if y is None or p is None or r is None:
                    continue

                t.append(t_val)
                yaw.append(y)
                pitch.append(p)
                roll.append(r)
                ax.append(cx)
                ay.append(cy)
                az.append(cz)

            except Exception:
                continue

    return t, ax, ay, az, yaw, pitch, roll


# ---------------------------------------------------------------
# Load robot pose rows (for comparison)
# ---------------------------------------------------------------
def load_robot_pose_from_csv(csv_path):
    """Extract robot RPY angles from pose matrices in CSV using pandas."""
    POSE_PREFIX = "pose_"

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return [], [], [], []

    if "source" not in df.columns:
        return [], [], [], []

    df_robot = df[df["source"] == "robot"].copy()

    if df_robot.empty:
        return [], [], [], []

    # Select time and pose columns
    time_col = "timestamp"
    pose_cols = [f"{POSE_PREFIX}{r}{c}" for r in range(4) for c in range(4)]

    if time_col not in df_robot.columns:
        return [], [], [], []

    if not all(col in df_robot.columns for col in pose_cols):
        return [], [], [], []

    df_robot = df_robot[[time_col] + pose_cols].dropna()

    if df_robot.empty:
        return [], [], [], []

    # Convert timestamp to seconds
    try:
        df_robot[time_col] = pd.to_datetime(df_robot[time_col]).astype(int) / 1e9
    except:
        return [], [], [], []

    # Extract RPY from pose matrices
    def extract_rpy(row):
        try:
            H = pose_row_to_matrix(row, prefix=POSE_PREFIX)
            xyzrpy = H_to_xyzrpy_ZYX(H)
            # Returns [x, y, z, roll, pitch, yaw]
            return np.degrees(xyzrpy[3:6])
        except:
            return [np.nan, np.nan, np.nan]

    df_robot["rpy"] = df_robot.apply(extract_rpy, axis=1)

    t_robot = df_robot[time_col].tolist()
    rpy_list = df_robot["rpy"].tolist()

    roll_robot = [rpy[0] for rpy in rpy_list]
    pitch_robot = [rpy[1] for rpy in rpy_list]
    yaw_robot = [rpy[2] for rpy in rpy_list]

    return t_robot, yaw_robot, pitch_robot, roll_robot


# ---------------------------------------------------------------
# Plot using Plotly
# ---------------------------------------------------------------
def plot_with_plotly(t, ax, ay, az, yaw, pitch, roll,
                     t_robot=None, yaw_robot=None, pitch_robot=None, roll_robot=None, title=""):
    """Plot IMU (Raw Accel + Orientation) and optionally robot orientation."""

    # Normalize time to start at zero
    if t:
        t0 = t[0]
        t = [ti - t0 for ti in t]

    if t_robot:
        t0_robot = t_robot[0]
        t_robot = [ti - t0_robot for ti in t_robot]

    # Pre-compute moving averages
    ax_ma = moving_average(ax, MOVING_AVG_WINDOW)
    ay_ma = moving_average(ay, MOVING_AVG_WINDOW)
    az_ma = moving_average(az, MOVING_AVG_WINDOW)

    yaw_ma = moving_average(yaw, MOVING_AVG_WINDOW)
    pitch_ma = moving_average(pitch, MOVING_AVG_WINDOW)
    roll_ma = moving_average(roll, MOVING_AVG_WINDOW)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Accel X (Raw) [m/s²]", "Accel Y (Raw) [m/s²]", "Accel Z (Raw) [m/s²]",
            "Yaw [deg]", "Pitch [deg]", "Roll [deg]"
        )
    )

    # Row 1: Accelerations (Raw)
    fig.add_trace(go.Scatter(x=t, y=ax, mode="lines", name="acc_x"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=ax_ma, mode="lines", name="acc_x (MA)"), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=ay, mode="lines", name="acc_y"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=ay_ma, mode="lines", name="acc_y (MA)"), row=1, col=2)

    fig.add_trace(go.Scatter(x=t, y=az, mode="lines", name="acc_z"), row=1, col=3)
    fig.add_trace(go.Scatter(x=t, y=az_ma, mode="lines", name="acc_z (MA)"), row=1, col=3)

    # Row 2: Orientations
    fig.add_trace(go.Scatter(x=t, y=yaw, mode="lines", name="yaw (IMU)", line=dict(color="blue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=yaw_ma, mode="lines", name="yaw (IMU MA)", line=dict(color="lightblue")), row=2,
                  col=1)

    if t_robot and yaw_robot:
        fig.add_trace(
            go.Scatter(x=t_robot, y=yaw_robot, mode="lines", name="yaw (Robot)", line=dict(color="red", dash="dash")),
            row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=pitch, mode="lines", name="pitch (IMU)", line=dict(color="blue")), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=pitch_ma, mode="lines", name="pitch (IMU MA)", line=dict(color="lightblue")), row=2,
                  col=2)

    if t_robot and pitch_robot:
        fig.add_trace(go.Scatter(x=t_robot, y=pitch_robot, mode="lines", name="pitch (Robot)",
                                 line=dict(color="red", dash="dash")), row=2, col=2)

    fig.add_trace(go.Scatter(x=t, y=roll, mode="lines", name="roll (IMU)", line=dict(color="blue")), row=2, col=3)
    fig.add_trace(go.Scatter(x=t, y=roll_ma, mode="lines", name="roll (IMU MA)", line=dict(color="lightblue")), row=2,
                  col=3)

    if t_robot and roll_robot:
        fig.add_trace(
            go.Scatter(x=t_robot, y=roll_robot, mode="lines", name="roll (Robot)", line=dict(color="red", dash="dash")),
            row=2, col=3)

    fig.update_layout(
        height=800,
        width=1200,
        title_text=title,
        showlegend=True,
    )

    fig.show()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="CSV to load. If omitted, uses latest in data/CSV/.")
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        csv_path = find_latest_csv()
        if not csv_path:
            print("No CSV files found in data/CSV/. Cannot continue.")
            return
        print(f"Using latest CSV: {csv_path}")

    t, ax, ay, az, yaw, pitch, roll = load_imu_from_csv(csv_path)

    if not t:
        print("No IMU rows found in CSV.")
        return

    # Try to load robot pose data for comparison
    t_robot, yaw_robot, pitch_robot, roll_robot = load_robot_pose_from_csv(csv_path)

    if t_robot:
        print(f"Loaded {len(t_robot)} robot pose samples")
    else:
        print("No robot pose data found (or no pose matrices in CSV)")
        t_robot = yaw_robot = pitch_robot = roll_robot = None

    plot_with_plotly(
        t, ax, ay, az, yaw, pitch, roll,
        t_robot=t_robot, yaw_robot=yaw_robot, pitch_robot=pitch_robot, roll_robot=roll_robot,
        title=f"IMU (Raw) & Robot Orientation ({os.path.basename(csv_path)})"
    )


if __name__ == "__main__":
    main()