#!/usr/bin/env python3
import argparse
import csv
import json
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------
# Find latest CSV in ./CSV/
# ---------------------------------------------------------------
def find_latest_csv():
    os.makedirs("CSV", exist_ok=True)
    csvs = glob.glob(os.path.join("CSV", "*.csv"))
    if not csvs:
        return None
    csvs.sort(key=lambda p: os.path.getmtime(p))
    return csvs[-1]


# ---------------------------------------------------------------
# Load IMU rows
# ---------------------------------------------------------------
def load_imu_from_csv(csv_path):
    t, ax, ay, az = [], [], [], []
    yaw, pitch, roll = [], [], []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("source") != "imu":
                continue

            raw = row.get("raw_data") or ""
            try:
                data = json.loads(raw)
            except Exception:
                continue

            t_sec = data.get("t_sec")
            acc = data.get("acc_world_m_s2")

            if t_sec is None or acc is None or len(acc) != 3:
                continue

            t.append(float(t_sec))
            ax.append(float(acc[0]))
            ay.append(float(acc[1]))
            az.append(float(acc[2]))

            yaw.append(float(data.get("yaw_deg", 0.0)))
            pitch.append(float(data.get("pitch_deg", 0.0)))
            roll.append(float(data.get("roll_deg", 0.0)))

    return t, ax, ay, az, yaw, pitch, roll


# ---------------------------------------------------------------
# Plot using Plotly
# ---------------------------------------------------------------
def plot_with_plotly(t, ax, ay, az, yaw, pitch, roll, title):
    if t:
        t0 = t[0]
        t = [ti - t0 for ti in t]  # start at zero

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Accel X [m/s²]", "Accel Y [m/s²]", "Accel Z [m/s²]",
            "Yaw [deg]", "Pitch [deg]", "Roll [deg]"
        )
    )

    # Accelerations
    fig.add_trace(go.Scatter(x=t, y=ax, mode="lines", name="a_x"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=ay, mode="lines", name="a_y"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=az, mode="lines", name="a_z"), row=1, col=3)

    # Orientations
    fig.add_trace(go.Scatter(x=t, y=yaw, mode="lines", name="yaw"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=pitch, mode="lines", name="pitch"), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=roll, mode="lines", name="roll"), row=2, col=3)

    fig.update_layout(
        height=700,
        width=1200,
        title_text=title,
        showlegend=False
    )

    fig.show()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="CSV to load. If omitted, uses latest in CSV/.")
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        csv_path = find_latest_csv()
        if not csv_path:
            print("No CSV files found in CSV/. Cannot continue.")
            return
        print(f"Using latest CSV: {csv_path}")

    t, ax, ay, az, yaw, pitch, roll = load_imu_from_csv(csv_path)

    if not t:
        print("No IMU rows found in CSV.")
        return

    plot_with_plotly(t, ax, ay, az, yaw, pitch, roll,
                     title=f"IMU Accelerations & Orientation ({os.path.basename(csv_path)})")


if __name__ == "__main__":
    main()
