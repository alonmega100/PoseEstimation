#!/usr/bin/env python3
import argparse
import csv
import json
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MOVING_AVG_WINDOW = 200  # number of samples in the moving average window


def moving_average(values, window):
    """Simple trailing moving average. Returns list of same length, NaN until window is full."""
    if window <= 1:
        return list(values)

    out = []
    cumsum = 0.0
    for i, v in enumerate(values):
        cumsum += v
        if i >= window:
            cumsum -= values[i - window]
        if i >= window - 1:
            out.append(cumsum / window)
        else:
            out.append(float("nan"))
    return out


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
    t, wax, way, waz = [], [], [], []          # world-frame accel (after analysis)
    bax, bay, baz = [], [], []          # body-frame accel (before analysis)
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
            acc_world = data.get("acc_world_m_s2")
            acc_body = data.get("acc_body")  # may be missing in older CSVs

            # Require valid world-frame acceleration
            if t_sec is None or acc_world is None or len(acc_world) != 3:
                continue

            t.append(float(t_sec))
            wax.append(float(acc_world[0]))
            way.append(float(acc_world[1]))
            waz.append(float(acc_world[2]))

            # If body-frame accel exists, use it; otherwise fill with NaNs so lengths match
            if acc_body is not None and len(acc_body) == 3:
                bax.append(float(acc_body[0]))
                bay.append(float(acc_body[1]))
                baz.append(float(acc_body[2]))
            else:
                bax.append(float("nan"))
                bay.append(float("nan"))
                baz.append(float("nan"))

            yaw.append(float(data.get("yaw_deg", 0.0)))
            pitch.append(float(data.get("pitch_deg", 0.0)))
            roll.append(float(data.get("roll_deg", 0.0)))

    return t, wax, way, waz, bax, bay, baz, yaw, pitch, roll


# ---------------------------------------------------------------
# Plot using Plotly
# ---------------------------------------------------------------
def plot_with_plotly(t, wax, way, waz, bax, bay, baz, yaw, pitch, roll, title):
    # Make time start at zero
    if t:
        t0 = t[0]
        t = [ti - t0 for ti in t]

    # Pre-compute moving averages
    wax_ma = moving_average(wax, MOVING_AVG_WINDOW)
    way_ma = moving_average(way, MOVING_AVG_WINDOW)
    waz_ma = moving_average(waz, MOVING_AVG_WINDOW)

    bax_ma = moving_average(bax, MOVING_AVG_WINDOW)
    bay_ma = moving_average(bay, MOVING_AVG_WINDOW)
    baz_ma = moving_average(baz, MOVING_AVG_WINDOW)

    yaw_ma = moving_average(yaw, MOVING_AVG_WINDOW)
    pitch_ma = moving_average(pitch, MOVING_AVG_WINDOW)
    roll_ma = moving_average(roll, MOVING_AVG_WINDOW)

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "World accel X [m/s²]", "World accel Y [m/s²]", "World accel Z [m/s²]",
            "Body accel X (raw) [m/s²]", "Body accel Y (raw) [m/s²]", "Body accel Z (raw) [m/s²]",
            "Yaw [deg]", "Pitch [deg]", "Roll [deg]"
        )
    )

    # Row 1: world-frame accelerations
    fig.add_trace(go.Scatter(x=t, y=wax,    mode="lines", name="a_world_x"),        row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=wax_ma, mode="lines", name="a_world_x (MA)"),   row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=way,    mode="lines", name="a_world_y"),        row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=way_ma, mode="lines", name="a_world_y (MA)"),   row=1, col=2)

    fig.add_trace(go.Scatter(x=t, y=waz,    mode="lines", name="a_world_z"),        row=1, col=3)
    fig.add_trace(go.Scatter(x=t, y=waz_ma, mode="lines", name="a_world_z (MA)"),   row=1, col=3)

    # Row 2: body-frame accelerations
    fig.add_trace(go.Scatter(x=t, y=bax,    mode="lines", name="a_body_x"),         row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=bax_ma, mode="lines", name="a_body_x (MA)"),    row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=bay,    mode="lines", name="a_body_y"),         row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=bay_ma, mode="lines", name="a_body_y (MA)"),    row=2, col=2)

    fig.add_trace(go.Scatter(x=t, y=baz,    mode="lines", name="a_body_z"),         row=2, col=3)
    fig.add_trace(go.Scatter(x=t, y=baz_ma, mode="lines", name="a_body_z (MA)"),    row=2, col=3)

    # Row 3: orientations
    fig.add_trace(go.Scatter(x=t, y=yaw,    mode="lines", name="yaw"),              row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=yaw_ma, mode="lines", name="yaw (MA)"),         row=3, col=1)

    fig.add_trace(go.Scatter(x=t, y=pitch,  mode="lines", name="pitch"),            row=3, col=2)
    fig.add_trace(go.Scatter(x=t, y=pitch_ma, mode="lines", name="pitch (MA)"),     row=3, col=2)

    fig.add_trace(go.Scatter(x=t, y=roll,   mode="lines", name="roll"),             row=3, col=3)
    fig.add_trace(go.Scatter(x=t, y=roll_ma, mode="lines", name="roll (MA)"),       row=3, col=3)

    fig.update_layout(
        height=1000,
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

    t, wax, way, waz, bax, bay, baz, yaw, pitch, roll = load_imu_from_csv(csv_path)

    if not t:
        print("No IMU rows found in CSV.")
        return

    plot_with_plotly(
        t, wax, way, waz, bax, bay, baz, yaw, pitch, roll,
        title=f"IMU Accelerations & Orientation ({os.path.basename(csv_path)}), Moving Average K: {MOVING_AVG_WINDOW}"
    )



if __name__ == "__main__":
    main()
