#!/usr/bin/env python3
import csv
import json
import ast
import argparse
from typing import Dict, Any, List, Tuple, Iterable
import numpy as np
import plotly.graph_objects as go
import re
import os
import glob
from pathlib import Path


# -----------------------------------------------------------
# Helpers for finding latest CSV or NPZ
# -----------------------------------------------------------
def _find_latest_csv(pattern: str = "CSV/session_log_*.csv") -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _find_default_transform(pattern: str = "DATA/hand_eye/*.npz") -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


# -----------------------------------------------------------
# Parsing helpers (same as before)
# -----------------------------------------------------------
def parse_data_cell(s: str):
    if not s:
        return {}
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    text = re.sub(r'\barray\(', '', s)
    text = re.sub(r',\s*dtype=[^)]+', '', text)
    text = text.replace('])', ']')
    try:
        return ast.literal_eval(text)
    except Exception:
        return {}


POSE_COLS = [f"pose_{r}{c}" for r in range(4) for c in range(4)]


def _row_has_flat_pose(row: Dict[str, str]) -> bool:
    return all(col in row and row[col] not in (None, "", "null", "None") for col in POSE_COLS)


def _flat_pose_from_row(row: Dict[str, str]) -> np.ndarray:
    vals = [float(row[f"pose_{r}{c}"]) for r in range(4) for c in range(4)]
    return np.array(vals, dtype=float).reshape(4, 4)


# -----------------------------------------------------------
# Extract points from CSV
# -----------------------------------------------------------
def extract_points(csv_path: str):
    points = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = (row.get("source") or "").strip()
            evt = (row.get("event") or "").strip()
            ts = (row.get("timestamp") or "").strip()

            if _row_has_flat_pose(row):
                H = _flat_pose_from_row(row)
                x, y, z = float(H[0, 3]), float(H[1, 3]), float(H[2, 3])
                tag_id = (row.get("tag_id") or "").strip() or None
                kind = "camera" if evt == "tag_pose_snapshot" or (src and src.lower() != "robot") else "robot"
                points.append(dict(timestamp=ts, source=src, kind=kind, event=evt,
                                   tag_id=tag_id, x=x, y=y, z=z))
    return points


# -----------------------------------------------------------
# Transform application
# -----------------------------------------------------------
def apply_rt_to_camera_points(points, R, t):
    out = []
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    for p in points:
        if p["kind"] != "camera":
            continue
        v = np.array([p["x"], p["y"], p["z"]], dtype=float)
        v2 = R @ v + t
        q = dict(p)
        q.update(x=float(v2[0]), y=float(v2[1]), z=float(v2[2]), kind="camera_aligned")
        out.append(q)
    return out


# -----------------------------------------------------------
# Plotting
# -----------------------------------------------------------
def build_figure(points: List[Dict[str, Any]],
                 connect_robot: bool = True,
                 connect_cameras: bool = False,
                 group_by_tag: bool = False) -> go.Figure:
    fig = go.Figure()
    if not points:
        fig.update_layout(title="No pose data found")
        return fig
    for i, p in enumerate(points):
        p["_idx"] = i

    robots = [p for p in points if p["kind"] == "robot"]
    cams = [p for p in points if p["kind"] == "camera"]
    cams_aligned = [p for p in points if p["kind"] == "camera_aligned"]

    def color_for(i):
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                  "#bcbd22", "#17becf"]
        return colors[i % len(colors)]

    if robots:
        rs = sorted(robots, key=lambda p: p["_idx"])
        fig.add_trace(go.Scatter3d(
            x=[p["x"] for p in rs], y=[p["y"] for p in rs], z=[p["z"] for p in rs],
            mode="markers+lines" if connect_robot else "markers",
            marker=dict(size=4), line=dict(width=2), name="Robot",
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>Robot</extra>"
        ))

    def add_traces(data, label_suffix="", size=3, offset=0):
        by_src = {}
        for p in data:
            by_src.setdefault(p["source"], []).append(p)
        for i, (src, pts) in enumerate(by_src.items()):
            color = color_for(i + offset)
            pts = sorted(pts, key=lambda p: p["_idx"])
            fig.add_trace(go.Scatter3d(
                x=[p["x"] for p in pts], y=[p["y"] for p in pts], z=[p["z"] for p in pts],
                mode="markers+lines" if connect_cameras else "markers",
                marker=dict(size=size, color=color),
                line=dict(width=2, color=color),
                name=f"Camera {src}{label_suffix}",
            ))

    add_traces(cams)
    add_traces(cams_aligned, " (aligned)", size=4, offset=5)

    fig.update_layout(scene=dict(aspectmode="data"),
                      title="3D Robot & Camera Observations",
                      margin=dict(l=0, r=0, b=0, t=40))
    return fig


# -----------------------------------------------------------
# Main CLI
# -----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot 3D robot & camera observations from session log CSV")
    ap.add_argument("--csv", help="Path to session_log_*.csv (defaults to latest CSV in CSV/)")
    ap.add_argument("--transform", help="Path to NPZ transform (defaults to DATA/hand_eye/*.npz)")
    ap.add_argument("--only-aligned", action="store_true", help="Plot only aligned camera traces")
    ap.add_argument("--group-by-tag", action="store_true", default=True)
    ap.add_argument("--robot-lines", action="store_true", default=True)
    ap.add_argument("--camera-lines", action="store_true", default=True)
    args = ap.parse_args()

    # Auto-select latest CSV
    csv_path = args.csv or _find_latest_csv()
    if not csv_path:
        print("❌ No CSV found in CSV/. Exiting.")
        return
    print(f"Using CSV: {csv_path}")

    # Auto-select default transform if not provided
    transform_path = args.transform or _find_default_transform()
    if transform_path:
        print(f"Using transform: {transform_path}")
    else:
        print("No transform file found. Plotting raw camera poses only.")

    # Load points
    points = extract_points(csv_path)
    print(f"Loaded {len(points)} poses from {Path(csv_path).name}")

    # Apply transform if available
    if transform_path:
        try:
            rt = np.load(transform_path, allow_pickle=True)
            R, t = rt["R"], rt["t"]
            aligned = apply_rt_to_camera_points(
                [p for p in points if p["kind"] == "camera"], R, t)
            points = ([p for p in points if p["kind"] != "camera"] + aligned
                      if args.only_aligned else points + aligned)
            print("Applied R,t transform")
        except Exception as e:
            print(f"Failed to load transform {transform_path}: {e}")

    # Build & save plot
    fig = build_figure(points, args.robot_lines, args.camera_lines, args.group_by_tag)
    html_path = Path(csv_path).with_suffix(".html")
    fig.write_html(html_path)
    print(f"Saved interactive HTML to {html_path}")
    fig.show()


if __name__ == "__main__":
    main()
