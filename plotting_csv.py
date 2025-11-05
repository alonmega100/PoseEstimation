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


def _find_latest_csv(pattern: str = "CSV/session_log_*.csv") -> str | None:
    try:
        files = glob.glob(pattern)
        if not files:
            return None
        latest_file = max(files, key=os.path.getmtime)
        return latest_file
    except Exception as e:
        print(f"Warning: Could not automatically find latest CSV file. Error: {e}")
        return None


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

    def _sanitize_numpy_arrays(text: str) -> str:
        text = re.sub(r'\barray\(', '', text)
        text = re.sub(r',\s*dtype=[^)]+', '', text)
        while '])' in text:
            text = text.replace('])', ']')
        return text

    try:
        sanitized = _sanitize_numpy_arrays(s)
        return ast.literal_eval(sanitized)
    except Exception:
        return {}


def _pose_entries_from_value(value: Any) -> Iterable[Tuple[str | None, np.ndarray]]:
    try:
        H = np.array(value, dtype=float)
        if H.shape == (4, 4):
            yield (None, H)
            return
    except Exception:
        pass

    if isinstance(value, dict):
        for tag_id, Hlike in value.items():
            try:
                H = np.array(Hlike, dtype=float)
                if H.shape == (4, 4):
                    yield (str(tag_id), H)
            except Exception:
                continue


def extract_points(csv_path: str):
    points = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = (row.get("source") or "").strip()
            evt = (row.get("event") or "").strip()
            ts = (row.get("timestamp") or "").strip()

            data = parse_data_cell(row.get("data", ""))
            if not isinstance(data, dict):
                continue
            if "pose" not in data:
                continue

            for tag_id, H in _pose_entries_from_value(data["pose"]):
                x, y, z = float(H[0, 3]), float(H[1, 3]), float(H[2, 3])

                kind = "camera" if evt == "tag_pose_snapshot" or src.lower() != "robot" else "robot"

                points.append({
                    "timestamp": ts,
                    "source": src,
                    "kind": kind,
                    "event": evt,
                    "tag_id": tag_id,
                    "x": x, "y": y, "z": z,
                })
    return points


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

    base_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]

    def color_for(idx: int) -> str:
        return base_colors[idx % len(base_colors)]

    if robots:
        robots_sorted = sorted(robots, key=lambda p: p["_idx"])
        fig.add_trace(go.Scatter3d(
            x=[p["x"] for p in robots_sorted],
            y=[p["y"] for p in robots_sorted],
            z=[p["z"] for p in robots_sorted],
            mode="markers+lines" if connect_robot else "markers",
            marker=dict(size=4),
            line=dict(width=2),
            name="Robot",
            text=[f"{p['timestamp']} | {p['event']}" for p in robots_sorted],
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                          "<br>%{text}<extra>Robot</extra>",
        ))

    if cams:
        if group_by_tag:
            keyfunc = lambda p: (p["source"], p.get("tag_id"))
        else:
            keyfunc = lambda p: (p["source"], None)

        buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for p in cams:
            k = keyfunc(p)
            buckets.setdefault(k, []).append(p)

        ordered_keys = sorted(buckets.keys(), key=lambda k: min(pp["_idx"] for pp in buckets[k]))
        source_index: Dict[str, int] = {}
        next_color_idx = 0

        for key in ordered_keys:
            src, tag = key
            plist = buckets[key]

            if src not in source_index:
                source_index[src] = next_color_idx
                next_color_idx += 1
            color = color_for(source_index[src])

            plist_sorted = sorted(plist, key=lambda p: p["_idx"])
            label = f"Camera {src}" if tag in (None, "", "None") else f"Cam {src} • tag {tag}"

            fig.add_trace(go.Scatter3d(
                x=[p["x"] for p in plist_sorted],
                y=[p["y"] for p in plist_sorted],
                z=[p["z"] for p in plist_sorted],
                mode="markers+lines" if connect_cameras else "markers",
                marker=dict(size=3, color=color),
                line=dict(width=1, color=color),
                name=label,
                text=[f"{p['timestamp']} | tag={p.get('tag_id') or '-'} | {p['event']}"
                      for p in plist_sorted],
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                              "<br>%{text}<extra></extra>",
            ))

    fig.update_layout(
        title="3D Robot & Camera Observations",
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="data"
        ),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description="Plot 3D robot & camera observations from session log CSV")
    ap.add_argument("--csv", required=False, help="Path to session_log_*.csv (defaults to latest)")
    ap.add_argument("--group-by-tag", action="store_true", default=True,
                    help="Split camera traces by tag_id (default: ON)")
    ap.add_argument("--robot-lines", action="store_true", default=True,
                    help="Connect robot points with lines (default: ON)")
    ap.add_argument("--camera-lines", action="store_true", default=True,
                    help="Connect camera points with lines (default: ON)")
    ap.add_argument("--verbose", action="store_true", help="Print counts by source")
    args = ap.parse_args()

    csv_path = args.csv
    if not csv_path:
        csv_path = _find_latest_csv("CSV/session_log_*.csv")
        if not csv_path:
            print("Error: No CSV file specified and no 'session_log_*.csv' found in the current directory.")
            ap.print_help()
            return
        print(f"Automatically selected latest CSV file: {csv_path}")

    points = extract_points(csv_path)
    print(f"Loaded {len(points)} poses from {Path(csv_path).name}")

    if args.verbose:
        from collections import Counter
        total = len(points)
        cams = sum(1 for p in points if p["kind"] == "camera")
        robs = total - cams
        print(f"Total points: {total} | robot: {robs} | camera: {cams}")
        print("By source:", Counter(p["source"] for p in points))

    if not points:
        print("No pose data found in CSV.")
        return

    fig = build_figure(
        points,
        connect_robot=args.robot_lines,
        connect_cameras=args.camera_lines,
        group_by_tag=args.group_by_tag,
    )

    # --- always save next to CSV with same name ---
    csv_path_obj = Path(csv_path)
    html_path = csv_path_obj.with_suffix(".html")  # CSV/session_log_2025-11-05_10-00-00.html
    if html_path.exists():
        print(f"HTML already exists, overwriting: {html_path}")
    fig.write_html(html_path)
    print(f"Saved interactive HTML to: {html_path}")
    fig.show()


if __name__ == "__main__":
    main()
