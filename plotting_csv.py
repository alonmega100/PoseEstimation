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
import glob  # New import for file searching
from pathlib import Path  # New import for modern path handling


# --- New Helper Function for File Search ---

def _find_latest_csv(pattern: str = "CSV/session_log_*.csv") -> str | None:
    """Finds the most recently modified file matching the pattern in a subdirectory."""
    try:
        # Get all matching files.
        # By omitting the leading /, it correctly looks for a 'CSV' directory
        # within the current working directory.
        files = glob.glob(pattern)

        if not files:
            return None

        # Sort by modification time (st_mtime)
        latest_file = max(files, key=os.path.getmtime)
        return latest_file
    except Exception as e:
        print(f"Warning: Could not automatically find latest CSV file. Error: {e}")
        return None
# --- End New Helper Function ---


def parse_data_cell(s: str):
    """Parse the 'data' column which may contain Python dicts with numpy array(...) prints.
    Strategy: try JSON, then AST; if that fails, sanitize away numpy 'array(...)' wrappers and retry AST.
    """
    if not s:
        return {}
    s = s.strip()

    # First: JSON (works if double quotes and lists)
    try:
        import json
        return json.loads(s)
    except Exception:
        pass

    # Second: AST (works for Python-literal dicts/lists, but not 'array(...)')
    import ast
    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    # Third: sanitize numpy array(...) textual form → lists
    def _sanitize_numpy_arrays(text: str) -> str:
        text = re.sub(r'\barray\(', '', text)  # remove leading "array("
        text = re.sub(r',\s*dtype=[^)]+', '', text)  # drop ", dtype=float32" etc.
        while '])' in text:  # close brackets: "])" -> "]"
            text = text.replace('])', ']')
        return text

    try:
        sanitized = _sanitize_numpy_arrays(s)
        return ast.literal_eval(sanitized)
    except Exception:
        # Last resort: give up; caller should treat as no data
        return {}


def _pose_entries_from_value(value: Any) -> Iterable[Tuple[str | None, np.ndarray]]:
    """
    Yields (tag_id, H) tuples:
      - value is a 4x4 -> yield (None, H)
      - value is {tag_id: 4x4} -> yield (str(tag_id), H) for each
    """
    # Try single 4x4 first
    try:
        H = np.array(value, dtype=float)
        if H.shape == (4, 4):
            yield (None, H)
            return
    except Exception:
        pass

    # Dict of tag_id -> 4x4
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

            data = parse_data_cell(row.get("data", ""))  # <-- now robust
            if not isinstance(data, dict):
                continue
            if "pose" not in data:
                continue

            # Pose can be a 4x4 OR a {tag_id: 4x4} dict
            for tag_id, H in _pose_entries_from_value(data["pose"]):
                x, y, z = float(H[0, 3]), float(H[1, 3]), float(H[2, 3])

                # classify: if the row is a camera snapshot or src is a serial → camera
                kind = "camera" if evt == "tag_pose_snapshot" or src.lower() != "robot" else "robot"

                points.append({
                    "timestamp": ts,
                    "source": src,
                    "kind": kind,
                    "event": evt,
                    "tag_id": tag_id,  # None for single pose; "2"/"7"/... for per-tag
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
        # group either by (source) or by (source, tag_id)
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
    # --- 1. Argument Parser Setup ---
    # Make --csv required=False
    ap = argparse.ArgumentParser(description="Plot 3D robot & camera observations from session log CSV")
    ap.add_argument("--csv", required=False, help="Path to session_log_*.csv (defaults to latest)")
    ap.add_argument("--save", default=None, help="Optional: save interactive HTML to this path")

    # Change action="store_true" to store_false and adjust default for clarity, or flip logic
    # We will flip the logic in the function calls below for simplicity
    ap.add_argument("--group-by-tag", action="store_true", default=True,
                    help="Split camera traces by tag_id (default: ON)")
    ap.add_argument("--robot-lines", action="store_true", default=True,
                    help="Connect robot points with lines (default: ON)")
    ap.add_argument("--camera-lines", action="store_true", default=True,
                    help="Connect camera points with lines (default: ON)")
    ap.add_argument("--verbose", action="store_true", help="Print counts by source")
    args = ap.parse_args()

    # --- 2. Automatic CSV Path Handling ---
    csv_path = args.csv
    if not csv_path:
        csv_path = _find_latest_csv("CSV/session_log_*.csv")
        if not csv_path:
            print("Error: No CSV file specified and no 'session_log_*.csv' found in the current directory.")
            ap.print_help()
            return
        print(f"Automatically selected latest CSV file: {csv_path}")

    # --- 3. Extract and Plot Data ---
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

    # --- 4. Flag Logic (Ensure desired defaults: Robot Lines ON, Camera Lines OFF, Group OFF) ---
    fig = build_figure(
        points,
        # Default: connect_robot=True. If --no-robot-lines is passed, it becomes False.
        connect_robot= args.robot_lines,
        # Default: connect_cameras=False. If --camera-lines is passed, it becomes True.
        connect_cameras=args.camera_lines,
        # Default: group_by_tag=False. If --group-by-tag is passed, it becomes True.
        group_by_tag=args.group_by_tag,
    )

    if args.save:
        fig.write_html(args.save)
        print(f"Saved interactive HTML to: {args.save}")

    # Use fig.show() which is the default for Plotly
    # Note: If you encounter the UserWarning again, refer to the previous chat answer to save the plot.
    fig.show()


if __name__ == "__main__":
    main()