#!/usr/bin/env python3
import csv
import json
import ast
import argparse
from typing import Dict, Any, List, Tuple, Iterable
from typing import Iterable, Tuple, Any
import numpy as np
import numpy as np
import plotly.graph_objects as go
import re

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
    # examples:
    #   array([[1,2],[3,4]])            -> [[1,2],[3,4]]
    #   array([...], dtype=float32)     -> [...]
    # Works even when nested inside dicts: {'pose': {2: array([[...]]), 7: array([[...]])}}
    def _sanitize_numpy_arrays(text: str) -> str:
        text = re.sub(r'\barray\(', '', text)                 # remove leading "array("
        text = re.sub(r',\s*dtype=[^)]+', '', text)           # drop ", dtype=float32" etc.
        while '])' in text:                                   # close brackets: "])" -> "]"
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
            ts  = (row.get("timestamp") or "").strip()

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
                    "tag_id": tag_id,   # None for single pose; "2"/"7"/... for per-tag
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
    cams   = [p for p in points if p["kind"] == "camera"]

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
    ap = argparse.ArgumentParser(description="Plot 3D robot & camera observations from session log CSV")
    ap.add_argument("--csv", required=True, help="Path to session_log_*.csv")
    ap.add_argument("--save", default=None, help="Optional: save interactive HTML to this path")
    ap.add_argument("--group-by-tag", action="store_true", help="Split camera traces by tag_id")
    ap.add_argument("--no-robot-lines", action="store_true", help="Do not connect robot points with lines")
    ap.add_argument("--camera-lines", action="store_true", help="Connect camera points with lines")
    ap.add_argument("--verbose", action="store_true", help="Print counts by source")
    args = ap.parse_args()

    points = extract_points(args.csv)
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
        connect_robot=not args.no_robot_lines,
        connect_cameras=args.camera_lines,
        group_by_tag=args.group_by_tag,
    )

    if args.save:
        fig.write_html(args.save)
        print(f"Saved interactive HTML to: {args.save}")

    fig.show()

if __name__ == "__main__":
    main()
