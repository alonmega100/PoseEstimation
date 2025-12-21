import numpy as np
import os
import glob

from typing import Dict, Tuple, Optional, Any


def to_H(R, t):
    H = np.eye(4); H[:3,:3] = R; H[:3,3] = t.reshape(3)
    return H

def inv_H(H):
    Ri = H[:3,:3].T
    ti = -Ri @ H[:3,3]
    Hi = np.eye(4); Hi[:3,:3] = Ri; Hi[:3,3] = ti
    return Hi

def H_to_xyzrpy_ZYX(H):
    R = H[:3,:3]; x,y,z = H[:3,3]
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-9:
        yaw   = np.arctan2(R[1,0], R[0,0])
        pitch = np.arctan2(-R[2,0], sy)
        roll  = np.arctan2(R[2,1], R[2,2])
    else:  # near gimbal lock
        yaw   = np.arctan2(-R[0,1], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        roll  = 0.0
    return np.array([x,y,z, roll,pitch,yaw])  # (x,y,z,r,p,y)

def se3_blend(H_new, H_old, alpha):
    """Blend two poses (translation linear, rotation via SVD). alpha = weight on NEW."""
    if H_old is None: return H_new
    t_old, t_new = H_old[:3,3], H_new[:3,3]
    t_out = (1.0 - alpha) * t_old + alpha * t_new
    R_old, R_new = H_old[:3,:3], H_new[:3,:3]
    M = (1.0 - alpha) * R_old + alpha * R_new
    U, _, Vt = np.linalg.svd(M)
    R_out = U @ Vt
    H_out = np.eye(4); H_out[:3,:3] = R_out; H_out[:3,3] = t_out
    return H_out


def rot_geodesic_angle_deg(R1, R2):
    """Smallest angle between two rotations (degrees)."""
    R = R1.T @ R2
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))



def pose_delta(Ha, Hb):
    """Return (pos_err_m, ang_err_deg) between two world poses."""
    pa, pb = Ha[:3,3], Hb[:3,3]
    print("calculating pose delta. this is pa and pb:", pa,pb)
    Ra, Rb = Ha[:3,:3], Hb[:3,:3]
    pos_err = np.linalg.norm(pa - pb )
    ang_err = rot_geodesic_angle_deg(Ra, Rb)
    return pos_err, ang_err


def is_4x4_matrix(obj):
    import numpy as _np
    # accept list-of-lists AND np.array
    if isinstance(obj, _np.ndarray):
        return obj.shape == (4, 4)
    return (
            isinstance(obj, (list, tuple)) and
            len(obj) == 4 and
            all(isinstance(row, (list, tuple)) and len(row) == 4 for row in obj)
    )


def pose_row_to_matrix(row: Any, prefix: str = "pose_") -> np.ndarray:
    """
    Construct a 4x4 matrix from a dictionary-like row (e.g. pandas Series)
    containing keys '{prefix}00' through '{prefix}33'.
    """
    H = np.eye(4, dtype=float)
    for r in range(4):
        for c in range(4):
            key = f"{prefix}{r}{c}"
            # We assume the key exists or let it raise KeyError,
            # but usually row is a Series/dict from the CSV.
            H[r, c] = float(row[key])
    return H


def matrix_to_flat_dict(prefix, mat):
    mat = np.array(mat)  # safe conversion
    out = {}
    for r in range(4):
        for c in range(4):
            out[f"{prefix}_{r}{c}"] = float(mat[r, c])
    return out


def list_of_movements_generator(
    num_commands: int,
    bounds = {"x": (-0.1, 0.1), "y": (0.0, 0.25), "z": (-0.15, 0.0)},
    p_axis: float = 0.5,
    precision: float = 0.1,
    rng: np.random.Generator | None = None,
):
    """
    Generate commands like 'x 0.1 y -0.2' with cumulative state.
    Any axis whose rounded delta == 0 is omitted (no 'x 0').
    Guarantees exactly num_commands non-empty commands.
    """
    if rng is None:
        rng = np.random.default_rng()

    pos = {"x": 0.0, "y": 0.0, "z": 0.0}

    def fmt(v: float) -> str:
        # nice formatting without trailing zeros
        s = f"{v:.10f}".rstrip("0").rstrip(".")
        return s if s else "0"

    cmds: list[str] = []
    axes = ("x", "y", "z")

    while len(cmds) < num_commands:
        parts = []
        for axis in axes:
            if rng.random() >= p_axis:
                continue
            lb, ub = bounds[axis]
            # sample relative to remaining room (still cumulative, but no hard clipping)
            raw = rng.uniform(lb - pos[axis], ub - pos[axis])
            delta = round(raw / precision) * precision
            if abs(delta) < 1e-12:   # <-- skip zeros so we never emit 'x 0'
                continue
            pos[axis] += delta
            parts.append(f"{axis} {fmt(delta)}")

        if parts:                   # ensure non-empty command
            cmds.append(" ".join(parts))
        # else: loop again until we produce a non-empty command

    return cmds


def make_vn_cmd(body: str) -> bytes:
    cs = 0
    for b in body.encode("ascii"):
        cs ^= b
    return f"${body}*{cs:02X}\r\n".encode("ascii")




def parse_vn_vnrrg_08(line: str) -> Optional[Tuple[float, ...]]:
    """
    Parse VectorNav ASCII response for a few useful registers.

    Supports:
    - $VNRRG,08,...  -> yaw, pitch, roll (no accel)
    - $VNRRG,27,...  -> yaw, pitch, roll, accelX, accelY, accelZ  (YMR)

    Returns:
        (yaw_deg, pitch_deg, roll_deg)
    or
        (yaw_deg, pitch_deg, roll_deg, ax, ay, az)
    """
    if not line.startswith("$VNRRG,"):
        return None
    try:
        data_part = line.split("*", 1)[0]
        parts = data_part.split(",")
        if len(parts) < 5:
            return None

        reg_id = parts[1]
        yaw = float(parts[2])
        pitch = float(parts[3])
        roll = float(parts[4])

        if reg_id == "08":
            # Plain YPR register: yaw, pitch, roll only
            return yaw, pitch, roll

        if reg_id == "27":
            # YMR: yaw, pitch, roll, magX, magY, magZ, accelX, accelY, accelZ, gyroX, gyroY, gyroZ
            # indices: 2=yaw, 3=pitch, 4=roll, 5=magX, 6=magY, 7=magZ, 8=accX, 9=accY, 10=accZ, ...
            if len(parts) < 11:
                return yaw, pitch, roll
            ax = float(parts[8])
            ay = float(parts[9])
            az = float(parts[10])
            return yaw, pitch, roll, ax, ay, az

        # Fallback: at least return orientation
        return yaw, pitch, roll
    except Exception:
        return None


def apply_rt_to_camera_points(points, R, t):
    """
    Legacy helper: apply a single R,t transform to all camera points.
    """
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


def apply_rt_to_camera_points_per_cam(points, rt_by_source: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """
    New helper: apply a different R,t per camera source.
    rt_by_source: dict[source] = (R, t)
    """
    out = []
    for p in points:
        if p["kind"] != "camera":
            continue
        src = p.get("source", "")
        if src not in rt_by_source:
            # No transform for this camera; skip it (it will remain in raw form)
            continue
        R, t = rt_by_source[src]
        R = np.asarray(R, dtype=float)
        t = np.asarray(t, dtype=float).reshape(3)
        v = np.array([p["x"], p["y"], p["z"]], dtype=float)
        v2 = R @ v + t
        q = dict(p)
        q.update(x=float(v2[0]), y=float(v2[1]), z=float(v2[2]), kind="camera_aligned")
        out.append(q)
    return out


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


def choose_csv_interactively(csv_dir: str) -> str:
    # Ensure directory exists before globbing
    if not os.path.exists(csv_dir):
        print(f"[ERROR] CSV directory not found: {os.path.abspath(csv_dir)}")
        # Fallback to local if data/CSV fails? Or just crash gracefully
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    pattern = os.path.join(csv_dir, "*.csv")
    files = [os.path.basename(f) for f in glob.glob(pattern)]
    if not files:
        raise FileNotFoundError(f"No CSV files found with pattern {pattern}")

    files = sorted(files)
    print("Available CSV files in", csv_dir)
    for i, name in enumerate(files):
        print(f"  {i}: {name}")

    num = input(
        "File to load? 0 for the first and so on...\n"
        "Press Enter for the last one\n :"
    )
    if not num.strip():
        idx = -1
    else:
        idx = int(num)

    chosen = files[idx]
    print("You chose", chosen)
    return os.path.join(csv_dir, chosen)


def find_latest_csv():
    # Looking in data/CSV relative to project root
    search_path = "data/CSV"
    if not os.path.exists(search_path):
        return None

    csvs = glob.glob(os.path.join(search_path, "*.csv"))
    if not csvs:
        return None
    csvs.sort(key=lambda p: os.path.getmtime(p))
    return csvs[-1]

def rpy_to_R_deg(yaw_deg, pitch_deg, roll_deg):
    y = np.radians(yaw_deg)
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx