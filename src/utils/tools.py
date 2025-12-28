import numpy as np
import pandas as pd
import os
import glob
from typing import Dict, List, Optional, Any, Tuple
from src.utils.config import CAMERA_SERIALS

# ---------------------------------------------------------------------
# MATH & GEOMETRY HELPERS
# ---------------------------------------------------------------------

def to_H(R, t):
    H = np.eye(4);
    H[:3, :3] = R;
    H[:3, 3] = t.reshape(3)
    return H


def inv_H(H):
    Ri = H[:3, :3].T
    ti = -Ri @ H[:3, 3]
    Hi = np.eye(4);
    Hi[:3, :3] = Ri;
    Hi[:3, 3] = ti
    return Hi


def H_to_xyzrpy_ZYX(H):
    R = H[:3, :3];
    x, y, z = H[:3, 3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-9:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return np.array([x, y, z, roll, pitch, yaw])


def rpy_to_R_deg(yaw_deg, pitch_deg, roll_deg):
    y, p, r = np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def rot_geodesic_angle_deg(R1, R2):
    R = R1.T @ R2
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def pose_row_to_matrix(row: Any, prefix: str = "pose_") -> np.ndarray:
    H = np.eye(4, dtype=float)
    for r in range(4):
        for c in range(4):
            key = f"{prefix}{r}{c}"
            if key in row:
                H[r, c] = float(row[key])
    return H


def rigid_transform(A, B):
    """Least-squares R,t s.t. R@A + t ≈ B."""
    A = np.asarray(A);
    B = np.asarray(B)
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    AA, BB = A - ca, B - cb
    U, S, Vt = np.linalg.svd(AA.T @ BB)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    return R, t


def ransac_rigid_transform(
        A, B, sample_size=3, thresh=0.01, max_iters=3000, min_inliers_ratio=0.3, random_state=None
):
    """Robustly fit R,t s.t. R@A + t ≈ B using RANSAC."""
    A = np.asarray(A, float);
    B = np.asarray(B, float)
    N = len(A)
    if N < sample_size:
        raise ValueError(f"Not enough points: {N} < {sample_size}")

    rng = np.random.default_rng(random_state)
    best_inliers = None
    best_count = -1

    for _ in range(max_iters):
        idx = rng.choice(N, size=sample_size, replace=False)
        try:
            R_try, t_try = rigid_transform(A[idx], B[idx])
        except np.linalg.LinAlgError:
            continue

        pred = (R_try @ A.T).T + t_try
        resid = np.linalg.norm(pred - B, axis=1)
        inliers = resid < thresh
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_inliers = inliers
            if count == N: break  # Perfect fit

    if best_inliers is None or best_count < max(sample_size, int(min_inliers_ratio * N)):
        raise RuntimeError(f"RANSAC failed: best_inliers={best_count}/{N}")

    # Final Refinement
    R, t = rigid_transform(A[best_inliers], B[best_inliers])
    pred = (R @ A.T).T + t
    resid = np.linalg.norm(pred - B, axis=1)

    stats = {
        "N_total": int(N), "inliers": int(best_inliers.sum()),
        "mean_error": float(resid[best_inliers].mean()),
        "rmse": float(np.sqrt((resid[best_inliers] ** 2).mean()))
    }
    return R, t, stats, best_inliers


# ---------------------------------------------------------------------
# PANDAS / DATA HELPERS
# ---------------------------------------------------------------------

def nearest_merge(cam_df, rob_df, time_col="timestamp", tol=pd.Timedelta('30ms'), suffix_cam="", suffix_rob="_rob"):
    """Align two dataframes by timestamp."""
    L = cam_df.sort_values(time_col).copy()
    R = rob_df.sort_values(time_col).copy()

    # Ensure datetime
    if not np.issubdtype(L[time_col].dtype, np.datetime64):
        L[time_col] = pd.to_datetime(L[time_col])
    if not np.issubdtype(R[time_col].dtype, np.datetime64):
        R[time_col] = pd.to_datetime(R[time_col])

    return pd.merge_asof(
        L, R, on=time_col, direction="nearest", tolerance=tol, suffixes=(suffix_cam, suffix_rob)
    )


def moving_average(values, window):
    if window <= 1: return list(values)
    out = []
    cumsum = 0.0
    for i, v in enumerate(values):
        cumsum += v
        if i >= window: cumsum -= values[i - window]
        if i >= window - 1:
            out.append(cumsum / window)
        else:
            out.append(float("nan"))
    return out


# ---------------------------------------------------------------------
# IO & TRANSFORMS
# ---------------------------------------------------------------------
def matrix_to_flat_dict(prefix, mat):
    mat = np.array(mat)  # safe conversion
    out = {}
    for r in range(4):
        for c in range(4):
            out[f"{prefix}_{r}{c}"] = float(mat[r, c])
    return out


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


def choose_csv_interactively(csv_dir: str) -> str:
    if not os.path.exists(csv_dir):
        raise FileNotFoundError(f"CSV dir not found: {csv_dir}")
    files = sorted([f for f in glob.glob(os.path.join(csv_dir, "*.csv"))])
    if not files:
        raise FileNotFoundError(f"No CSVs in {csv_dir}")

    print(f"Available CSV files in {csv_dir}:")
    for i, f in enumerate(files):
        print(f"  {i}: {os.path.basename(f)}")

    idx = input("File # (Enter for last): ")
    return files[-1] if not idx.strip() else files[int(idx)]


def find_latest_csv(csv_dir="data/CSV") -> Optional[str]:
    files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not files: return None
    return max(files, key=os.path.getmtime)


def load_cam_to_robot_transforms(calib_dir):
    """
    Scans the directory for files named 'cam_{serial}_to_robot_transform.npz'
    and loads the 'transform' key into a dictionary: { serial: 4x4_numpy_matrix }
    """
    calibs = {}
    if not os.path.exists(calib_dir):
        print(f"Warning: Calibration directory {calib_dir} not found.")
        return calibs

    print(f"Scanning {calib_dir} for transforms...")

    # Regex to extract serial from filename: cam_(SERIAL)_to_robot_transform.npz
    # pattern = re.compile(r"cam_([a-zA-Z0-9]+)_to_robot_transform\.npz")
    for serial_number in CAMERA_SERIALS:
        for fname in os.listdir(calib_dir):
            # match = pattern.match(fname)
            file_name = fr"cam_{serial_number}_to_robot_transform.npz"
            if fname == file_name:
                full_path = os.path.join(calib_dir, fname)
                try:
                    # Load NPZ file
                    with np.load(full_path) as data:
                        if 'transform' in data:
                            calibs[serial_number] = data['transform']
                            print(f"  -> Loaded transform for camera {serial_number}")
                        else:
                            print(f"  -> Warning: 'transform' key missing in {fname}")
                except Exception as e:
                    print(f"  -> Failed to load {fname}: {e}")

    return calibs

def load_imu_to_robot_transform(imu_source: str, transform_dir: str) -> Optional[np.ndarray]:
    fpath = os.path.join(transform_dir, f"imu_{imu_source}_to_robot_transform.npz")
    if not os.path.exists(fpath): return None
    try:
        d = np.load(fpath, allow_pickle=True)
        return to_H(d["R"], d.get("t", np.zeros(3)))
    except:
        return None


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

def make_serializable(obj):
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    return obj