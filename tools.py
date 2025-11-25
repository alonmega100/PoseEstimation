import numpy as np
from typing import Optional, Tuple


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


def matrix_to_flat_dict(prefix, mat):
    mat = np.array(mat)  # safe conversion
    out = {}
    for r in range(4):
        for c in range(4):
            out[f"{prefix}_{r}{c}"] = float(mat[r, c])
    return out


def list_of_movements_generator(
    num_commands: int,
    bounds = {"x": (-0.1, 0.3), "y": (-0.3, 0.0), "z": (-0.1, 0.0)},
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


def parse_vn_vnrrg_08(line: str) -> Optional[Tuple[float, float, float]]:
    if not line.startswith("$VNRRG,08"):
        return None
    try:
        data_part = line.split("*", 1)[0]
        parts = data_part.split(",")
        if len(parts) < 5:
            return None
        yaw = float(parts[2])
        pitch = float(parts[3])
        roll = float(parts[4])
        return yaw, pitch, roll
    except Exception:
        return None
