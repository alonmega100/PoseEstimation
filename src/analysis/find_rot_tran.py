import numpy as np
import pandas as pd
from os import listdir
import os
import sys
from src.utils.tools import rot_geodesic_angle_deg
from src.utils.config import WORLD_TAG_ID

# ---------- config ----------
# Paths relative to project root
CSV_DIR = "data/CSV"
SAVE_PATH = "data/DATA/hand_eye/cam_to_robot_transform.npz"

TIME_TOL = pd.Timedelta('30ms')    # cam↔robot pairing tolerance


def rpy_to_R_deg(yaw_deg, pitch_deg, roll_deg):
    """
    Build a 3x3 rotation matrix from yaw, pitch, roll in degrees
    using ZYX (yaw-pitch-roll) convention.
    """
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

# Check if directory exists
if not os.path.exists(CSV_DIR):
    print(f"[ERROR] CSV directory not found at: {os.path.abspath(CSV_DIR)}")
    sys.exit(1)

csv_files = [f for f in listdir(CSV_DIR) if f.endswith(".csv")]
sorted_files = sorted(csv_files)

if not sorted_files:
    print(f"[ERROR] No .csv files found in {CSV_DIR}")
    sys.exit(1)

print("Available CSV files:")
for i, f in enumerate(sorted_files):
    print(f"  {i}: {f}")

num = input(
    "File to load? 0 for the first and so on...\n"
    " Press Enter for the last one\n :"
)

if not num.strip():
    idx = -1
else:
    idx = int(num)

chosen_file = sorted_files[idx]
print("You chose ", chosen_file)
df = pd.read_csv(os.path.join(CSV_DIR, chosen_file))


# ---------- helpers ----------

def ransac_rigid_transform(
    A, B,
    sample_size=3,              # 3 points are enough for a rigid 3D transform
    thresh=0.01,                # in meters (1 cm). Tune to your noise level
    max_iters=3000,
    min_inliers_ratio=0.3,      # require at least this fraction as inliers
    random_state=None
):
    """
    Robustly fit R,t s.t. R@A + t ≈ B using RANSAC.
    Returns: R (3x3), t (3,), stats dict, inliers_mask (bool[N])
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    assert A.shape == B.shape and A.shape[1] == 3
    N = len(A)
    if N < sample_size:
        raise ValueError(f"Not enough points for RANSAC: N={N} < sample_size={sample_size}")

    rng = np.random.default_rng(random_state)
    best_inliers = None
    best_count = -1
    best_err = np.inf
    best_R = None
    best_t = None

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
            # track the best by inlier count, tie-break by mean error
            if count >= sample_size:
                pred_i = (R_try @ A[inliers].T).T + t_try
                err_mean = float(np.linalg.norm(pred_i - B[inliers], axis=1).mean())
            else:
                err_mean = np.inf

            best_count = count
            best_err = err_mean
            best_inliers = inliers
            best_R, best_t = R_try, t_try

    if best_inliers is None or best_count < max(sample_size, int(min_inliers_ratio * N)):
        raise RuntimeError(
            f"RANSAC failed: best_inliers={best_count} of {N}. "
            f"Try increasing thresh or max_iters."
        )

    # Refit on all inliers for final R,t
    R, t = rigid_transform(A[best_inliers], B[best_inliers])
    pred = (R @ A.T).T + t
    resid = np.linalg.norm(pred - B, axis=1)

    stats = {
        "N_total": int(N),
        "inliers": int(best_inliers.sum()),
        "inlier_ratio": float(best_inliers.mean()),
        "thresh": float(thresh),
        "rms_all": float(np.sqrt((resid**2).mean())),
        "mean_all": float(resid.mean()),
        "median_all": float(np.median(resid)),
        "max_all": float(resid.max()),
        "rms_inliers": float(np.sqrt((resid[best_inliers]**2).mean())),
        "mean_inliers": float(resid[best_inliers].mean()),
        "median_inliers": float(np.median(resid[best_inliers])),
        "max_inliers": float(resid[best_inliers].max()),
    }
    return R, t, stats, best_inliers


def pick_xyz(df_like, suffix=""):
    cols = [f"pose_03{suffix}", f"pose_13{suffix}", f"pose_23{suffix}"]
    return df_like[cols].to_numpy()


def rigid_transform(A, B):
    """Least-squares R,t s.t. R@A + t ≈ B."""
    A = np.asarray(A); B = np.asarray(B)
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    AA, BB = A - ca, B - cb
    U, S, Vt = np.linalg.svd(AA.T @ BB)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # reflection fix
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    return R, t


def nearest_merge(cam_df, rob_df, tol=TIME_TOL):
    """Nearest-in-time merge; keeps only matches within tol."""
    L = cam_df.sort_values("timestamp").copy()
    R = rob_df.sort_values("timestamp").copy()
    m = pd.merge_asof(
        L, R,
        on="timestamp",
        direction="nearest",
        tolerance=tol,
        suffixes=("_cam", "_rob")
    )
    return m.dropna(subset=["pose_03_rob"])

def nearest_merge_imu(imu_df, rob_df, tol=TIME_TOL):
    """
    Nearest-in-time merge for IMU vs robot.

    We only require IMU orientation (imu_yaw_deg, imu_pitch_deg, imu_roll_deg)
    and robot pose (pose_ij_rob). IMU position is ignored.
    """
    L = imu_df.sort_values("timestamp").copy()
    R = rob_df.sort_values("timestamp").copy()
    m = pd.merge_asof(
        L,
        R,
        on="timestamp",
        direction="nearest",
        tolerance=tol,
        suffixes=("", "_rob"),  # keep IMU cols unchanged; robot pose gets _rob
    )

    needed = [
        "imu_yaw_deg",
        "imu_pitch_deg",
        "imu_roll_deg",
        "pose_00_rob",
        "pose_01_rob",
        "pose_02_rob",
        "pose_10_rob",
        "pose_11_rob",
        "pose_12_rob",
        "pose_20_rob",
        "pose_21_rob",
        "pose_22_rob",
    ]
    m = m.dropna(subset=[c for c in needed if c in m.columns])
    return m

def stack_and_fit_ransac(
    cam_list, robot_df, tag_id,
    time_tol=pd.Timedelta('30ms'),
    ransac_thresh=0.01,         # meters
    max_iters=3000,
    min_inliers_ratio=0.3,
    random_state=None
):
    """Build stacked A,B from cams and run RANSAC."""
    A_list, B_list = [], []
    for cam_df in cam_list:
        cam_tag = cam_df[
            (cam_df['event'] == 'tag_pose_snapshot') &
            (cam_df['tag_id'] == tag_id)
        ]
        m = nearest_merge(cam_tag, robot_df, tol=time_tol)
        if m.empty:
            continue
        A_list.append(pick_xyz(m, "_cam"))
        B_list.append(pick_xyz(m, "_rob"))

    if not A_list:
        raise RuntimeError("No cam↔robot matches found (check tag_id or TIME_TOL).")

    A = np.vstack(A_list)
    B = np.vstack(B_list)

    R, t, stats, inliers = ransac_rigid_transform(
        A, B,
        sample_size=3,
        thresh=ransac_thresh,
        max_iters=max_iters,
        min_inliers_ratio=min_inliers_ratio,
        random_state=random_state
    )
    stats.update({"N_pairs_stacked": int(len(A))})
    return R, t, stats, inliers


# ---------- parse timestamps ----------
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Split sources
robot = df[df['source'] == 'robot'][(df["event"] == "pose_snapshot")]

# For compatibility: keep these if you still want them
cam1 = df[df['source'] == '839112062097']
cam2 = df[df['source'] == '845112070338']


# Optional diagnostics: plain LSQ per camera
def fit_one(cam_df, name):
    cam_tag = cam_df[
        (cam_df["event"] == "tag_pose_snapshot") &
        (cam_df["tag_id"] == WORLD_TAG_ID)
    ]
    m = nearest_merge(cam_tag, robot)
    if m.empty:
        print(f"[{name}] no matches")
        return None
    R, t = rigid_transform(pick_xyz(m, "_cam"), pick_xyz(m, "_rob"))
    aligned = (R @ pick_xyz(m, "_cam").T).T + t
    e = np.linalg.norm(aligned - pick_xyz(m, "_rob"), axis=1)
    print(f"[{name}] N={len(e)} | mean={e.mean():.6f} | median={np.median(e):.6f} | max={e.max():.6f}")
    return R, t


fit_one(cam1, "cam1")
fit_one(cam2, "cam2")


# --- Combined RANSAC fit using both cams (unchanged) ---
# NOTE: This block might fail if cam1/cam2 don't exist in your specific CSV
# You might want to wrap this in a try/except or check if they are empty
if not cam1.empty and not cam2.empty:
    try:
        R, t, stats, inliers = stack_and_fit_ransac(
            [cam1, cam2],
            robot,
            WORLD_TAG_ID,
            time_tol=TIME_TOL,
            ransac_thresh=0.01,
            max_iters=3000,
            min_inliers_ratio=0.3,
            random_state=42
        )

        print("\n=== RANSAC (cam1+cam2) -> Robot transform ===")
        print("R =\n", R)
        print("t =", t)
        print("stats =", stats)

        # save combined transform (as before)
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        np.savez(SAVE_PATH, R=R, t=t, stats=stats, source="combined")
        print(f"Saved combined transform to {SAVE_PATH}")
    except Exception as e:
        print("Skip save combined:", e)
else:
    print("\n[INFO] Skipping hardcoded cam1+cam2 combined fit (cameras not found in CSV).")


# --- NEW: per-camera RANSAC + save one file per SN ---

def fit_cam_ransac_and_save(cam_df, robot_df, src, tag_id=WORLD_TAG_ID):
    """Run RANSAC for a single camera and save to cam_<SN>_to_robot_transform.npz."""
    if cam_df.empty:
        print(f"[{src}] no rows for this camera, skipping.")
        return

    cam_tag = cam_df[
        (cam_df["event"] == "tag_pose_snapshot") &
        (cam_df["tag_id"] == tag_id)
    ]
    if cam_tag.empty:
        print(f"[{src}] no tag_pose_snapshot rows for tag_id={tag_id}, skipping.")
        return

    try:
        R_cam, t_cam, stats_cam, inliers_cam = stack_and_fit_ransac(
            [cam_df],
            robot_df,
            tag_id,
            time_tol=TIME_TOL,
            ransac_thresh=0.02,
            max_iters=3000,
            min_inliers_ratio=0.3,
            random_state=42
        )
    except Exception as e:
        print(f"[{src}] RANSAC failed: {e}")
        return

    print(f"\n=== RANSAC {src} -> Robot transform ===")
    print("R =\n", R_cam)
    print("t =", t_cam)
    print("stats =", stats_cam)

    save_path = f"data/DATA/hand_eye/cam_{src}_to_robot_transform.npz"
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(
            save_path,
            R=R_cam,
            t=t_cam,
            stats=stats_cam,
            source=src
        )
        print(f"Saved {src} transform to {save_path}")
    except Exception as e:
        print(f"[{src}] Skip save (error):", e)


def fit_imu_rotation_and_save(imu_df, robot_df, src, tag_for_print="IMU"):
    """
    Estimate a single rotation R_cal such that:

        R_robot_i  ≈  R_cal @ R_imu_i
    """
    if imu_df.empty:
        print(f"[{tag_for_print} {src}] no rows for this source, skipping.")
        return

    required_cols = {"imu_yaw_deg", "imu_pitch_deg", "imu_roll_deg"}
    if not required_cols.issubset(imu_df.columns):
        missing = required_cols - set(imu_df.columns)
        print(f"[{tag_for_print} {src}] missing columns {missing}, skipping.")
        return

    # Time-align IMU ↔ robot
    m = nearest_merge_imu(imu_df, robot_df, tol=TIME_TOL)
    if m.empty:
        print(f"[{tag_for_print} {src}] no imu↔robot matches within {TIME_TOL}, skipping.")
        return

    # --- Build rotation pairs (R_robot_i, R_imu_i) ---
    R_robot_list = []
    R_imu_list = []

    for _, row in m.iterrows():
        # Robot rotation from pose_ij_rob
        Rr = np.array([
            [row["pose_00_rob"], row["pose_01_rob"], row["pose_02_rob"]],
            [row["pose_10_rob"], row["pose_11_rob"], row["pose_12_rob"]],
            [row["pose_20_rob"], row["pose_21_rob"], row["pose_22_rob"]],
        ], dtype=float)

        # IMU rotation matrix from yaw/pitch/roll (deg) in CSV
        yaw = float(row["imu_yaw_deg"])
        pitch = float(row["imu_pitch_deg"])
        roll = float(row["imu_roll_deg"])
        Ri = rpy_to_R_deg(yaw, pitch, roll)

        R_robot_list.append(Rr)
        R_imu_list.append(Ri)

    if not R_robot_list:
        print(f"[{tag_for_print} {src}] no valid IMU/robot orientation pairs, skipping.")
        return

    R_robot_arr = np.stack(R_robot_list, axis=0)  # (N,3,3)
    R_imu_arr = np.stack(R_imu_list, axis=0)      # (N,3,3)

    # --- Estimate R_cal via orthogonal Procrustes on rotations ---
    # We want: R_robot ≈ R_cal @ R_imu  ⇒  R_cal ≈ argmin || R_cal R_imu - R_robot ||
    M = np.zeros((3, 3), dtype=float)
    for Rr, Ri in zip(R_robot_arr, R_imu_arr):
        M += Rr @ Ri.T

    U, _, Vt = np.linalg.svd(M)
    R_cal = U @ Vt
    if np.linalg.det(R_cal) < 0:
        # Reflection fix
        U[:, -1] *= -1
        R_cal = U @ Vt

    # --- Compute geodesic angle errors after applying R_cal ---
    ang_errs = []
    for Rr, Ri in zip(R_robot_arr, R_imu_arr):
        R_imu_aligned = R_cal @ Ri
        ang = rot_geodesic_angle_deg(Rr, R_imu_aligned)
        ang_errs.append(ang)

    ang_errs = np.asarray(ang_errs, dtype=float)
    mean_ang = float(np.mean(ang_errs))
    median_ang = float(np.median(ang_errs))
    p95_ang = float(np.percentile(ang_errs, 95))
    rms_ang = float(np.sqrt(np.mean(ang_errs ** 2)))
    max_ang = float(np.max(ang_errs))

    stats = {
        "N_pairs": int(len(ang_errs)),
        "mean_deg": mean_ang,
        "median_deg": median_ang,
        "p95_deg": p95_ang,
        "rms_deg": rms_ang,
        "max_deg": max_ang,
    }

    # Pretty printing
    print("\n" + "=" * 72)
    print(f"=== IMU '{src}' → Robot | Orientation-only Calibration ===")
    print("=" * 72)
    print("Estimated rotation (R_cal):")
    print(R_cal)
    print("\nAngle residual statistics (after applying R_cal):")
    print(f"  Samples used      : {len(ang_errs)}")
    print(f"  Mean error        : {mean_ang:.3f}°")
    print(f"  Median error      : {median_ang:.3f}°")
    print(f"  95th percentile   : {p95_ang:.3f}°")
    print(f"  RMS error         : {rms_ang:.3f}°")
    print(f"  Max error         : {max_ang:.3f}°")


    # Save R_cal with dummy t=[0,0,0] for compatibility
    save_path = f"data/DATA/hand_eye/imu_{src}_to_robot_transform.npz"
    t_dummy = np.zeros(3, dtype=float)
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(
            save_path,
            R=R_cal,
            t=t_dummy,
            stats=stats,
            source=str(src),
        )
        print(f"[{tag_for_print} {src}] Saved orientation transform to {save_path}")
    except Exception as e:
        print(f"[{tag_for_print} {src}] Skip save (error):", e)

# Find all sources
all_sources = sorted(df["source"].unique())

# Cameras: anything that's not robot and not an IMU
camera_sources = [
    s for s in all_sources
    if s != "robot" and not str(s).lower().startswith("imu")
]

# IMUs: sources whose name starts with "imu" (e.g. "imu", "imu_vn100", etc.)
imu_sources = [
    s for s in all_sources
    if str(s).lower().startswith("imu")
]

# --- Per-camera RANSAC (unchanged behavior) ---
for src in camera_sources:
    cam_df = df[df["source"] == src]
    fit_cam_ransac_and_save(cam_df, robot, src)

# --- IMU orientation calibration ---
for src in imu_sources:
    imu_df = df[df["source"] == src]
    fit_imu_rotation_and_save(imu_df, robot, src)