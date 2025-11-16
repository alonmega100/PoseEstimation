import numpy as np
import pandas as pd
from os import listdir
# from os.path import isfile, join


csv_files = [f for f in listdir("CSV") if f.endswith(".csv")]
sorted_files = sorted(csv_files)

print(sorted_files)


num = input("File to load? 0 for the first and so on...\n Press Enter for the last one\n :")

if not num:
    num = -1
# Load your big CSV
print("You chose ", sorted_files[int(num)])
df = pd.read_csv("CSV/" + sorted_files[int(num)])

cam1 = df[df['source'] == '839112062097']
cam2 = df[df['source'] == '845112070338']
robot = df[df['source'] == 'robot']


# ---------- config ----------
WORLD_TAG_ID = 2.0                 # change if your world tag id differs
TIME_TOL = pd.Timedelta('30ms')    # cam↔robot pairing tolerance
SAVE_PATH = "DATA/hand_eye/cam_to_robot_transform.npz"  # optional

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


def stack_and_fit_ransac(
    cam_list, robot_df, tag_id,
    time_tol=pd.Timedelta('30ms'),
    ransac_thresh=0.01,         # meters
    max_iters=3000,
    min_inliers_ratio=0.3,
    random_state=None
):
    """Build stacked A,B from both cams (like stack_and_fit) and run RANSAC."""
    A_list, B_list = [], []
    for cam_df in cam_list:
        cam_tag = cam_df[(cam_df['event'] == 'tag_pose_snapshot') & (cam_df['tag_id'] == tag_id)]
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
    # Attach counts to stats for transparency
    stats.update({"N_pairs_stacked": int(len(A))})
    return R, t, stats, inliers

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
    m = pd.merge_asof(L, R, on="timestamp", direction="nearest",
                      tolerance=tol, suffixes=("_cam","_rob"))
    return m.dropna(subset=["pose_03_rob"])

def stack_and_fit(cam_list, robot_df, tag_id=WORLD_TAG_ID):
    A_list, B_list = [], []
    for cam_df in cam_list:
        cam_tag = cam_df[(cam_df["event"]=="tag_pose_snapshot") & (cam_df["tag_id"]==tag_id)]
        m = nearest_merge(cam_tag, robot_df)
        if m.empty:
            continue
        A_list.append(pick_xyz(m, "_cam"))
        B_list.append(pick_xyz(m, "_rob"))
    if not A_list:
        raise RuntimeError("No cam↔robot matches found (check tag_id or TIME_TOL).")
    A = np.vstack(A_list); B = np.vstack(B_list)
    R, t = rigid_transform(A, B)
    aligned = (R @ A.T).T + t
    err = np.linalg.norm(aligned - B, axis=1)
    stats = {"N": int(len(A)), "mean_err": float(err.mean()),
             "median_err": float(np.median(err)), "max_err": float(err.max())}
    return R, t, stats

# ---------- parse timestamps now (so merge_asof is fast) ----------
# If df already exists from your loader:
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Split sources
cam1 = df[df['source'] == '839112062097']
cam2 = df[df['source'] == '845112070338']
robot = df[df['source'] == 'robot'][(df["event"] == "pose_snapshot")]

# Fit per camera (optional diagnostics)
def fit_one(cam_df, name):
    cam_tag = cam_df[(cam_df["event"]=="tag_pose_snapshot") & (cam_df["tag_id"]==WORLD_TAG_ID)]
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

# Combined fit (both cams share the same world tag → one transform)
# R, t, stats = stack_and_fit([cam1, cam2], robot, WORLD_TAG_ID)
# print("\n=== Combined cams -> Robot transform ===")
# print("R =\n", R)
# print("t =", t)
# print("stats =", stats)
# --- RANSAC-based combined fit ---
R, t, stats, inliers = stack_and_fit_ransac(
    [cam1, cam2], robot, WORLD_TAG_ID,
    time_tol=TIME_TOL,
    ransac_thresh=0.01,      # tweak: 0.005–0.02 m typical
    max_iters=3000,
    min_inliers_ratio=0.3,
    random_state=42
)

print("\n=== RANSAC cams -> Robot transform ===")
print("R =\n", R)
print("t =", t)
print("stats =", stats)

# save like before
try:
    import os; os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez(SAVE_PATH, R=R, t=t, stats=stats)
    print(f"Saved to {SAVE_PATH}")
except Exception as e:
    print("Skip save:", e)



# After this, you're in interactive mode (because you launched with `python -i`).
# Try:
# >>> R
# >>> t
# >>> stats
# Exit with Ctrl+D.


# print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns.")
# print("DataFrame is available as variable 'df'.")