import numpy as np
import pandas as pd
import csv
import json
import os
from scipy.spatial.transform import Rotation as R
from src.utils.config import CAMERA_SERIALS
from src.utils.tools import pose_row_to_matrix

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
RANSAC_ITERATIONS = 1000
RANSAC_THRESHOLD = 0.02  # 2cm threshold for inliers
TIME_SYNC_TOLERANCE = 0.1  # 100ms max diff
CSV_FILE_PATH = "data/CSV/session_log_20251224_155001.csv"
CALIBRATION_DIR = "data/DATA/hand_eye"


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def load_data(csv_path):
    if not os.path.exists(csv_path):
        from src.utils.tools import find_latest_csv
        found_csv = find_latest_csv()
        if found_csv:
            print(f"File {csv_path} not found. Using latest: {found_csv}")
            csv_path = found_csv
        else:
            raise FileNotFoundError("No CSV file found.")

    df = pd.read_csv(csv_path)
    df['t_epoch'] = pd.to_datetime(df['timestamp']).astype(int) / 1e9

    robot_data = []
    camera_data = {sn: [] for sn in CAMERA_SERIALS}

    for _, row in df.iterrows():
        src = str(row['source'])
        try:
            mat = pose_row_to_matrix(row, prefix="pose_")
            pos = mat[:3, 3]
        except Exception:
            continue

        t = row['t_epoch']
        if src == 'robot':
            robot_data.append({'time': t, 'pos': pos})
        elif src in CAMERA_SERIALS:
            camera_data[src].append({'time': t, 'pos': pos})

    return robot_data, camera_data


def sync_data(robot_list, cam_list):
    if not robot_list or not cam_list:
        return np.empty((0, 3)), np.empty((0, 3))

    df_r = pd.DataFrame(robot_list).sort_values('time')
    df_c = pd.DataFrame(cam_list).sort_values('time')

    merged = pd.merge_asof(
        df_c, df_r, on='time', direction='nearest',
        suffixes=('_cam', '_rob'), tolerance=TIME_SYNC_TOLERANCE
    ).dropna()

    if merged.empty:
        return np.empty((0, 3)), np.empty((0, 3))

    return np.vstack(merged['pos_cam'].values), np.vstack(merged['pos_rob'].values)


def fit_rigid_transform(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R_est = np.dot(Vt.T, U.T)
    if np.linalg.det(R_est) < 0:
        Vt[2, :] *= -1
        R_est = np.dot(Vt.T, U.T)
    t_est = centroid_B - np.dot(R_est, centroid_A)
    return R_est, t_est


def ransac_rigid_transform(pts_cam, pts_rob):
    n_points = pts_cam.shape[0]
    if n_points < 4:
        return None, None, 0

    best_inliers_count = 0
    best_R = np.eye(3)
    best_t = np.zeros(3)
    best_inliers_mask = np.zeros(n_points, dtype=bool)

    for i in range(RANSAC_ITERATIONS):
        indices = np.random.choice(n_points, 3, replace=False)
        src = pts_cam[indices]
        dst = pts_rob[indices]

        try:
            R_curr, t_curr = fit_rigid_transform(src, dst)
        except np.linalg.LinAlgError:
            continue

        pts_rob_est = (np.dot(R_curr, pts_cam.T).T) + t_curr
        errors = np.linalg.norm(pts_rob - pts_rob_est, axis=1)
        inliers_mask = errors < RANSAC_THRESHOLD
        inliers_count = np.sum(inliers_mask)

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_R = R_curr
            best_t = t_curr
            best_inliers_mask = inliers_mask

    if best_inliers_count > 3:
        final_R, final_t = fit_rigid_transform(pts_cam[best_inliers_mask], pts_rob[best_inliers_mask])
        return final_R, final_t, best_inliers_count
    return best_R, best_t, best_inliers_count


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    print(f"Loading data from {CSV_FILE_PATH}...")
    robot_data, camera_data_dict = load_data(CSV_FILE_PATH)

    # Ensure output directory exists
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    print(f"Loaded {len(robot_data)} robot poses. Output dir: {CALIBRATION_DIR}")

    for serial in CAMERA_SERIALS:
        cam_list = camera_data_dict.get(serial, [])
        if not cam_list:
            print(f"\n[Camera {serial}] No data found.")
            continue

        pts_cam, pts_rob = sync_data(robot_data, cam_list)
        n_samples = pts_cam.shape[0]

        print(f"\n[Camera {serial}] {n_samples} sync samples found.")
        if n_samples < 10:
            print("  -> Not enough samples.")
            continue

        R_est, t_est, n_inliers = ransac_rigid_transform(pts_cam, pts_rob)

        if R_est is not None:
            print(f"  -> RANSAC Success! Inliers: {n_inliers}/{n_samples}")

            # Construct 4x4 Matrix
            H = np.eye(4)
            H[:3, :3] = R_est
            H[:3, 3] = t_est

            # Save to NPZ file
            filename = f"cam_{serial}_to_robot_transform.npz"
            filepath = os.path.join(CALIBRATION_DIR, filename)

            # Saving with key 'transform'
            np.savez(filepath, transform=H)

            print(f"  -> Saved to {filepath}")
        else:
            print("  -> RANSAC failed.")


if __name__ == "__main__":
    main()