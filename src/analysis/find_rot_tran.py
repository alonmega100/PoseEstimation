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
TIME_SYNC_TOLERANCE = 0.1  # 100ms max diff between robot and camera
CSV_FILE_PATH = "data/CSV/session_log_20251224_155001.csv"  # Update this if needed, or use latest


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def load_data(csv_path):
    """
    Parses the CSV and splits data into:
      - robot_data: list of {'time': float, 'pos': np.array([x,y,z])}
      - camera_data: dict of serial -> list of {'time': float, 'pos': np.array([x,y,z])}
    """
    data = []

    # Check if file exists
    if not os.path.exists(csv_path):
        # Fallback to finding the latest csv if specific path not found
        from src.utils.tools import find_latest_csv
        found_csv = find_latest_csv()
        if found_csv:
            print(f"File {csv_path} not found. Using latest: {found_csv}")
            csv_path = found_csv
        else:
            raise FileNotFoundError("No CSV file found.")

    # Load into Pandas for easier handling
    df = pd.read_csv(csv_path)

    # Convert timestamp to seconds (float)
    # Assumes timestamp is ISO format strings
    df['t_epoch'] = pd.to_datetime(df['timestamp']).astype(int) / 1e9

    robot_data = []
    camera_data = {sn: [] for sn in CAMERA_SERIALS}

    for _, row in df.iterrows():
        src = str(row['source'])

        # 1. Parse Pose Matrix
        try:
            # Reconstruct 4x4 matrix from flat columns
            mat = pose_row_to_matrix(row, prefix="pose_")
            # Extract position (x, y, z)
            pos = mat[:3, 3]
        except Exception:
            continue

        # 2. Sort into Robot or Camera lists
        t = row['t_epoch']

        if src == 'robot':
            robot_data.append({'time': t, 'pos': pos})
        elif src in CAMERA_SERIALS:
            camera_data[src].append({'time': t, 'pos': pos})

    return robot_data, camera_data


def sync_data(robot_list, cam_list):
    """
    Matches each camera sample to the closest robot sample within tolerance.
    Returns: (points_cam, points_robot) as (N,3) arrays.
    """
    if not robot_list or not cam_list:
        return np.empty((0, 3)), np.empty((0, 3))

    # Convert to DataFrames for merge_asof
    df_r = pd.DataFrame(robot_list).sort_values('time')
    df_c = pd.DataFrame(cam_list).sort_values('time')

    # Merge closest robot time to camera time
    # direction='nearest' finds the closest timestamp
    merged = pd.merge_asof(
        df_c, df_r,
        on='time',
        direction='nearest',
        suffixes=('_cam', '_rob'),
        tolerance=TIME_SYNC_TOLERANCE
    )

    # Drop rows where no match was found (NaNs)
    merged = merged.dropna()

    if merged.empty:
        return np.empty((0, 3)), np.empty((0, 3))

    # Extract positions
    pts_cam = np.vstack(merged['pos_cam'].values)
    pts_rob = np.vstack(merged['pos_rob'].values)

    return pts_cam, pts_rob


def fit_rigid_transform(A, B):
    """
    Finds rotation R and translation t such that B = R @ A + t
    Using SVD (Kabsch algorithm).
    """
    assert A.shape == B.shape
    num_rows, dim = A.shape

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R_est = np.dot(Vt.T, U.T)

    # Reflection case
    if np.linalg.det(R_est) < 0:
        Vt[dim - 1, :] *= -1
        R_est = np.dot(Vt.T, U.T)

    t_est = centroid_B - np.dot(R_est, centroid_A)
    return R_est, t_est


def ransac_rigid_transform(pts_cam, pts_rob):
    """
    RANSAC loop to find best R, t ignoring outliers.
    """
    n_points = pts_cam.shape[0]
    if n_points < 4:
        print("Not enough points for RANSAC (<4)")
        return None, None, 0

    best_inliers_count = 0
    best_R = np.eye(3)
    best_t = np.zeros(3)
    best_inliers_mask = np.zeros(n_points, dtype=bool)

    for i in range(RANSAC_ITERATIONS):
        # 1. Sample 3 random points
        indices = np.random.choice(n_points, 3, replace=False)
        src_subset = pts_cam[indices]
        dst_subset = pts_rob[indices]

        # 2. Estimate model
        R_curr, t_curr = fit_rigid_transform(src_subset, dst_subset)

        # 3. Apply model to all points: P_est = R * P_cam + t
        pts_rob_est = (np.dot(R_curr, pts_cam.T).T) + t_curr

        # 4. Calculate errors
        errors = np.linalg.norm(pts_rob - pts_rob_est, axis=1)

        # 5. Count inliers
        inliers_mask = errors < RANSAC_THRESHOLD
        inliers_count = np.sum(inliers_mask)

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_R = R_curr
            best_t = t_curr
            best_inliers_mask = inliers_mask

    # Refine using ALL inliers
    if best_inliers_count > 3:
        final_R, final_t = fit_rigid_transform(
            pts_cam[best_inliers_mask],
            pts_rob[best_inliers_mask]
        )
        return final_R, final_t, best_inliers_count
    else:
        return best_R, best_t, best_inliers_count


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    print(f"Loading data from {CSV_FILE_PATH}...")
    robot_data, camera_data_dict = load_data(CSV_FILE_PATH)
    print(f"Loaded {len(robot_data)} robot poses.")

    print("\n------------------------------------------------")
    print(" running RANSAC Calibration for each camera")
    print("------------------------------------------------")

    for serial in CAMERA_SERIALS:
        cam_list = camera_data_dict.get(serial, [])
        if not cam_list:
            print(f"\n[Camera {serial}] No data found in CSV.")
            continue

        # 1. Sync
        pts_cam, pts_rob = sync_data(robot_data, cam_list)
        n_samples = pts_cam.shape[0]

        print(f"\n[Camera {serial}] Found {n_samples} synchronized samples.")
        if n_samples < 10:
            print("  -> Not enough matched samples to calibrate reliable.")
            continue

        # 2. RANSAC
        R_est, t_est, n_inliers = ransac_rigid_transform(pts_cam, pts_rob)

        if R_est is not None:
            print(f"  -> RANSAC Success! Inliers: {n_inliers}/{n_samples} ({(n_inliers / n_samples) * 100:.1f}%)")

            # Format output
            print("  -> Translation (x, y, z):")
            print(f"     {t_est}")

            r = R.from_matrix(R_est)
            print("  -> Rotation (Euler XYZ degrees):")
            print(f"     {r.as_euler('xyz', degrees=True)}")

            # Create 4x4 Homogeneous Matrix
            H_cam_to_base = np.eye(4)
            H_cam_to_base[:3, :3] = R_est
            H_cam_to_base[:3, 3] = t_est

            print("  -> Full Transform Matrix (Camera -> RobotBase):")
            print(np.array2string(H_cam_to_base, separator=', '))

            # Validation Error on Inliers
            pts_rob_est = (np.dot(R_est, pts_cam.T).T) + t_est
            errors = np.linalg.norm(pts_rob - pts_rob_est, axis=1)
            mean_error = np.mean(errors)
            print(f"  -> Mean Error (All Points): {mean_error * 1000:.2f} mm")

        else:
            print("  -> RANSAC failed.")


if __name__ == "__main__":
    main()