# single_cam_apriltag_pose_tag0_rel.py
import os, sys, numpy as np, cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector
from tools import to_H, inv_H, rot_mat_to_euler_zyx

# ========= CONFIG =========
SERIAL = "839112062097"
FRAME_W, FRAME_H, FPS = 1280, 720, 30

WORLD_TAG_SIZE = 0.138   # meters (ID 0)
OBJ_TAG_SIZE   = 0.032   # meters (IDs 1,2)
WORLD_TAG_ID = 0
OBJ_TAG_IDS = {1, 2}

# ========= AprilTag detector =========
detector = Detector(
    families="tag25h9",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
)

# ========= RealSense color wrapper =========
class RealSenseColorCap:
    def __init__(self, serial, w, h, fps):
        self.serial = serial
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(str(serial))
        self.cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.profile = self.pipe.start(self.cfg)

        # Extract intrinsics once at startup
        vsp = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        intr = vsp.get_intrinsics()
        self.K = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.D = np.array(intr.coeffs[:5], dtype=np.float32)
        print(f"Opened RealSense RGB S/N {serial}: {intr.width}x{intr.height}@{fps}")
        print(f"Intrinsics read from device:\n{self.K}\nDistortion: {self.D}")

    def read(self):
        frames = self.pipe.wait_for_frames()
        color = frames.get_color_frame()
        if not color: return False, None
        return True, np.asanyarray(color.get_data())

    def release(self):
        try: self.pipe.stop()
        except Exception: pass

# ========= Drawing =========
def draw_axes(img, K, dist, rvec, tvec, length_m):
    axis = np.float32([[0,0,0], [length_m,0,0], [0,length_m,0], [0,0,length_m]])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    p0 = tuple(imgpts[0].ravel().astype(int))
    cv2.line(img, p0, tuple(imgpts[1].ravel().astype(int)), (0,0,255), 2)
    cv2.line(img, p0, tuple(imgpts[2].ravel().astype(int)), (0,255,0), 2)
    cv2.line(img, p0, tuple(imgpts[3].ravel().astype(int)), (255,0,0), 2)

def detect_ids_with_size(gray_undist, camera_params, tag_size_m, allowed_ids):
    out = {}
    dets = detector.detect(
        gray_undist,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size_m
    )
    for det in dets:
        tid = int(det.tag_id)
        if allowed_ids and (tid not in allowed_ids):
            continue
        R_c_i = np.array(det.pose_R, dtype=float)
        t_c_i = np.array(det.pose_t, dtype=float).reshape(3,1)
        H_c_i = to_H(R_c_i, t_c_i)
        rvec, _ = cv2.Rodrigues(R_c_i)
        out[tid] = (H_c_i, rvec, t_c_i, det.corners.astype(int), tag_size_m)
    return out

# ========= Init camera =========
cap = RealSenseColorCap(SERIAL, FRAME_W, FRAME_H, FPS)

print("Press 'q' or ESC to quit.")
while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("[warn] grab failed")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    newK, _ = cv2.getOptimalNewCameraMatrix(cap.K, cap.D, gray.shape[::-1], 1.0)
    undist = cv2.undistort(gray, cap.K, cap.D, None, newK)
    vis = cv2.cvtColor(undist, cv2.COLOR_GRAY2BGR)

    camera_params = (newK[0,0], newK[1,1], newK[0,2], newK[1,2])

    # Detect world and object tags with their respective sizes
    world_map = detect_ids_with_size(undist, camera_params, WORLD_TAG_SIZE, {WORLD_TAG_ID})
    obj_map = detect_ids_with_size(undist, camera_params, OBJ_TAG_SIZE, OBJ_TAG_IDS)
    H_c_by_id = {**world_map, **obj_map}

    # Draw all tags
    for tid, (H_c_i, rvec, tvec, corners, size_used) in H_c_by_id.items():
        for k in range(4):
            cv2.line(vis, tuple(corners[k]), tuple(corners[(k+1)%4]), (0,255,255), 2)
        draw_axes(vis, newK, np.zeros(5), rvec, tvec, 0.5*size_used)
        cv2.putText(vis, f"TAG {tid} ({size_used*1000:.0f}mm)",
                    (10, 25 + 25*tid), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

    # Compute relative poses in tag0 frame
    if WORLD_TAG_ID not in H_c_by_id:
        cv2.putText(vis, "worldtag (ID 0) NOT VISIBLE — cannot compute relative poses",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
    else:
        H_c_0 = H_c_by_id[WORLD_TAG_ID][0]
        H_0_c = inv_H(H_c_0)
        for row, tid in enumerate(sorted(OBJ_TAG_IDS)):
            y_base = 60 + 24*row
            if tid not in H_c_by_id:
                cv2.putText(vis, f"TAG {tid} not visible", (10, y_base),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
                continue

            H_c_i = H_c_by_id[tid][0]
            H_0_i = H_0_c @ H_c_i
            tx, ty, tz = H_0_i[:3, 3]
            yaw, pitch, roll = rot_mat_to_euler_zyx(H_0_i[:3,:3], degrees=True)

            cv2.putText(vis,
                        f"REL tag{tid} in tag0 | x:{tx:+.3f} y:{ty:+.3f} z:{tz:+.3f}  yaw:{yaw:+.1f} pitch:{pitch:+.1f} roll:{roll:+.1f}",
                        (10, y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,220,50), 2)

    cv2.imshow("AprilTag REL Pose (tag0 frame)", vis)
    if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
