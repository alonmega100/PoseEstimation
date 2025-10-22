
import os, sys, numpy as np, cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector
from tools import to_H, inv_H, rot_mat_to_euler_zyx

SERIALS = ["839112062097", "845112070338"]
FRAME_W, FRAME_H, FPS = 1280, 720, 30
WORLD_TAG_SIZE = 0.138
OBJ_TAG_SIZE   = 0.032
WORLD_TAG_ID = 0
OBJ_TAG_IDS = {1, 2}

detector = Detector(
    families="tag25h9",
    nthreads=4, quad_decimate=1.0, quad_sigma=0.0,
    refine_edges=True, decode_sharpening=0.25,
)

# ---------- small helpers ----------
def rot_angle_deg(R):
    """Angle of rotation (deg) from a 3x3 rotation matrix."""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

def pose_delta(H_a, H_b):
    """Delta between two 4x4 poses: H_delta = inv(H_a) @ H_b -> (pos_err_m, ang_err_deg)."""
    H = np.linalg.inv(H_a) @ H_b
    pos_err = float(np.linalg.norm(H[:3, 3]))
    ang_err = rot_angle_deg(H[:3, :3])
    return pos_err, ang_err, H

# ---------- camera wrapper ----------
class RealSenseColorCap:
    def __init__(self, serial, w, h, fps):
        self.serial = serial
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(str(serial))
        self.cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.profile = self.pipe.start(self.cfg)
        vsp = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        intr = vsp.get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx],
                           [0, intr.fy, intr.ppy],
                           [0, 0, 1]], np.float32)
        self.D = np.array(intr.coeffs[:5], np.float32)
        print(f"[cam] {serial}: {intr.width}x{intr.height}@{fps}")

    def read(self):
        frames = self.pipe.wait_for_frames()
        c = frames.get_color_frame()
        if not c: return False, None
        return True, np.asanyarray(c.get_data())

    def release(self):
        try: self.pipe.stop()
        except: pass

def draw_axes(img, K, dist, rvec, tvec, length_m):
    # 3D axis points: origin + endpoints
    axis = np.float32([
        [0, 0, 0],
        [length_m, 0, 0],   # X
        [0, length_m, 0],   # Y
        [0, 0, length_m]    # Z
    ])

    # Project 3D points to 2D image
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    p0 = tuple(imgpts[0])  # origin
    pX, pY, pZ = map(tuple, imgpts[1:])

    # Draw axes (X=red, Y=green, Z=blue)
    cv2.line(img, p0, pX, (0, 0, 255), 2)
    cv2.line(img, p0, pY, (0, 255, 0), 2)
    cv2.line(img, p0, pZ, (255, 0, 0), 2)

    # Add axis labels slightly beyond the tips
    offset = 10  # pixels
    cv2.putText(img, "X", (pX[0] + offset, pX[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, "Y", (pY[0] + offset, pY[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "Z", (pZ[0] + offset, pZ[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def detect_ids_with_size(gray_undist, camera_params, tag_size_m, allowed_ids):
    out = {}
    dets = detector.detect(gray_undist, estimate_tag_pose=True,
                           camera_params=camera_params, tag_size=tag_size_m)
    for det in dets:
        tid = int(det.tag_id)
        if allowed_ids and tid not in allowed_ids:
            continue
        R = np.array(det.pose_R, float)
        t = np.array(det.pose_t, float).reshape(3,1)
        H = to_H(R, t)
        rvec,_ = cv2.Rodrigues(R)
        out[tid] = (H, rvec, t, det.corners.astype(int), tag_size_m)
    return out

def compute_H0i_map_for_cam(cap, frame):
    """
    Returns:
      vis_img, H0i_dict: dict[tag_id] = H_0_i (4x4) in tag-0 frame from THIS camera
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    newK,_ = cv2.getOptimalNewCameraMatrix(cap.K, cap.D, gray.shape[::-1], 1.0)
    undist = cv2.undistort(gray, cap.K, cap.D, None, newK)
    vis = cv2.cvtColor(undist, cv2.COLOR_GRAY2BGR)
    campar = (newK[0,0], newK[1,1], newK[0,2], newK[1,2])

    world_map = detect_ids_with_size(undist, campar, WORLD_TAG_SIZE, {WORLD_TAG_ID})
    obj_map   = detect_ids_with_size(undist, campar, OBJ_TAG_SIZE, OBJ_TAG_IDS)
    H_c_by_id = {**world_map, **obj_map}

    # draw
    for tid,(H, rvec, tvec, corners, size_used) in H_c_by_id.items():
        for k in range(4):
            cv2.line(vis, tuple(corners[k]), tuple(corners[(k+1)%4]), (0,255,255), 2)
        draw_axes(vis, newK, np.zeros(5), rvec, tvec, 0.5*size_used)
        cv2.putText(vis, f"{cap.serial} | TAG {tid}", (10, 20+20*tid),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

    H0i = {}
    if WORLD_TAG_ID not in H_c_by_id:
        cv2.putText(vis, "worldtag 0 NOT VISIBLE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
    else:
        H_0c = inv_H(H_c_by_id[WORLD_TAG_ID][0])
        y0 = 80
        for row, tid in enumerate(sorted(OBJ_TAG_IDS)):
            yb = y0 + 24*row
            if tid not in H_c_by_id:
                cv2.putText(vis, f"tag {tid} not visible", (10, yb),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
                continue
            H_0i = H_0c @ H_c_by_id[tid][0]
            H0i[tid] = H_0i
            tx,ty,tz = H_0i[:3,3]

            yaw,pitch,roll = rot_mat_to_euler_zyx(H_0i[:3,:3], degrees=True)
            cv2.putText(vis, f"tag{tid}@tag0 x:{tx:+.3f} y:{ty:+.3f} z:{tz:+.3f}",
                        (int((corners[0])[0]/1.25),int((corners[0])[1]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,220,50), 2)
            # cv2.putText(vis, f"tag{tid}@tag0 x:{tx:+.3f} y:{ty:+.3f} z:{tz:+.3f} yaw:{yaw:+.1f} pit:{pitch:+.1f} rol:{roll:+.1f}",
            #             (int((corners[0])[0]/2),(corners[0])[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,220,50), 2)
    return vis, H0i

# ----- open cams -----
caps = []
for sn in SERIALS:
    try:
        caps.append(RealSenseColorCap(sn, FRAME_W, FRAME_H, FPS))
    except Exception as e:
        print(f"[cam] open failed {sn}: {e}")
if len(caps) != 2:
    raise RuntimeError("Expected exactly 2 cameras for comparison.")

# ----- window -----
WIN = "AprilTag REL Pose — comparison (cam A | cam B)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(WIN, 2*FRAME_W//2 + 80, FRAME_H + 100)

print("Press q/Esc to quit.")
while True:
    frames = []
    for cap in caps:
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        frames.append(frame)

    vis_list, H0i_list = [], []
    for cap, frame in zip(caps, frames):
        vis, H0i = compute_H0i_map_for_cam(cap, frame)
        # stamp cam serial
        cv2.putText(vis, f"S/N: {cap.serial}", (10, vis.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        vis_list.append(vis)
        H0i_list.append(H0i)

    # compare if both cams have same tag
    A, B = H0i_list[0], H0i_list[1]
    delta_lines = []
    for tid in sorted(OBJ_TAG_IDS):
        if tid in A and tid in B:
            pos_err, ang_err, H_delta = pose_delta(A[tid], B[tid])
            delta_lines.append(f"tag{tid}: dpos={pos_err*1000:6.1f} mm | dang={ang_err:5.2f} deg")
        else:
            missing = "A" if tid not in A else "B"
            delta_lines.append(f"tag{tid}: (missing in cam {missing})")

    max_h = max(v.shape[0] for v in vis_list)
    vis_list = [cv2.copyMakeBorder(v, 0, max_h - v.shape[0], 0, 0,
                                   cv2.BORDER_CONSTANT, value=(0,0,0)) for v in vis_list]
    mosaic = cv2.hconcat(vis_list)

    y = 100
    cv2.putText(mosaic, "Cross-camera delta (A vs B) in tag0 frame:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    y += 26
    for line in delta_lines:
        cv2.putText(mosaic, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        y += 24

    # console print (optional)
    if delta_lines:
        print(" | ".join(delta_lines))

    cv2.imshow(WIN, mosaic)
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q')): break

for c in caps: c.release()
cv2.destroyAllWindows()
