import os, sys, numpy as np, cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector
from tools import to_H, inv_H, rot_mat_to_euler_zyx

# If you're on Wayland you can also do:
# os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

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
    axis = np.float32([[0,0,0],[length_m,0,0],[0,length_m,0],[0,0,length_m]])
    imgpts,_ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    p0 = tuple(imgpts[0].ravel().astype(int))
    cv2.line(img, p0, tuple(imgpts[1].ravel().astype(int)), (0,0,255), 2)
    cv2.line(img, p0, tuple(imgpts[2].ravel().astype(int)), (0,255,0), 2)
    cv2.line(img, p0, tuple(imgpts[3].ravel().astype(int)), (255,0,0), 2)

def detect_ids_with_size(gray_undist, camera_params, tag_size_m, allowed_ids):
    out = {}
    dets = detector.detect(gray_undist, estimate_tag_pose=True,
                           camera_params=camera_params, tag_size=tag_size_m)
    for det in dets:
        tid = int(det.tag_id)
        if allowed_ids and tid not in allowed_ids: continue
        R = np.array(det.pose_R, float);  t = np.array(det.pose_t, float).reshape(3,1)
        H = to_H(R, t);  rvec,_ = cv2.Rodrigues(R)
        out[tid] = (H, rvec, t, det.corners.astype(int), tag_size_m)
    return out

def make_vis_for_cam(cap, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    newK,_ = cv2.getOptimalNewCameraMatrix(cap.K, cap.D, gray.shape[::-1], 1.0)
    undist = cv2.undistort(gray, cap.K, cap.D, None, newK)
    vis = cv2.cvtColor(undist, cv2.COLOR_GRAY2BGR)
    campar = (newK[0,0], newK[1,1], newK[0,2], newK[1,2])

    world_map = detect_ids_with_size(undist, campar, WORLD_TAG_SIZE, {WORLD_TAG_ID})
    obj_map   = detect_ids_with_size(undist, campar, OBJ_TAG_SIZE, OBJ_TAG_IDS)
    H_c_by_id = {**world_map, **obj_map}

    for tid,(H, rvec, tvec, corners, size_used) in H_c_by_id.items():
        for k in range(4):
            cv2.line(vis, tuple(corners[k]), tuple(corners[(k+1)%4]), (0,255,255), 2)
        draw_axes(vis, newK, np.zeros(5), rvec, tvec, 0.5*size_used)
        cv2.putText(vis, f"{cap.serial} | TAG {tid}", (10, 20+20*tid),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

    y0 = 60
    if WORLD_TAG_ID not in H_c_by_id:
        cv2.putText(vis, "worldtag 0 NOT VISIBLE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
    else:
        H_0c = inv_H(H_c_by_id[WORLD_TAG_ID][0])
        for row, tid in enumerate(sorted(OBJ_TAG_IDS)):
            yb = y0 + 24*row
            if tid not in H_c_by_id:
                cv2.putText(vis, f"tag {tid} not visible", (10, yb),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
                continue
            H_0i = H_0c @ H_c_by_id[tid][0]
            tx,ty,tz = H_0i[:3,3]
            yaw,pitch,roll = rot_mat_to_euler_zyx(H_0i[:3,:3], degrees=True)
            cv2.putText(vis, f"tag{tid}@tag0 x:{tx:+.3f} y:{ty:+.3f} z:{tz:+.3f} yaw:{yaw:+.1f} pit:{pitch:+.1f} rol:{roll:+.1f}",
                        (10, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,220,50), 2)
    return vis

# --- open cams
caps = []
for sn in SERIALS:
    try: caps.append(RealSenseColorCap(sn, FRAME_W, FRAME_H, FPS))
    except Exception as e: print(f"[cam] open failed {sn}: {e}")
if not caps: raise RuntimeError("No cameras opened.")

# --- create a single HighGUI window explicitly
WIN = "AprilTag REL Pose — both cams"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(WIN, 2*FRAME_W//2 + 40, FRAME_H + 80)

print("Press q/Esc to quit.")
while True:
    visuals = []
    for cap in caps:
        ok, frame = cap.read()
        if not ok or frame is None:
            # placeholder if a cam misses a frame
            visuals.append(np.zeros((FRAME_H, FRAME_W, 3), np.uint8))
            continue
        vis = make_vis_for_cam(cap, frame)
        visuals.append(vis)

    # hstack all camera views (pad to same size if needed)
    max_h = max(v.shape[0] for v in visuals)
    visuals = [cv2.copyMakeBorder(v, 0, max_h - v.shape[0], 0, 0,
                                  cv2.BORDER_CONSTANT, value=(0,0,0)) for v in visuals]
    mosaic = cv2.hconcat(visuals) if len(visuals) > 1 else visuals[0]

    # tiny sanity overlay: show mean intensity so you know it's not “all zeros”
    mean_val = float(mosaic.mean())
    cv2.putText(mosaic, f"mean:{mean_val:.1f}", (10, mosaic.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    cv2.imshow(WIN, mosaic)
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q')): break

for c in caps: c.release()
cv2.destroyAllWindows()
