#april_tag_processor.py
import numpy as np

import cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R
from tools import to_H, inv_H
from config import FRAME_W, FRAME_H, FPS, WORLD_TAG_ID, WORLD_TAG_SIZE, OBJ_TAG_SIZE, OBJ_TAG_IDS


class RealSenseInfraredCap:
    def __init__(self, serial, w, h, fps):
        self.serial = serial
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(str(serial))

        # Stream Infrared 1 (Left Imager = Depth Origin)
        self.cfg.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, fps)

        self.profile = self.pipe.start(self.cfg)

        # --- NEW CODE: DISABLE LASER EMITTER ---
        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()

        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0.0)  # 0 = Off
        # ---------------------------------------

        # Get intrinsics
        vsp = rs.video_stream_profile(self.profile.get_stream(rs.stream.infrared, 1))
        intr = vsp.get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx],
                           [0, intr.fy, intr.ppy],
                           [0, 0, 1]], np.float32)
        self.D = np.array(intr.coeffs[:5], np.float32)
        print(f"[cam-IR] {serial}: {intr.width}x{intr.height}@{fps} (Emitter OFF)")
    def read(self):
        frames = self.pipe.wait_for_frames()
        # MODIFIED: Get infrared frame instead of color
        ir = frames.get_infrared_frame(1)
        if not ir: return False, None
        # Return the data directly (it is already a numpy array of uint8)
        return True, np.asanyarray(ir.get_data())

    def release(self):
        try:
            self.pipe.stop()
        except:
            pass

class AprilTagProcessor:
    """
    Handles tag detection, pose calculation, and image visualization for a single camera.
    """

    def __init__(self, serial: str, world_tag_size: float, obj_tag_size: float, obj_tag_ids: set,
                 w=FRAME_W, h=FRAME_H, fps=FPS):

        self.WORLD_TAG_SIZE = world_tag_size
        self.OBJ_TAG_SIZE = obj_tag_size
        self.OBJ_TAG_IDS = obj_tag_ids
        self.cap = RealSenseInfraredCap(serial, w, h, fps)

        # Initialize detector once
        self.detector = Detector(
            families="tag25h9", nthreads=4, quad_decimate=1.0, quad_sigma=0.0,
            refine_edges=True, decode_sharpening=0.25,
        )

    def _detect_tags(self, gray_undist: np.ndarray, camera_params: tuple, tag_size_m: float, allowed_ids: set):
        """Helper to run detector and convert pose to H matrix."""
        out = {}
        dets = self.detector.detect(gray_undist, estimate_tag_pose=True,
                                    camera_params=camera_params, tag_size=tag_size_m)
        for det in dets:
            tid = int(det.tag_id)
            if allowed_ids and tid not in allowed_ids:
                continue
            R = np.array(det.pose_R, float)
            t = np.array(det.pose_t, float).reshape(3, 1)
            H = to_H(R, t)
            rvec, _ = cv2.Rodrigues(R)
            out[tid] = (H, rvec, t, det.corners.astype(int), tag_size_m)
        return out
    def _canonicalize_pose_z_positive(self, H: np.ndarray) -> np.ndarray:
        """
        Enforce a deterministic, right-handed orientation for a tag pose H.

        We require that the tag's z-axis (third column of R) has non-negative
        projection on the world z axis, i.e. z_world[2] >= 0. If not, we
        rotate the tag frame by 180° about its x-axis, which flips y,z
        but keeps det(R)=+1.
        """
        H = H.copy()
        z_world = H[:3, 2]        # 3rd column = tag's z axis in world frame
        if z_world[2] < 0:
            # rotate tag frame by 180° about its local x-axis
            H[:3, :3] = H[:3, :3] @ np.diag([1, -1, -1])
        return H
    def _draw_axes(self, img, K, rvec, tvec, length_m):
        """Draws axes on the image for visualization."""
        axis = np.float32([[0, 0, 0], [length_m, 0, 0], [0, length_m, 0], [0, 0, length_m]])
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, np.zeros(5))  # Use zeros for D since image is undistorted
        imgpts = imgpts.reshape(-1, 2).astype(int)

        p0 = tuple(imgpts[0])
        pX, pY, pZ = map(tuple, imgpts[1:])

        # Draw axes (X=red, Y=green, Z=blue)
        cv2.line(img, p0, pX, (0, 0, 255), 2)
        cv2.line(img, p0, pY, (0, 255, 0), 2)
        cv2.line(img, p0, pZ, (255, 0, 0), 2)

    def process_frame(self) -> tuple[np.ndarray, dict]:
        """
        Reads a frame, computes tag poses relative to the World Tag (H0i),
        and returns the visualization image and the dictionary of poses.

        Returns: vis_img, H0i_dict[tag_id] = H_0_i (4x4)
        """
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return np.zeros((FRAME_H, FRAME_W, 3), np.uint8), {}

        # --- Pre-processing ---
        gray = frame
        newK, _ = cv2.getOptimalNewCameraMatrix(self.cap.K, self.cap.D, gray.shape[::-1], 1.0)
        undist = cv2.undistort(gray, self.cap.K, self.cap.D, None, newK)
        vis = cv2.cvtColor(undist, cv2.COLOR_GRAY2BGR)
        campar = (newK[0, 0], newK[1, 1], newK[0, 2], newK[1, 2])

        # --- Detection ---
        world_map = self._detect_tags(undist, campar, self.WORLD_TAG_SIZE, {WORLD_TAG_ID})
        obj_map = self._detect_tags(undist, campar, self.OBJ_TAG_SIZE, self.OBJ_TAG_IDS)
        H_c_by_id = {**world_map, **obj_map}

        # --- Visualization & Pose Calculation ---
        H0i = {}
        world_tag_visible = WORLD_TAG_ID in H_c_by_id

        # 1. Draw all detected tags
        for tid, (H, rvec, tvec, corners, size_used) in H_c_by_id.items():
            for k in range(4):
                cv2.line(vis, tuple(corners[k]), tuple(corners[(k + 1) % 4]), (0, 255, 255), 2)
            self._draw_axes(vis, newK, rvec, tvec, 0.5 * size_used)

            # 2. Calculate H0i (Tag relative to World Anchor)
            # (Inside AprilTagProcessor.process_frame, where H_0i is calculated)

            if world_tag_visible and tid in self.OBJ_TAG_IDS:
                H_0c = inv_H(H_c_by_id[WORLD_TAG_ID][0])
                H_0i = H_0c @ H_c_by_id[tid][0]

                H_0i = self._canonicalize_pose_z_positive(H_0i)

                H0i[tid] = H_0i
                tx, ty, tz = H_0i[:3, 3]

                R_matrix = H_0i[:3, :3]

                # Calculate ZYX Euler angles (Yaw, Pitch, Roll)
                # The order 'zyx' matches the common industrial robotics standard
                r_obj = R.from_matrix(R_matrix)
                yaw, pitch, roll = r_obj.as_euler('zyx', degrees=True)
                # -----------------------------------------------------------------

                # Draw position text (and optional rotation text)
                rot_text = f"yaw:{yaw:+.1f} pit:{pitch:+.1f} rol:{roll:+.1f}"

                cv2.putText(vis, f"tag{tid}@tag0 x:{tx:+.3f} y:{ty:+.3f} z:{tz:+.3f}",
                            (int(corners[0][0] / 1.25), int(corners[0][1] + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 50), 2)
                # If you want the rotation displayed:
                # cv2.putText(vis, rot_text, (int(corners[0][0] / 1.25), int(corners[0][1] + 40)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 50), 2)


        # 3. Status messages
        if not world_tag_visible:
            cv2.putText(vis, "WORLD TAG 0 NOT VISIBLE - NO RELATIVE POSES", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.putText(vis, f"S/N: {self.cap.serial}", (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        return vis, H0i

    def release(self):
        self.cap.release()