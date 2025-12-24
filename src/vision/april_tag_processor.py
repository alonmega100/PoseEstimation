import numpy as np
import cv2

from pupil_apriltags import Detector
from src.utils.tools import to_H, inv_H
from src.utils.config import (FRAME_W, FRAME_H, WORLD_TAG_ID, WORLD_TAG_SIZE, OBJ_TAG_SIZE,
                              OBJ_TAG_IDS)
from src.vision.realsense_driver import RealSenseInfraredCap



class AprilTagProcessor:
    """
    Handles tag detection, pose calculation, and image visualization for a single camera.
    """

    def __init__(self, serial: str):
        self.world_tag_size = WORLD_TAG_SIZE
        self.obj_tag_size = OBJ_TAG_SIZE
        self.obj_tag_ids = OBJ_TAG_IDS
        self.cap = RealSenseInfraredCap(serial)

        # NOTE: RealSense handles distortion internally, so we use camera intrinsics for reference only
        # The frames returned from cap.read() are already undistorted by RealSense hardware

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
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return np.zeros((FRAME_H, FRAME_W, 3), np.uint8), {}

        # NOTE: Frame is already undistorted by RealSense hardware, no need for cv2.remap()
        undist = frame

        # Convert to BGR only for visualization
        vis = cv2.cvtColor(undist, cv2.COLOR_GRAY2BGR)

        # Use the original camera matrix (RealSense intrinsics)
        campar = (self.cap.K[0, 0], self.cap.K[1, 1], self.cap.K[0, 2], self.cap.K[1, 2])

        # --- Detection ---
        world_map = self._detect_tags(undist, campar, self.world_tag_size, {WORLD_TAG_ID})
        obj_map = self._detect_tags(undist, campar, self.obj_tag_size, self.obj_tag_ids)
        H_c_by_id = {**world_map, **obj_map}

        # ... (Rest of the function logic for H0i and drawing remains exactly the same) ...

        # (Be sure to include the logic for calculating H0i, drawing axes, and returning vis, H0i here)
        H0i = {}
        world_tag_visible = WORLD_TAG_ID in H_c_by_id

        for tid, (H, rvec, tvec, corners, size_used) in H_c_by_id.items():
            for k in range(4):
                cv2.line(vis, tuple(corners[k]), tuple(corners[(k + 1) % 4]), (0, 255, 255), 2)
            self._draw_axes(vis, self.cap.K, rvec, tvec, 0.5 * size_used)

            if world_tag_visible and tid in self.obj_tag_ids:
                H_0c = inv_H(H_c_by_id[WORLD_TAG_ID][0])
                H_0i = H_0c @ H_c_by_id[tid][0]
                H_0i = self._canonicalize_pose_z_positive(H_0i)
                H0i[tid] = H_0i

                # ... (Drawing text logic) ...
                tx, ty, tz = H_0i[:3, 3]
                cv2.putText(vis, f"tag{tid} x:{tx:+.3f} y:{ty:+.3f} z:{tz:+.3f}",
                            (int(corners[0][0]), int(corners[0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 50), 2)

        if not world_tag_visible:
            cv2.putText(vis, "WORLD TAG 0 NOT VISIBLE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        return vis, H0i

    def release(self):
        self.cap.release()