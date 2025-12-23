import sys
import numpy as np
from typing import List, Tuple, Optional

from src.vision.april_tag_processor import AprilTagProcessor
from src.vision.vision_display import VisionDisplay
from src.utils.tools import inv_H
from src.utils.config import CAMERA_SERIALS, FRAME_W, FRAME_H


def rot_angle_deg(R: np.ndarray) -> float:
    """Angle of rotation (deg) from a 3x3 rotation matrix."""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def pose_delta(H_a: np.ndarray, H_b: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Delta between two 4x4 poses: H_delta = inv(H_a) @ H_b -> (pos_err_m, ang_err_deg)."""
    H = inv_H(H_a) @ H_b
    pos_err = float(np.linalg.norm(H[:3, 3]))
    ang_err = rot_angle_deg(H[:3, :3])
    return pos_err, ang_err, H


def run_vision_comparison():
    """Open cameras, run AprilTag processing, and display feeds using VisionDisplay."""
    processors: List[AprilTagProcessor] = []
    display: Optional[VisionDisplay] = None

    try:
        # ----- Open Cameras (INIT) -----
        for serial_num in CAMERA_SERIALS:
            try:
                processors.append(
                    AprilTagProcessor(serial=serial_num)
                )
            except Exception as e:
                print(f"[cam] Open failed {serial_num}: {e}", file=sys.stderr)

        if len(processors) < 1:
            raise RuntimeError("Failed to open any cameras. Cannot run vision comparison.")

        if len(processors) != 2:
            print("[warning] Running with != 2 cameras. Cross-camera delta overlay will be shown only when 2 cameras are present.")

        # ----- Display Setup (VisionDisplay) -----
        WIN = "AprilTag REL Pose — comparison"
        display = VisionDisplay(window_title=WIN)

        print("Press q/Esc to quit.")

        # ----- Main Loop -----
        while True:
            vis_list, H0i_list = [], []

            for processor in processors:
                vis, H0i = processor.process_frame()
                vis_list.append(vis)
                H0i_list.append(H0i)

            # Update frames in display (by serial order)
            for serial_num, vis in zip(CAMERA_SERIALS, vis_list):
                display.update_frame(serial_num, vis)


            # print(vis_list)
            # Build global overlay text (cross-camera delta)
            # overlay_global = []
            # if len(H0i_list) == 2:
            #     A, B = H0i_list[0], H0i_list[1]
            #     overlay_global.append("Cross-camera delta (A vs B) in tag0 frame:")
            #     for tid in sorted(OBJ_TAG_IDS):
            #         if tid in A and tid in B:
            #             pos_err, ang_err, _ = pose_delta(A[tid], B[tid])
            #             overlay_global.append(f"tag{tid}: dpos={pos_err * 1000:6.1f} mm | dang={ang_err:5.2f} deg")
            #         else:
            #             missing = "A" if tid not in A else "B"
            #             overlay_global.append(f"tag{tid}: (missing in cam {missing})")

            # Show mosaic; VisionDisplay handles q/Esc
            if not display.show_mosaic(): #overlay_global_text=overlay_global if overlay_global else None):
                break

    except Exception as e:
        print(f"[Fatal Error] {e}", file=sys.stderr)

    finally:
        for p in processors:
            try:
                p.release()
            except Exception:
                pass
        if display is not None:
            try:
                display.cleanup()
            except Exception:
                pass


def main():
    run_vision_comparison()


if __name__ == "__main__":
    main()