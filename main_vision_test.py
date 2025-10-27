import sys
import cv2
import numpy as np
from april_tag_processor import AprilTagProcessor
from tools import inv_H

# --- CONFIGURATION CONSTANTS ---
SERIALS = ["839112062097", "845112070338"]
FRAME_W, FRAME_H = 1280, 720
WORLD_TAG_SIZE = 0.138
OBJ_TAG_SIZE = 0.032
OBJ_TAG_IDS = {1, 2}
WORLD_TAG_ID = 0


def pose_delta(H_a: np.ndarray, H_b: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Delta between two 4x4 poses: H_delta = inv(H_a) @ H_b -> (pos_err_m, ang_err_deg)."""
    H = inv_H(H_a) @ H_b
    pos_err = float(np.linalg.norm(H[:3, 3]))
    ang_err = rot_angle_deg(H[:3, :3])
    return pos_err, ang_err, H

def rot_angle_deg(R: np.ndarray) -> float:
    """Angle of rotation (deg) from a 3x3 rotation matrix."""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def run_vision_comparison():
    """Sets up cameras, runs the processing loop, displays results, and handles cleanup."""
    processors = []
    try:
        # ----- Open Cameras (INIT) -----
        for sn in SERIALS:
            try:
                processors.append(AprilTagProcessor(
                    serial=sn,
                    world_tag_size=WORLD_TAG_SIZE,
                    obj_tag_size=OBJ_TAG_SIZE,
                    obj_tag_ids=OBJ_TAG_IDS
                ))
            except Exception as e:
                print(f"[cam] Open failed {sn}: {e}", file=sys.stderr)  # Print errors to stderr

        # This check is good but should allow the system to proceed if it's just being tested
        if len(processors) < 1:
            raise RuntimeError("Failed to open any cameras. Cannot run vision comparison.")

        if len(processors) != 2:
            print("[warning] Running with less or more than 2 cameras. Comparison logic may be incomplete.")

        # ----- Window Setup -----
        WIN = "AprilTag REL Pose — comparison (cam A | cam B)"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(WIN, 2 * FRAME_W // 2 + 80, FRAME_H + 100)

        print("Press q/Esc to quit.")

        # ----- Main Processing Loop -----
        while True:
            vis_list, H0i_list = [], []

            # Process frames
            for processor in processors:
                vis, H0i = processor.process_frame()
                vis_list.append(vis)
                H0i_list.append(H0i)

            # --- Cross-Camera Comparison (Only if 2 cameras are present) ---
            delta_lines = []
            if len(H0i_list) == 2:
                A, B = H0i_list[0], H0i_list[1]

                for tid in sorted(OBJ_TAG_IDS):
                    if tid in A and tid in B:
                        pos_err, ang_err, H_delta = pose_delta(A[tid], B[tid])
                        delta_lines.append(f"tag{tid}: dpos={pos_err * 1000:6.1f} mm | dang={ang_err:5.2f} deg")
                    else:
                        missing = "A" if tid not in A else "B"
                        delta_lines.append(f"tag{tid}: (missing in cam {missing})")

                if delta_lines:
                    print(" | ".join(delta_lines))
            # -------------------------------------------------------------

            # --- Mosaic and Display ---
            if vis_list:
                # Ensure numpy is imported for these operations if not already used elsewhere
                max_h = max(v.shape[0] for v in vis_list)
                vis_list_padded = [cv2.copyMakeBorder(v, 0, max_h - v.shape[0], 0, 0,
                                                      cv2.BORDER_CONSTANT, value=(0, 0, 0)) for v in vis_list]
                mosaic = cv2.hconcat(vis_list_padded)

                # Add delta text to the mosaic (only if comparison was run)
                if len(H0i_list) == 2:
                    y = 100
                    cv2.putText(mosaic, "Cross-camera delta (A vs B) in tag0 frame:", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y += 26
                    for line in delta_lines:
                        cv2.putText(mosaic, line, (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                        y += 24

                cv2.imshow(WIN, mosaic)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')): break

    except Exception as e:
        print(f"[Fatal Error] {e}", file=sys.stderr)

    finally:
        # ----- Cleanup -----
        for p in processors: p.release()
        cv2.destroyAllWindows()


def main():
    run_vision_comparison()


if __name__ == "__main__":
    main()