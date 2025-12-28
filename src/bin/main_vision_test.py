import sys
import argparse
from typing import List, Optional
from src.vision.april_tag_processor import AprilTagProcessor
from src.vision.vision_display import VisionDisplay
from src.utils.config import CAMERA_SERIALS, SHOW_DISPLAY


def run_vision_comparison(show_display: bool = False) -> None:
    """Open cameras, run AprilTag processing, and optionally display feeds using VisionDisplay.

    Graceful exits:
      - If show_display=True, VisionDisplay can quit on q/Esc.
      - Ctrl+C (KeyboardInterrupt) always exits cleanly (no traceback).
    """
    processors: List[AprilTagProcessor] = []
    display: Optional[VisionDisplay] = None

    try:
        # ----- Open Cameras (INIT) -----
        for serial_num in CAMERA_SERIALS:
            try:
                processors.append(AprilTagProcessor(serial=serial_num))
            except Exception as e:
                print(f"[cam] Open failed {serial_num}: {e}", file=sys.stderr)

        if len(processors) < 1:
            raise RuntimeError("Failed to open any cameras. Cannot run vision comparison.")

        # ----- Display Setup (VisionDisplay) -----
        if show_display:
            WIN = "AprilTag REL Pose — comparison"
            display = VisionDisplay(window_title=WIN)

        print("Press q/Esc to quit (if display is on). Press Ctrl+C to quit anytime.")

        # ----- Main Loop -----
        try:
            while True:
                vis_list, H0i_list = [], []

                for processor in processors:
                    vis, H0i = processor.process_frame()
                    vis_list.append(vis)
                    H0i_list.append(H0i)

                if show_display and display is not None:
                    # Update frames in display (by serial order)
                    for serial_num, vis in zip(CAMERA_SERIALS, vis_list):
                        display.update_frame(serial_num, vis)

                    # VisionDisplay handles q/Esc
                    if not display.show_mosaic():
                        break

        except KeyboardInterrupt:
            print("\n[info] Ctrl+C received. Exiting gracefully...")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="AprilTag vision comparison")
    parser.add_argument(
        "-vision",
        type=str,
        default="Off",
        choices=["On", "Off", "on", "off", "ON", "OFF"],
        help="Enable vision display window (On/Off)"
    )


    args = parser.parse_args()

    show_display = args.vision.lower() == "on"

    run_vision_comparison(show_display=show_display)

if __name__ == "__main__":
    main()
