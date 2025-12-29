import sys
import threading

from typing import List, Optional
from src.vision.april_tag_processor import AprilTagProcessor
from src.utils.config import CAMERA_SERIALS, SHOW_DISPLAY
if SHOW_DISPLAY:
    from src.vision.vision_display import VisionDisplay


class TagLogger:
    def __init__(self):
        self.processors: List[AprilTagProcessor] = []
        self.display: Optional[VisionDisplay] = None
        self.thread = threading.Thread(target=self.log, daemon=True)
        try:
            # ----- Open Cameras (INIT) -----
            for serial_num in CAMERA_SERIALS:
                try:
                    self.processors.append(AprilTagProcessor(serial=serial_num))
                except Exception as e:
                    print(f"[cam] Open failed {serial_num}: {e}", file=sys.stderr)

            if len(self.processors) < 1:
                raise RuntimeError("Failed to open any cameras. Cannot run vision comparison.")

            # ----- Display Setup (VisionDisplay) -----
            if SHOW_DISPLAY:
                WIN = "AprilTag REL Pose — comparison"
                self.display = VisionDisplay(window_title=WIN)
        except Exception as e:
            print(f"[Fatal Error] {e}", file=sys.stderr)

    def start(self):
        # self.thread = threading.Thread(target=self.log, daemon=True)
        self.thread.start()

    def stop(self):
        self.thread.join()

    def log(self):
        try:
            while True:
                vis_list, Hci_list = [], []

                for processor in self.processors:
                    vis, Hci = processor.process_frame()
                    vis_list.append(vis)
                    Hci_list.append(Hci)
                    # print(Hci_list)

                if SHOW_DISPLAY and self.display is not None:
                    # Update frames in display (by serial order)
                    for serial_num, vis in zip(CAMERA_SERIALS, vis_list):
                        self.display.update_frame(serial_num, vis)

                    # VisionDisplay handles q/Esc
                    if not self.display.show_mosaic():
                        break
        except Exception as e:
            print(f"[Fatal Error] {e}", file=sys.stderr)

        finally:
            for p in self.processors:
                try:
                    p.release()
                except Exception:
                    pass
            if self.display is not None:
                try:
                    self.display.cleanup()
                except Exception:
                    pass

if __name__ == '__main__':
    t = TagLogger()
    t.start()