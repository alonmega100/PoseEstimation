import sys
import time
import threading
from typing import List, Optional
from src.vision.april_tag_processor import AprilTagProcessor
from src.utils.config import CAMERA_SERIALS, SHOW_DISPLAY

# Import VisionDisplay only if needed
if SHOW_DISPLAY:
    from src.vision.vision_display import VisionDisplay

class TagLogger:
    def __init__(self, writer):
        self.writer = writer
        self.processors: List[AprilTagProcessor] = []
        self.display: Optional[VisionDisplay] = None
        self.running = False  # Control flag for the loop
        self.thread: Optional[threading.Thread] = None

        try:
            # ----- Open Cameras (INIT) -----
            for serial_num in CAMERA_SERIALS:
                try:
                    self.processors.append(AprilTagProcessor(serial=serial_num))
                except Exception as e:
                    print(f"[cam] Open failed {serial_num}: {e}", file=sys.stderr)

            if len(self.processors) < 1:
                raise RuntimeError("Failed to open any cameras. Cannot run vision comparison.")



        except Exception as e:
            print(f"[Fatal Error Init] {e}", file=sys.stderr)
            # If init fails, ensure we don't try to run
            self.processors = []

    def start(self):
        """Starts the logging thread."""
        if not self.processors:
            print("[Error] No cameras initialized. Cannot start.")
            return

        self.running = True
        # daemon=False ensures the process waits for this thread (safer for cameras)
        # However, we will control exit via the main loop anyway.
        self.thread = threading.Thread(target=self.log, daemon=False)
        self.thread.start()
        print("[TagLogger] Thread started.")

    def stop(self):
        """Signals the thread to stop and waits for it to join."""
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()
        print("[TagLogger] Stopped.")

    def log(self):
        """The main loop running inside the thread."""
        try:
            # ----- Display Setup (VisionDisplay) -----
            if SHOW_DISPLAY:
                WIN = "AprilTag REL Pose — comparison"
                self.display = VisionDisplay(window_title=WIN)
            while self.running:
                vis_list, Hci_list = [], []

                for processor in self.processors:
                    vis, Hci = processor.process_frame()
                    vis_list.append(vis)
                    print(Hci)
                    Hci_list.append(Hci)
                    print(Hci_list)
                    if not self.writer.write_enabled and vis is not None:
                        time.sleep(0.002)
                        continue
                    mat = Hci[2][0]
                    R= mat[:3,:3]
                    t = mat[:3,3]
                    self.writer.add_pose_estimation((R,t))

                if SHOW_DISPLAY and self.display is not None:
                    # Update frames in display
                    for serial_num, vis in zip(CAMERA_SERIALS, vis_list):
                        self.display.update_frame(serial_num, vis)


                    if not self.display.show_mosaic():
                        print("[TagLogger] User requested quit via Display.")
                        self.running = False
                        break
                time.sleep(0.01)

        except Exception as e:
            print(f"[Thread Error] {e}", file=sys.stderr)

        finally:
            self._cleanup()

    def _cleanup(self):
        """Internal cleanup method."""
        print("[TagLogger] Cleaning up resources...")
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

def main():
    logger = TagLogger()
    logger.start()

    # --- MAIN THREAD LOOP ---
    # We must keep the main thread alive while the background thread runs.
    try:
        while logger.running:
            # If the thread dies unexpectedly (or finishes via 'q'), exit main loop
            if logger.thread and not logger.thread.is_alive():
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C received. Stopping logger...")
        logger.stop()

if __name__ == '__main__':
    main()