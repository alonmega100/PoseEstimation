import numpy as np
import pyrealsense2 as rs
from src.utils.config import FRAME_W, FRAME_H, FPS, GAIN, EXPOSURE


class RealSenseInfraredCap:
    """
    Encapsulates the Intel RealSense camera setup for the INFRARED (Depth) lens.
    Uses Infrared stream #1 (Left Imager), which aligns with the depth origin.
    """

    def __init__(self, serial: str, w=FRAME_W, h=FRAME_H, fps=FPS):
        self.serial = serial
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(str(serial))

        # 1. Enable Infrared stream (Index 1) instead of Color
        self.cfg.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, fps)

        self.profile = self.pipe.start(self.cfg)

        # 2. ENABLE HARDWARE-BASED DISTORTION CORRECTION
        # RealSense handles undistortion internally with specialized hardware
        device = self.profile.get_device()
        
        # Get IR sensor for configuration
        ir_sensor = None
        for sensor in device.query_sensors():
            if sensor.get_info(rs.camera_info.name) == "Stereo Module":
                ir_sensor = sensor
                break

        # 3. DISABLE LASER EMITTER (Crucial for AprilTags)
        depth_sensor = device.first_depth_sensor()

        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0.0)  # 0 = Off

        # 4. MANUAL EXPOSURE & GAIN
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)

        if depth_sensor.supports(rs.option.exposure):
            # 8000us = 8ms.
            depth_sensor.set_option(rs.option.exposure, EXPOSURE)

        if depth_sensor.supports(rs.option.gain):
            depth_sensor.set_option(rs.option.gain, GAIN)

        # 5. Get intrinsics (for reference, but RealSense handles undistortion internally)
        vsp = rs.video_stream_profile(self.profile.get_stream(rs.stream.infrared, 1))
        intr = vsp.get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx],
                           [0, intr.fy, intr.ppy],
                           [0, 0, 1]], np.float32)
        self.D = np.array(intr.coeffs[:5], np.float32)
        print(f"[cam-IR] {serial}: {intr.width}x{intr.height}@{fps} (Emitter OFF, Exp=8ms)")

    def read(self):
        try:
            # SAFETY 1: Add timeout (ms). If shutdown starts, this won't hang forever.
            if self.pipe is None:
                return False, None

            frames = self.pipe.wait_for_frames(timeout_ms=1000)

            ir = frames.get_infrared_frame(1)
            if not ir:
                return False, None

            # SAFETY 2: Copy data immediately
            # np.array creates a copy of the buffer, so we are safe to let 'ir' die naturally.
            data_copy = np.array(ir.get_data())

            # Removed manual 'del' to prevent double-free corruption in fastbins
            return True, data_copy

        except RuntimeError:
            # This happens if we call read() while the pipeline is stopping
            return False, None
        except Exception as e:
            print(f"[Error] RealSense read failed: {e}")
            return False, None

    def release(self):
        """
        Safely stops the pipeline and unlinks references to prevent
        GC-induced double-free errors at shutdown.
        """
        try:
            if self.pipe is not None:
                self.pipe.stop()
        except RuntimeError:
            # Often throws if already stopped
            pass
        except Exception as e:
            print(f"[Error] RealSense release failed: {e}")
        finally:
            # Crucial: Dereference C++ wrappers so destructor isn't called again by GC
            self.pipe = None
            self.profile = None
            self.cfg = None