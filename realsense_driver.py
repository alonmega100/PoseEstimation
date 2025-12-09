import numpy as np
import pyrealsense2 as rs

from config import FRAME_W, FRAME_H, FPS


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
        # Format is Y8 (8-bit grayscale)
        self.cfg.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, fps)

        self.profile = self.pipe.start(self.cfg)

        # 2. DISABLE LASER EMITTER (Crucial for AprilTags)
        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()

        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0.0)  # 0 = Off

        # 3. MANUAL EXPOSURE (Reduces motion blur)
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)

        if depth_sensor.supports(rs.option.exposure):
            # 4000us = 4ms. Lower = sharper motion, darker image.
            depth_sensor.set_option(rs.option.exposure, 4000.0)

        if depth_sensor.supports(rs.option.gain):
            # Compensate for low exposure with gain
            depth_sensor.set_option(rs.option.gain, 60.0)

        # 4. Get intrinsics
        vsp = rs.video_stream_profile(self.profile.get_stream(rs.stream.infrared, 1))
        intr = vsp.get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx],
                           [0, intr.fy, intr.ppy],
                           [0, 0, 1]], np.float32)
        self.D = np.array(intr.coeffs[:5], np.float32)
        print(f"[cam-IR] {serial}: {intr.width}x{intr.height}@{fps} (Emitter OFF, Exp=4ms)")

    def read(self):
        frames = self.pipe.wait_for_frames()
        ir = frames.get_infrared_frame(1)
        if not ir:
            return False, None

        # IMPORTANT: Use np.array() to COPY data, avoiding memory crashes
        return True, np.array(ir.get_data())

    def release(self):
        try:
            self.pipe.stop()
        except:
            pass

