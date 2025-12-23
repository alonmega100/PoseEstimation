"""
Modular vision display manager for handling multiple camera feeds.
Encapsulates common display logic for mosaic creation, serial number overlays, and window management.
"""

import cv2
import numpy as np
from typing import Dict, Optional, List


class VisionDisplay:
    """Manages display of multiple camera feeds with customizable overlays."""
    
    def __init__(self, camera_serials: List[str], frame_w: int, frame_h: int, window_title: str = "Vision Feeds"):
        """
        Initialize vision display manager.
        
        Args:
            camera_serials: List of camera serial numbers in display order
            frame_w: Frame width in pixels
            frame_h: Frame height in pixels
            window_title: Title for the display window
        """
        self.camera_serials = camera_serials
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.window_title = window_title
        
        # Storage for current frames
        self.frames: Dict[str, np.ndarray] = {sn: np.zeros((frame_h, frame_w, 3), dtype=np.uint8) 
                                               for sn in camera_serials}
        self.overlay_texts: Dict[str, List[str]] = {sn: [] for sn in camera_serials}
        
        # Setup window
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        
    def update_frame(self, serial: str, vis_img: np.ndarray, overlay_text: Optional[List[str]] = None):
        """
        Update frame for a specific camera and optionally set overlay text.
        
        Args:
            serial: Camera serial number
            vis_img: Visualization image (numpy array)
            overlay_text: Optional list of text lines to overlay on the frame
        """
        if serial not in self.frames:
            raise ValueError(f"Unknown camera serial: {serial}")
        
        self.frames[serial] = vis_img.copy()
        self.overlay_texts[serial] = overlay_text if overlay_text else []
    
    def show_mosaic(self, overlay_global_text: Optional[List[str]] = None) -> bool:
        """
        Display all cameras in a horizontal mosaic with serial numbers and overlays.
        
        Args:
            overlay_global_text: Optional list of text lines to overlay on the entire mosaic
            
        Returns:
            True if display loop should continue, False if user pressed ESC/q
        """
        # Prepare frames with serial numbers
        vis_list = []
        for serial in self.camera_serials:
            frame = self.frames[serial].copy()
            
            # Add serial number at lower left corner
            self._add_serial_number(frame, serial)
            
            # Add per-camera overlay text if any
            if self.overlay_texts[serial]:
                self._add_text_overlay(frame, self.overlay_texts[serial], position=(10, 40))
            
            vis_list.append(frame)
        
        # Pad to same height and concatenate
        if vis_list:
            max_h = max(v.shape[0] for v in vis_list)
            vis_list_padded = [
                cv2.copyMakeBorder(v, 0, max_h - v.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                for v in vis_list
            ]
            mosaic = cv2.hconcat(vis_list_padded)
            
            # Add global overlay text if provided
            if overlay_global_text:
                y = 100
                for line in overlay_global_text:
                    cv2.putText(mosaic, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                    y += 24
            
            cv2.imshow(self.window_title, mosaic)
        
        # Handle key press
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):  # ESC or 'q'
            return False
        
        return True
    
    def _add_serial_number(self, frame: np.ndarray, serial: str):
        """Add serial number at lower left corner of frame."""
        text = f"SN: {serial}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (255, 255, 255)  # White
        
        # Get text size for background rectangle
        # text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x, text_y = 10, frame.shape[0] - 10  # Lower left corner
        #
        # Draw dark background rectangle behind text
        # cv2.rectangle(
        #     frame,
        #     (text_x - 3, text_y - text_size[1] - 3),
        #     (text_x + text_size[0] + 3, text_y + 3),
        #     (0, 0, 0),  # Black background
        #     -1  # Filled
        # )
        
        # Draw the serial number text
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
    
    def _add_text_overlay(self, frame: np.ndarray, texts: List[str], position: tuple = (10, 40)):
        """Add text lines to frame starting at position."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        color = (0, 255, 255)  # Yellow
        
        x, y = position
        line_height = 22
        
        for text in texts:
            cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
            y += line_height
    
    def cleanup(self):
        """Close windows and clean up resources."""
        cv2.destroyWindow(self.window_title)
        cv2.destroyAllWindows()
