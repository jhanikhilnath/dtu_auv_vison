import cv2
import numpy as np


class VisionEnhancer:
    def __init__(self, gamma=1.5, clahe_clip=2.0, clahe_tile=(8, 8)):
        """Initialize all static math objects and matrices once to save CPU."""

        # 1. Precompute Gamma LUT
        inv_gamma = 1.0 / gamma
        self.lut = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        # 2. Precompute CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip, tileGridSize=clahe_tile)

    def apply_gray_world(self, frame):
        """Forces the average color of the image to be neutral gray."""
        mean_val = cv2.mean(frame)
        avg_b, avg_g, avg_r = mean_val[0], mean_val[1], mean_val[2]
        avg_gray = (avg_b + avg_g + avg_r) / 3.0

        scale_b = avg_gray / avg_b if avg_b > 0 else 1.0
        scale_g = avg_gray / avg_g if avg_g > 0 else 1.0
        scale_r = avg_gray / avg_r if avg_r > 0 else 1.0

        b, g, r = cv2.split(frame)

        b = cv2.convertScaleAbs(b, alpha=scale_b)
        g = cv2.convertScaleAbs(g, alpha=scale_g)
        r = cv2.convertScaleAbs(r, alpha=scale_r)
        return cv2.merge([b, g, r])

    def apply_gamma(self, frame):
        """applying pre-computed gamma curve."""
        return cv2.LUT(frame, self.lut)

    def apply_clahe(self, frame):
        """Apply CLAHE."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        merged = cv2.merge([l, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def process_frame(self, frame):
        """Apply underwater enhancement"""

        # Correct Color Cast
        color_fixed = self.apply_gray_world(frame)

        # Lift Shadows
        brightened = self.apply_gamma(color_fixed)

        # Enhance Local Contrast
        final_frame = self.apply_clahe(brightened)

        return final_frame
