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

        self.color_matrix = np.array([
            [1.2, -0.1, -0.1],  # Blue channel output
            [-0.1, 1.1,  0.0],  # Green channel output
            [-0.2, -0.1, 1.3]   # Red channel output
        ], dtype=np.float32)

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

    def apply_red_compensation(self, frame, green_ratio=0.3):
        """
        Dynamic: Rebuilds the dead red channel by stealing structural detail 
        from the green channel. Extremely fast and avoids noise amplification.
        """
        b, g, r = cv2.split(frame)

        # r_new = (1.0 * r) + (green_ratio * g) + 0.0
        # cv2.addWeighted is highly optimized for C++/OpenCL
        r_new = cv2.addWeighted(r, 1.0, g, green_ratio, 0.0)

        return cv2.merge([b, g, r_new])

    def apply_channel_stretch(self, frame):
        """Normalizes each color channel independently to span from 0 to 255."""
        b, g, r = cv2.split(frame)

        b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)

        return cv2.merge([b, g, r])

    def apply_static_matrix(self, frame):
        """Applies a hardcoded color shift"""

        return cv2.transform(frame, self.color_matrix)

    def apply_clahe_and_saturation(self, frame, saturation_boost=1.3):
        """Performs CLAHE and Saturation Boosting simultaneously"""
        # 1. Convert to LAB space ONCE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 2. Structure: Apply CLAHE to the Lightness channel
        l = self.clahe.apply(l)

        # 3. Vibrancy: Boost the A and B color channels
        beta = 128 * (1.0 - saturation_boost)

        a = cv2.convertScaleAbs(a, alpha=saturation_boost, beta=beta)
        b = cv2.convertScaleAbs(b, alpha=saturation_boost, beta=beta)

        # 4. Merge and convert back ONCE
        merged = cv2.merge([l, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

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
        # color_fixed = self.apply_static_matrix(frame)
        # color_fixed = self.apply_red_compensation(frame, green_ratio=0.35)
        # color_fixed = self.apply_channel_stretch(frame)

        # Lift Shadows
        brightened = self.apply_gamma(color_fixed)

        # Enhance Local Contrast
        final_frame = self.apply_clahe_and_saturation(brightened)

        return final_frame
