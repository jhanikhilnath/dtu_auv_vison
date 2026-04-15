import cv2
import numpy as np


class VisionEnhancerCUDA:
    def __init__(self, gamma=1.5, clahe_clip=2.0, clahe_tile=(8, 8)):
        """Initialize the Hybrid CPU/GPU architecture for the Jetson Nano."""

        # 1. CPU: Precompute Gamma LUT
        inv_gamma = 1.0 / gamma
        self.lut = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        # 2. GPU: Precompute CUDA-accelerated CLAHE object
        self.clahe_cuda = cv2.cuda.createCLAHE(
            clipLimit=clahe_clip, tileGridSize=clahe_tile)

        # 3. CPU: Precompute the Static Matrix
        self.color_matrix = np.array([
            [1.2, -0.1, -0.1],  # Blue channel output
            [-0.1, 1.1,  0.0],  # Green channel output
            [-0.2, -0.1, 1.3]   # Red channel output
        ], dtype=np.float32)

        # 4. GPU: Pre-allocate GPU memory to avoid memory leaks during 30fps loops
        self.gpu_frame = cv2.cuda_GpuMat()

    def apply_gray_world_cpu(self, frame):
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

    def apply_red_compensation_cpu(self, frame, green_ratio=0.3):
        """Rebuilds the dead red channel by stealing from the green channel."""
        b, g, r = cv2.split(frame)
        r_new = cv2.addWeighted(r, 1.0, g, green_ratio, 0.0)
        return cv2.merge([b, g, r_new])

    def apply_channel_stretch_cpu(self, frame):
        """Normalizes each color channel independently to span from 0 to 255."""
        b, g, r = cv2.split(frame)
        b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.merge([b, g, r])

    def apply_static_matrix_cpu(self, frame):
        """Applies a hardcoded color shift."""
        return cv2.transform(frame, self.color_matrix)

    def apply_gamma_cpu(self, frame):
        """Applies pre-computed gamma curve instantly via LUT."""
        return cv2.LUT(frame, self.lut)


    def apply_clahe_and_saturation_cuda(self, gpu_mat, saturation_boost=1.3):
        """
        Performs CLAHE and Saturation Boosting simultaneously entirely inside the NVIDIA GPU VRAM
        """
        # 1. Convert to LAB space using CUDA
        gpu_lab = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_BGR2LAB)

        # 2. Split channels in VRAM
        gpu_channels = cv2.cuda.split(gpu_lab)
        gpu_l = gpu_channels[0]
        gpu_a = gpu_channels[1]
        gpu_b = gpu_channels[2]

        # 3. Structure: Apply CUDA CLAHE to the L channel
        gpu_l = self.clahe_cuda.apply(gpu_l)

        # 4. Vibrancy: Boost A and B channels
        # .convertTo() is the CUDA equivalent of cv2.convertScaleAbs
        beta = 128 * (1.0 - saturation_boost)
        gpu_a = gpu_a.convertTo(cv2.CV_8U, alpha=saturation_boost, beta=beta)
        gpu_b = gpu_b.convertTo(cv2.CV_8U, alpha=saturation_boost, beta=beta)

        # 5. Merge and convert back ONCE
        gpu_merged = cv2.cuda.merge([gpu_l, gpu_a, gpu_b])
        gpu_final = cv2.cuda.cvtColor(gpu_merged, cv2.COLOR_LAB2BGR)

        return gpu_final

    # ==========================================
    # MASTER PIPELINE
    # ==========================================

    def process_frame(self, frame):
        """The Pipeline."""

        color_fixed = self.apply_gray_world_cpu(frame)
        # color_fixed = self.apply_static_matrix_cpu(frame)
        # color_fixed = self.apply_red_compensation_cpu(frame, green_ratio=0.35)
        # color_fixed = self.apply_channel_stretch_cpu(frame)

        # 2. Lift Shadows
        brightened = self.apply_gamma_cpu(color_fixed)

        # --- THE BRIDGE ---
        # 3. Upload the CPU array into pre-allocated GPU memory
        self.gpu_frame.upload(brightened)

        # --- GPU PHASE ---
        # 4. Enhance Local Contrast and Saturation on the CUDA cores
        gpu_result = self.apply_clahe_and_saturation_cuda(self.gpu_frame)

        # --- THE RETURN ---
        # 5. Pull the finished image back to standard CPU RAM
        final_frame = gpu_result.download()

        return final_frame
