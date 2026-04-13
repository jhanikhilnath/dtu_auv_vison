import cv2
import os
import time

from image_enhancement import VisionEnhancer


def test_pipeline(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    enhancer = VisionEnhancer(gamma=1.5, clahe_clip=2.0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_filename = os.path.join(
        output_dir, "enhanced_test_output_grayworld+clahe+saturationInc.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print(f"Processing Video: {width}x{height} @ {fps} FPS")
    print(f"Saving to: {output_filename}")

    start_time = time.time()
    frames_processed = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if cv2.ocl.useOpenCL():
            frame = cv2.UMat(frame)

        enhanced_frame = enhancer.process_frame(frame)

        if isinstance(enhanced_frame, cv2.UMat):
            enhanced_frame = enhanced_frame.get()

        writer.write(enhanced_frame)
        frames_processed += 1

        if frames_processed % 30 == 0:
            print(f"Processed {frames_processed}/{total_frames} frames...")

    end_time = time.time()
    cap.release()
    writer.release()

    duration = end_time - start_time
    actual_fps = frames_processed / duration
    print(
        f"Processed {frames_processed} frames in {duration:.2f} seconds.")
    print(f"Pipeline Speed: {actual_fps:.2f} FPS")


if __name__ == '__main__':
    INPUT_VIDEO = "AUV-Pool.mp4"
    OUTPUT_FOLDER = "test_results"

    test_pipeline(INPUT_VIDEO, OUTPUT_FOLDER)
