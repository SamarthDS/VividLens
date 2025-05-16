import cv2
import numpy as np


def read_video_frames(video_path, max_frames=30):
    """
    Read frames from a video file

    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to read

    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def augment_frames(frames):
    """
    Apply various augmentations to frames

    Args:
        frames: List of frames as numpy arrays

    Returns:
        List of tuples (augmentation_name, augmented_frame)
    """
    augmented = []

    for frame in frames:
        # Original
        augmented.append(("Original", frame.copy()))

        # Grayscale conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert back to BGR for display
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        augmented.append(("Grayscale", gray))

        # Brightness adjustment
        brightness = np.clip(frame * 1.5, 0, 255).astype(np.uint8)
        augmented.append(("Brightness +50%", brightness))

        # Color enhancement
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0,
                               255).astype(np.uint8)  # Increase saturation
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        augmented.append(("Color Enhanced", enhanced))

    return augmented
