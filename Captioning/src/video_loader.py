# src/video_loader.py

import cv2
import os

def extract_frames_from_video(video_path, output_folder, frame_rate=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_rate == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved + 1}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"[✓] Saved: {frame_filename}")
            saved += 1

        count += 1

    cap.release()
    print(f"[✓] Total frames saved: {saved}")
