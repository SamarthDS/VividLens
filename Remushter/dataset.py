# dataset.py (Flexible Minimal Version)
import os
import cv2
import torch
from torch.utils.data import Dataset


class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, img_size=(240, 320), max_categories=2, max_videos_per_cat=1, frames_per_video=1):
        self.root_dir = root_dir
        self.img_size = img_size
        self.samples = []

        # Safety counters
        loaded_categories = 0
        print("Scanning dataset structure...")

        for category in sorted(os.listdir(root_dir)):
            if loaded_categories >= max_categories:
                break

            cat_path = os.path.join(root_dir, category)
            if not os.path.isdir(cat_path) or category.startswith('.'):
                continue

            print(f"│─ {category}")
            loaded_videos = 0

            for subfolder in sorted(os.listdir(cat_path)):
                if loaded_videos >= max_videos_per_cat:
                    break

                subfolder_path = os.path.join(cat_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                print(f"│  ├─ {subfolder}")
                videos = sorted([v for v in os.listdir(
                    subfolder_path) if v.endswith('.mp4')])

                for video in videos[:1]:  # Only take first video
                    video_path = os.path.join(subfolder_path, video)
                    cap = cv2.VideoCapture(video_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    if frame_count > 0:
                        self.samples.append((video_path, 0))  # Only frame 0
                        loaded_videos += 1
                        print(f"│  │  └─ {video} (frame 0)")

            if loaded_videos > 0:
                loaded_categories += 1

        print(f"Total loaded frames: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_idx = self.samples[idx]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return torch.zeros(3, *self.img_size)

        frame = cv2.resize(frame, self.img_size[::-1])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
