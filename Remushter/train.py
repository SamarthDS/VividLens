# import os
# import cv2
# import torch
# import random
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from model import TransformerColorizer
# from tqdm import tqdm
# import torch.nn.functional as F

# # ------------ Config ------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 4
# EPOCHS = 10
# LR = 1e-4
# VIDEO_DIR = "data/sample_videos"
# IMG_SIZE = 128
# UPSCALE = 2
# # --------------------------------


# class VideoFrameDataset(Dataset):
#     def __init__(self, video_dir, img_size=128):
#         self.frames = []
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((img_size, img_size)),
#             transforms.ToTensor()
#         ])

#         for file in os.listdir(video_dir):
#             if not file.lower().endswith((".mp4", ".avi", ".mov")):
#                 continue
#             path = os.path.join(video_dir, file)
#             cap = cv2.VideoCapture(path)
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 self.frames.append(frame)
#             cap.release()

#     def __len__(self):
#         return len(self.frames)

#     def __getitem__(self, idx):
#         img = self.frames[idx]
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_tensor = self.transform(img_rgb)
#         img_gray = transforms.Grayscale()(img_tensor)

#         # Normalize target to [0, 1]
#         return img_gray, img_tensor


# def train():
#     dataset = VideoFrameDataset(VIDEO_DIR, IMG_SIZE)
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#     model = TransformerColorizer(upscale_factor=UPSCALE).to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#     for epoch in range(EPOCHS):
#         model.train()
#         epoch_loss = 0

#         for gray, color in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
#             gray, color = gray.to(DEVICE), color.to(DEVICE)

#             output = model(gray)
#             color = F.interpolate(
#                 color, size=output.shape[2:], mode='bilinear')
#             loss = F.mse_loss(output, color)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()

#         print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")

#     torch.save(model.state_dict(), "colorizer_upscaler.pt")
#     print("✅ Model saved as 'colorizer_upscaler.pt'")


# if __name__ == "__main__":
#     train()


# train.py (Configurable Version)

# ----------------------------------------------------------------------------------------

# from dataset import VideoFrameDataset
# from torch.utils.data import DataLoader


# def train():
#     # CONFIGURATION (change these as needed)
#     config = {
#         "data_dir": "data",
#         "img_size": (240, 320),
#         "max_categories": 15,    # Max parent folders (e.g. Abuse, Arrest)
#         "max_videos_per_cat": 5,  # Max subfolders per category (e.g. Abuse001)
#         "batch_size": 10
#     }

#     dataset = VideoFrameDataset(
#         root_dir=config["data_dir"],
#         img_size=config["img_size"],
#         max_categories=config["max_categories"],
#         max_videos_per_cat=config["max_videos_per_cat"]
#     )

#     loader = DataLoader(dataset,
#                         batch_size=config["batch_size"],
#                         num_workers=0,
#                         shuffle=True)

#     for i, batch in enumerate(loader):
#         print(f"Batch {i}: shape={batch.shape}")
#         if i >= 3:  # Early stop for testing
#             break


# if __name__ == "__main__":
#     print("=== Starting Flexible Loader ===")
#     train()
#     print("=== Test Complete ===")


# -----------------------------------------------------------------------------------------


# train.py (Final Version with Training + Model Saving)

# train.py (Fixed Version)
# import os
# import cv2
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from tqdm import tqdm
# from model import TransformerColorizer

# # Configuration
# CONFIG = {
#     "data_dir": "data",
#     "img_size": 128,            # Input size (will be upscaled by model)
#     "upscale_factor": 2,        # Should match your model's upscale capability
#     "max_categories": 2,        # Safety limit
#     "max_videos_per_cat": 1,    # Safety limit
#     "frames_per_video": 10,     # Max frames per video
#     "batch_size": 4,
#     "epochs": 10,
#     "learning_rate": 1e-4,
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "model_save_path": "model.pth"
# }


# class VideoFrameDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, img_size=128, max_frames=10):
#         self.root_dir = root_dir
#         self.img_size = img_size
#         self.max_frames = max_frames
#         self.samples = []
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((img_size, img_size)),
#             transforms.ToTensor()
#         ])

#         print("Scanning dataset...")
#         for category in sorted(os.listdir(root_dir)):
#             cat_path = os.path.join(root_dir, category)
#             if not os.path.isdir(cat_path) or category.startswith('.'):
#                 continue

#             for subfolder in sorted(os.listdir(cat_path)):
#                 subfolder_path = os.path.join(cat_path, subfolder)
#                 if not os.path.isdir(subfolder_path):
#                     continue

#                 for video in sorted([v for v in os.listdir(subfolder_path) if v.lower().endswith(('.mp4', '.avi', '.mov'))]):
#                     video_path = os.path.join(subfolder_path, video)
#                     cap = cv2.VideoCapture(video_path)
#                     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#                     # Sample frames evenly throughout the video
#                     frame_indices = set()
#                     if frame_count > 0:
#                         step = max(1, frame_count // self.max_frames)
#                         frame_indices = set(range(0, frame_count, step))

#                     cap.release()

#                     for idx in frame_indices:
#                         self.samples.append((video_path, idx))
#                     print(
#                         f"Loaded {len(frame_indices)} frames from {video_path}")

#         print(f"Total frames: {len(self.samples)}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         video_path, frame_idx = self.samples[idx]

#         cap = cv2.VideoCapture(video_path)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#         ret, frame = cap.read()
#         cap.release()

#         if not ret:
#             # Return blank frames if read fails
#             blank = torch.zeros(3, self.img_size, self.img_size)
#             return blank[:1], blank  # Gray, Color

#         # Process frame
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img_tensor = self.transform(frame_rgb)
#         img_gray = transforms.Grayscale()(img_tensor)

#         return img_gray, img_tensor


# def train():
#     # Initialize
#     dataset = VideoFrameDataset(
#         root_dir=CONFIG["data_dir"],
#         img_size=CONFIG["img_size"],
#         max_frames=CONFIG["frames_per_video"]
#     )
#     dataloader = DataLoader(
#         dataset,
#         batch_size=CONFIG["batch_size"],
#         shuffle=True,
#         num_workers=0  # Safer for Windows
#     )

#     model = TransformerColorizer(upscale_factor=1).to(
#         CONFIG["device"])  # Changed to upscale_factor=1
#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=CONFIG["learning_rate"])

#     # Training loop
#     for epoch in range(CONFIG["epochs"]):
#         model.train()
#         epoch_loss = 0.0

#         pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
#         for gray, color in pbar:
#             gray, color = gray.to(CONFIG["device"]), color.to(CONFIG["device"])

#             # Forward pass
#             output = model(gray)

#             # Ensure output matches target size
#             if output.shape[-2:] != color.shape[-2:]:
#                 output = F.interpolate(
#                     output, size=color.shape[-2:], mode='bilinear')

#             # Loss calculation
#             loss = F.mse_loss(output, color)

#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()
#             pbar.set_postfix({"loss": f"{loss.item():.4f}"})

#         print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(dataloader):.4f}")

#     # Save model
#     torch.save(model.state_dict(), CONFIG["model_save_path"])
#     print(f"✅ Model saved to {CONFIG['model_save_path']}")


# if __name__ == "__main__":
#     print(f"Using device: {CONFIG['device'].upper()}")
#     train()


import os
import cv2
import torch
import pickle
import hashlib
import psutil
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import TransformerColorizer

# Memory-aware configuration
CONFIG = {
    "data_dir": "data",
    "img_size": 128,  # Reduced from original
    "upscale_factor": 1,
    "max_categories": 2,
    "max_videos_per_cat": 1,
    "frames_per_video": 5,  # Reduced from 10
    "batch_size": 4,  # Reduced from 16
    "epochs": 10,
    "learning_rate": 1e-4,
    "model_save_path": "model.pth",
    "cache_dir": "frame_cache",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,  # Reduced from 4
    "pin_memory": False,  # Disabled to save memory
    "compile_model": False,
    "max_cache_size_mb": 2000,  # 2GB cache limit
    "memory_safety_margin": 1024  # 1GB safety margin
}


class MemoryAwareCacheDataset(Dataset):
    def __init__(self, root_dir, img_size=128, max_frames=5, cache_dir="frame_cache"):
        self.root_dir = root_dir
        self.img_size = img_size
        self.max_frames = max_frames
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Memory-friendly transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        # Generate cache ID
        cache_id = hashlib.md5(
            f"{root_dir}_{img_size}_{max_frames}".encode()).hexdigest()
        self.cache_file = self.cache_dir / f"{cache_id}.pkl"

        # Memory check before loading/building
        self._check_memory()
        self.samples = self._load_or_build_cache()

    def _check_memory(self):
        available_mem = psutil.virtual_memory().available / (1024 ** 2)  # MB
        safe_limit = available_mem - CONFIG["memory_safety_margin"]

        if safe_limit < CONFIG["max_cache_size_mb"]:
            # Keep at least 500MB cache
            adjusted_size = max(500, safe_limit * 0.8)
            print(
                f"⚠️  Insufficient memory. Reducing cache size to {adjusted_size:.0f}MB")
            CONFIG["max_cache_size_mb"] = adjusted_size

    def _load_or_build_cache(self):
        if self.cache_file.exists():
            print("🔍 Loading frames from cache...")
            try:
                with open(self.cache_file, 'rb') as f:
                    return self._load_cache_in_chunks(f)
            except Exception as e:
                print(f"⚠️  Cache load failed: {str(e)}. Rebuilding cache...")
                return self._build_cache()
        return self._build_cache()

    def _load_cache_in_chunks(self, file):
        """Load cache in chunks to prevent memory overload"""
        samples = []
        while True:
            try:
                chunk = pickle.load(file)
                samples.extend(chunk)
                # Check memory during loading
                # Approx 0.5KB per sample
                if len(samples) * 0.0005 > CONFIG["max_cache_size_mb"]:
                    samples = samples[:len(samples)//2]
                    print(
                        f"⚠️  Memory limit reached. Loaded {len(samples)} samples")
                    break
            except EOFError:
                break
        return samples

    def _build_cache(self):
        """Build cache with memory monitoring"""
        samples = []
        processed_count = 0
        total_videos = sum(1 for _ in self._video_generator())

        for video_path in tqdm(self._video_generator(), total=total_videos, desc="Processing videos"):
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_count > 0:
                step = max(1, frame_count // self.max_frames)
                for idx in range(0, frame_count, step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img_tensor = self.transform(frame_rgb)
                            img_gray = transforms.Grayscale()(img_tensor)
                            samples.append((img_gray, img_tensor))
                            processed_count += 1

                            # Memory check
                            if processed_count % 100 == 0:
                                current_mem = psutil.virtual_memory().used / (1024 ** 2)
                                if current_mem > CONFIG["max_cache_size_mb"]:
                                    print(
                                        f"⚠️  Memory limit reached at {processed_count} samples")
                                    cap.release()
                                    return samples
                        except Exception as e:
                            print(f"⚠️  Error processing frame: {str(e)}")
                            continue
            cap.release()

        # Save in chunks
        self._save_cache_in_chunks(samples)
        return samples

    def _video_generator(self):
        """Generator for video paths to minimize memory usage"""
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    yield os.path.join(root, file)

    def _save_cache_in_chunks(self, samples, chunk_size=1000):
        """Save cache in chunks to prevent memory issues"""
        with open(self.cache_file, 'wb') as f:
            for i in range(0, len(samples), chunk_size):
                pickle.dump(samples[i:i+chunk_size], f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def setup_training():
    # Initialize dataset with memory limits
    dataset = MemoryAwareCacheDataset(
        root_dir=CONFIG["data_dir"],
        img_size=CONFIG["img_size"],
        max_frames=CONFIG["frames_per_video"],
        cache_dir=CONFIG["cache_dir"]
    )

    # Memory-friendly DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        persistent_workers=False  # Reduces memory usage
    )

    # Lightweight model setup
    model = TransformerColorizer(upscale_factor=CONFIG["upscale_factor"])
    model = model.to(CONFIG["device"])

    # Disable compilation to save memory
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["learning_rate"])
    scaler = None  # Disable mixed precision to save memory

    return model, dataloader, optimizer, scaler


def train():
    model, dataloader, optimizer, _ = setup_training()
    dataset = dataloader.dataset

    print(f"\n🚀 Starting training (Memory Optimized)")
    print(f"💻 Device: {CONFIG['device'].upper()}")
    if CONFIG["device"] == "cuda":
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 Training samples: {len(dataset)}")
    print(f"🔄 Batch size: {CONFIG['batch_size']}")
    print(f"💾 Memory limit: {CONFIG['max_cache_size_mb']}MB")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for gray, color in pbar:
            # Manual memory management
            torch.cuda.empty_cache() if CONFIG["device"] == "cuda" else None

            gray, color = gray.to(CONFIG["device"]), color.to(CONFIG["device"])

            # Forward pass
            output = model(gray)
            if output.shape[-2:] != color.shape[-2:]:
                output = F.interpolate(
                    output, size=color.shape[-2:], mode='bilinear')
            loss = F.mse_loss(output, color)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"📉 Epoch {epoch+1} Avg Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), CONFIG["model_save_path"])
    print(f"\n💾 Model saved to {CONFIG['model_save_path']}")


if __name__ == "__main__":
    # Memory check before starting
    available_mem = psutil.virtual_memory().available / (1024 ** 2)
    print(f"Available memory: {available_mem:.0f}MB")

    if available_mem < 4000:  # Less than 4GB available
        CONFIG["batch_size"] = max(2, CONFIG["batch_size"] // 2)
        CONFIG["frames_per_video"] = max(2, CONFIG["frames_per_video"] // 2)
        print(
            f"⚠️  Low memory detected. Reducing batch size to {CONFIG['batch_size']}")

    train()
