import os
import cv2
import numpy as np

image_data = []
# replace the path with your training data path
train_path = "/Users/snehapratap/Desktop/Avenue Dataset/training_videos"
fps = 5
video_exts = ('.mp4', '.avi', '.mov', '.mkv')
train_videos = [f for f in os.listdir(train_path) if f.lower().endswith(video_exts)]
train_images_path = os.path.join(train_path, 'frames')

# Create frames directory if it doesn't exist
if not os.path.exists(train_images_path):
    os.makedirs(train_images_path)

def data_store(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_AREA)
    # Pad to 228x228
    image = np.pad(image, ((0, 1), (0, 1), (0, 0)), mode='constant')
    gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    image_data.append(gray)

for video in train_videos:
    video_path = os.path.join(train_path, video)
    cap = cv2.VideoCapture(video_path)
    count = 0
    success, frame = cap.read()
    while success:
        frame_path = os.path.join(train_images_path, f"{count:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        # Skip frames to control fps
        for _ in range(fps):
            success, frame = cap.read()
            if not success:
                break
        count += 1
    cap.release()

    images = os.listdir(train_images_path)
    for image in images:
        image_path = os.path.join(train_images_path, image)
        data_store(image_path)

image_data = np.array(image_data)
# Rearrange to (228, 228, N)
image_data = np.transpose(image_data, (1, 2, 0))
# Pad temporal dimension to nearest multiple of 12
frames = image_data.shape[2]
pad_frames = (12 - frames % 12) % 12
if pad_frames > 0:
    image_data = np.pad(image_data, ((0, 0), (0, 0), (0, pad_frames)), mode='constant')
image_data = (image_data - image_data.mean()) / (image_data.std())
image_data = np.clip(image_data, 0, 1)

# Storing the data 
np.save('training.npy', image_data)
