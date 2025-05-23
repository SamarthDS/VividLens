import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import json
import tempfile
import os

class Conv3DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv3DAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Conv3DAutoencoder().to(device)
model.load_state_dict(torch.load("model/saved_model.pth", map_location=device))
model.eval()

def mean_squared_loss(x1, x2):
    diff = x1 - x2
    n_samples = diff.numel()
    sq_diff = diff ** 2
    total = torch.sum(sq_diff)
    distance = torch.sqrt(total)
    mean_distance = distance / n_samples
    return mean_distance.item()

def detect_anomalies(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    frame_count = 0
    im_frames = []
    abnormal_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        image = cv2.resize(frame, (700, 600), interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, (228, 228), interpolation=cv2.INTER_AREA)
        gray = 0.2989 * frame[:,:,0] + 0.5870 * frame[:,:,1] + 0.1140 * frame[:,:,2]
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)
        im_frames.append(gray)
        if len(im_frames) == 12:
            im_frames_np = np.array(im_frames)
            im_frames_np = im_frames_np.reshape(1, 1, 12, 228, 228)
            im_frames_tensor = torch.tensor(im_frames_np, dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(im_frames_tensor)
                loss = mean_squared_loss(im_frames_tensor, output)
            if loss > 0.000038:
                st.error(f'🚨 Abnormal Event Detected at frame {frame_count} 🚨')
                st.image(image, caption=f"Frame {frame_count}", channels="BGR")
                abnormal_frames.append(frame_count)
            im_frames = []
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    with open('abnormal_frames.json', 'w') as json_file:
        json.dump(abnormal_frames, json_file)

st.markdown("<h1 style='text-align: center; color: #006699;'>DeepEYE Anomaly Surveillance 👁️</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    detect_anomalies(tmp_file_path)
    os.unlink(tmp_file_path)
