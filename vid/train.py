import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

# === CONFIG ===
EPOCHS = 5
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "training.npy"

# === Dataset ===
class DeepEYEDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)  # shape: (228, 228, N)
        frames = data.shape[2]
        frames = frames - frames % 12  # divisible by 12
        data = data[:, :, :frames]     # (228, 228, N)
        data = data.reshape(228, 228, -1, 12)  # (228, 228, num_samples, 12)
        data = np.transpose(data, (2, 3, 0, 1))  # (num_samples, 12, 228, 228)
        data = np.expand_dims(data, axis=1)  # (num_samples, 1, 12, 228, 228)
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # (1, 12, 228, 228)
        return x, x  # input == target

# === Model ===
class Conv3DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv3DAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, 5, 114, 114)
            nn.ReLU(True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1), # (B, 128, 3, 57, 57)
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # (B, 64, 6, 114, 114)
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # (B, 1, 10, 227, 227)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# === Load Data ===
dataset = DeepEYEDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model, Loss, Optimizer ===
model = Conv3DAutoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === Training ===
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for x, y in dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}")

# === Save Model ===
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/saved_model.pth")
print("Model saved to model/saved_model.pth")