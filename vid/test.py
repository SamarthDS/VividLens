import cv2
import numpy as np
import torch
import torch.nn as nn

# === Model Definition (must match train.py) ===
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

def mean_squared_loss(x1, x2):
    diff = x1 - x2
    n_samples = diff.numel()
    sq_diff = diff ** 2
    total = torch.sum(sq_diff)
    distance = torch.sqrt(total)
    mean_distance = distance / n_samples
    return mean_distance.item()

# === Load PyTorch model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Conv3DAutoencoder().to(device)
model.load_state_dict(torch.load("model/saved_model.pth", map_location=device))
model.eval()

# Replace the test video data path here
cap = cv2.VideoCapture("/Users/snehapratap/Desktop/Avenue Dataset/testing_videos/04.avi")
print(cap.isOpened())

while cap.isOpened():
    im_frames = []
    ret, frame = cap.read()
    if not ret:
        break
    for i in range(12):
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.resize(frame, (700, 600), interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, (228, 228), interpolation=cv2.INTER_AREA)
        gray = 0.2989 * frame[:,:,0] + 0.5870 * frame[:,:,1] + 0.1140 * frame[:,:,2]
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)
        im_frames.append(gray)
    if len(im_frames) < 12:
        break
    im_frames_np = np.array(im_frames)
    im_frames_np = im_frames_np.reshape(1, 1, 12, 228, 228)
    im_frames_tensor = torch.tensor(im_frames_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(im_frames_tensor)
        loss = mean_squared_loss(im_frames_tensor, output)
    print("Mean Squared Loss:", loss)
    if frame is None:
        print("Frame is None")
    if 0.00062 < loss < 0.00067:
        print('Abnormal Event Detected')
        cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), 2)
        text = "Abnormal Event"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(image, (50, 50 - text_height), (50 + text_width, 50), (255, 255, 255), -1)
        cv2.putText(image, text, (45, 46), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    resized_frame = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("DeepEYE Anomaly Surveillance", resized_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()