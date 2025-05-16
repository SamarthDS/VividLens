import streamlit as st
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
# Make sure this matches your model class
from model import TransformerColorizer

# Set up the app
st.set_page_config(layout="wide")
st.title("🎨 Video Frame Augmentation Visualizer")

# Load your trained model


@st.cache_resource
def load_model():
    # Match your training config
    model = TransformerColorizer(upscale_factor=1)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model


model = load_model()

# Create directories
data_dir = os.path.join("data", "sample_videos")
os.makedirs(data_dir, exist_ok=True)


def read_video_frames(video_path, max_frames=1):
    """Read first frame from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    ret, frame = cap.read()
    if ret:
        frames.append(frame)
    cap.release()
    return frames


def colorize_frame(frame, model):
    """Colorize frame using your model"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),  # Match your training size
        transforms.ToTensor()
    ])

    # Convert to grayscale and process
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    img_tensor = transform(gray_rgb).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        output = output.squeeze(0).permute(1, 2, 0).numpy()
        output = (output * 255).astype(np.uint8)
        output = cv2.resize(output, (frame.shape[1], frame.shape[0]))

    return output


def adjust_brightness(frame, percent=50):
    """Adjust brightness by percentage"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + percent/100), 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def enhance_color(frame, percent=30):
    """Enhance color saturation"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + percent/100), 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def augment_frames(frames):
    """Generate augmented versions matching your screenshot"""
    augmented = []
    for frame in frames:
        # Original
        augmented.append(("Original", frame))

        # Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        augmented.append(("Grayscale", gray_rgb))

        # Brightness +50%
        bright = adjust_brightness(frame, 50)
        augmented.append(("Brightness +50%", bright))

        # Color Enhanced
        color_enhanced = enhance_color(frame, 30)
        augmented.append(("Color Enhanced", color_enhanced))

        # Colorized (using your model)
        try:
            colorized = colorize_frame(frame, model)
            augmented.append(("Colorized", colorized))
        except Exception as e:
            # Optionally log the error for debugging
            # print(f"Colorization failed: {str(e)}")
            pass  # Do nothing, just skip colorized frame if it fails

    return augmented


# Main app interface
uploaded_file = st.file_uploader(
    "Upload a short video clip",
    type=["mp4", "avi", "mov"]
)

if uploaded_file:
    # Save temporarily
    video_path = os.path.join(data_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ Video uploaded successfully!")
    st.video(video_path)

    # Process first frame only (matches your screenshot)
    frames = read_video_frames(video_path)
    augmented = augment_frames(frames[:1])

    # Display results in 4 columns matching your screenshot
    st.subheader("Augmented Frames (Sample from 1st Frame)")
    # 4 columns for Original, Grayscale, Brightness, Color Enhanced
    cols = st.columns(4)

    # Display first 4 augmentations (matches your screenshot)
    for i, (title, frame) in enumerate(augmented[:4]):
        with cols[i]:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.image(img, caption=title, use_column_width=True)

    # # Show colorized version separately if available
    # if len(augmented) > 4:
    #     st.subheader("AI Colorized Version")
    #     st.image(
    #         Image.fromarray(cv2.cvtColor(augmented[4][1], cv2.COLOR_BGR2RGB)),
    #         caption=augmented[4][0],
    #         use_column_width=True
    #     )

    # Clean up
    os.remove(video_path)
