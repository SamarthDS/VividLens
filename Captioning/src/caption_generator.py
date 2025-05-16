import os

def generate_caption_for_frames(model, frames_folder):
    for fname in sorted(os.listdir(frames_folder)):
        if fname.endswith(".jpg"):
            frame_path = os.path.join(frames_folder, fname)
            try:
                caption, confidence = model.predict(frame_path)
                yield frame_path, caption, confidence
            except Exception as e:
                print(f"⚠️ Error on {frame_path}: {e}")
