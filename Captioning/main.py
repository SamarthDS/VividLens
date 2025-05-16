from collections import Counter
import os
import re
import streamlit as st
import tempfile
from src.actionclip_model import ActionCLIPWrapper
from src.caption_generator import generate_caption_for_frames
from config.prompts import get_prompts_for_filename
from src.video_loader import extract_frames_from_video

st.markdown('<div style="font-size:2.5rem;font-weight:bold;display:flex;align-items:center;gap:0.7em;margin-bottom:0.5em;">🔍 <span>Surveillance Video Analysis</span> 📷</div>', unsafe_allow_html=True)

uploaded_video = st.file_uploader("Upload a surveillance video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    st.video(uploaded_video)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_video.read())
        video_path = temp_video.name

    frames_folder = "frames"
    output_txt = "outputs/captions.txt"
    summary_txt = "outputs/summary.txt"

    # Clean up frames folder before extracting new frames
    if os.path.exists(frames_folder):
        for f in os.listdir(frames_folder):
            os.remove(os.path.join(frames_folder, f))
    else:
        os.makedirs(frames_folder)

    # Clean up output files
    for out_file in [output_txt, summary_txt]:
        if os.path.exists(out_file):
            with open(out_file, "w") as f:
                f.truncate(0)

    video_filename = os.path.basename(video_path)

    # 🔎 Use helper to get prompts based on filename
    selected_prompts = get_prompts_for_filename(video_filename)

    st.info("📽️ Extracting frames from video...")
    extract_frames_from_video(video_path, frames_folder, frame_rate=10)

    st.info("🎯 Loading ActionCLIP model...")
    model = ActionCLIPWrapper(prompts=selected_prompts)

    st.info("🧠 Generating captions...")
    all_captions = []
    frame_to_caption = []

    for frame_path, caption, confidence in generate_caption_for_frames(model, frames_folder):
        all_captions.append(caption.lower())
        frame_to_caption.append((os.path.basename(frame_path), caption, confidence))

    st.success("✅ Captions generated!")

    # 📝 Display frame-by-frame captions
    st.subheader("📸 Frame-by-Frame Captions")
    captions_text = ""
    for frame, caption, conf in frame_to_caption:
        cap_line = f"[{frame}]: {caption} ({conf:.2f})"
        captions_text += cap_line + "\n"
        st.text(cap_line)

    # 📌 Generate summary
    keywords = [
        "fighting", "shooting", "robbery", "assault", "vandalism", "stealing",
        "shoplifting", "arson", "abuse", "arrest", "explosion", "burglary", "road accident", "crash"
    ]

    summary_counter = Counter()
    for cap in all_captions:
        for kw in keywords:
            if kw in cap:
                summary_counter[kw] += 1

    if summary_counter:
        most_common_event, count = summary_counter.most_common(1)[0]
        summary = f"The video primarily depicts {most_common_event} activity."
    else:
        summary = "The video shows general suspicious or anomalous activity."

    st.subheader("📝 Summary")
    st.write(summary)

    # 🔁 Map summary to time range (assume 10 FPS, 1 frame per 10th second)
    time_spans = {}
    for idx, (frame_name, cap, _) in enumerate(frame_to_caption):
        for kw in keywords:
            if kw in cap.lower():
                seconds = idx / 10
                mm_ss = f"{int(seconds//60):02d}:{int(seconds%60):02d}"
                if kw not in time_spans:
                    time_spans[kw] = []
                time_spans[kw].append(mm_ss)

    if time_spans:
        for kw, times in time_spans.items():
            st.markdown(f"- **{kw.title()} observed at:** {', '.join(times)}")

    # 📋 Copy captions and summary
    st.subheader("📤 Export")
    st.download_button("📄 Download Analysis", captions_text, file_name="captions.txt")
    st.download_button("📝 Download Summary", summary, file_name="summary.txt")