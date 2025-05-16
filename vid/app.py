import streamlit as st
import cv2
import numpy as np
import tempfile
import time

st.set_page_config(page_title="Theft Detection System", layout="centered")
st.title("🔐 Real-time Theft Detection ")

# Function to detect motion
def motion_detection(video_path):
    cam = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cam.isOpened():
        ret, frame1 = cam.read()
        ret, frame2 = cam.read()
        if not ret:
            break

        diff = cv2.absdiff(frame1, frame2)
        grey = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilate = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 2000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, "Motion Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame1 = cv2.resize(frame1, (640, 480))
        stframe.image(frame1, channels="BGR", use_column_width=True)
        time.sleep(0.01)

    cam.release()

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    motion_detection(tfile.name)
