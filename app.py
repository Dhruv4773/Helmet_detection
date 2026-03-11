import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

model = YOLO(r"your_file_path")
st.title("Helmet and Number Plate Detection App (YOLOv8)")
st.markdown("Upload an **Image or Video** below for Detection:")

option = st.radio("Select Input Type:", ("Image", "Video"))

if option == "Image":
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        if st.button("Detect Objects in Image"):
            # Read image as bytes & decode with OpenCV
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # YOLOv8 Detection
            results = model.predict(source=image, save=False)

            # Draw Results
            for r in results:
                annotated_image = r.plot()

            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image, channels="RGB", caption="Detection Result")

elif option == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        if st.button("Detect Objects in Video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_video.read())
                tmp_path = tmp_file.name

            cap = cv2.VideoCapture(tmp_path)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLOv8 Detection on Frame
                results = model.predict(source=frame, save=False, conf=0.5)

                for r in results:
                    annotated_frame = r.plot()

                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame, channels="RGB")

            cap.release()

            os.remove(tmp_path)
            st.success("Video Processing Completed")
