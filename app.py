import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os

st.set_page_config(page_title="ADAS YOLO Detection", layout="wide")

st.title("🚗 ADAS Object Detection using YOLO")

st.write("""
Upload an image or video to detect:
- Cars
- Pedestrians
- Trucks
- Buses
- Motorcycles
""")

# Load YOLO model
model = YOLO("yolo11n.pt")

# Upload file
uploaded_file = st.file_uploader(
    "Upload image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi"]
)

# Confidence slider
conf = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.35,
    step=0.05
)

if uploaded_file is not None:

    suffix = os.path.splitext(uploaded_file.name)[1]

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    # =========================
    # IMAGE DETECTION
    # =========================
    if suffix.lower() in [".jpg", ".jpeg", ".png"]:

        st.info("Processing image...")

        results = model.predict(
            source=input_path,
            conf=conf
        )

        annotated_image = results[0].plot()

        st.image(
            annotated_image,
            caption="Detection Result",
            channels="BGR",
            use_container_width=True
        )

        st.success("Image processing completed!")

    # =========================
    # VIDEO DETECTION
    # =========================
    else:

        st.info("Processing video... Please wait ⏳")

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            st.error("Error: Could not open video.")

        else:

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_path = "processed_video.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            progress_bar = st.progress(0)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                results = model.predict(frame, conf=conf)

                annotated_frame = results[0].plot()

                out.write(annotated_frame)

                frame_count += 1

                if total_frames > 0:
                    progress_bar.progress(
                        min(frame_count / total_frames, 1.0)
                    )

            cap.release()
            out.release()

            st.success("Video processing completed!")

            with open(output_path, "rb") as file:

                st.download_button(
                    label="📥 Download Processed Video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
