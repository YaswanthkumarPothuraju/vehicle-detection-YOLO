"""
Vehicle Detection App — YOLO11n + Streamlit
===========================================
Run:
    pip install -r requirements.txt
    streamlit run app.py

Expected model for Streamlit Community Cloud:
    best_vehicle_yolo11n.pt in the same GitHub repository folder as app.py.

You can also set the MODEL_PATH environment variable, but the default is
relative to this app's location (not a Kaggle path).
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import opencv as cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


APP_TITLE = "Vehicle Detection — YOLO11n"
APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CANDIDATES = [
    APP_DIR / "best_vehicle_yolo11n.pt",
]
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
SUPPORTED_VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "mpeg", "mpg"]


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading YOLO model…")
def load_model(model_path_string: str) -> YOLO:
    """Load and cache the YOLO model."""
    path = Path(model_path_string).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Model file was not found: {path}")
    return YOLO(str(path))


def find_default_model() -> Optional[Path]:
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def get_model_names(model: YOLO) -> dict:
    names = model.names
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    return names


def render_detection_image(
    model: YOLO,
    image_rgb: np.ndarray,
    confidence: float,
    image_size: int,
    device: str,
):
    results = model.predict(
        source=image_rgb,
        conf=confidence,
        imgsz=image_size,
        device=device,
        verbose=False,
    )
    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    detections = []
    result = results[0]
    names = get_model_names(model)

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            score = float(box.conf.item())
            x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "Class": names.get(class_id, str(class_id)),
                    "Confidence": round(score, 4),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
    return annotated_rgb, detections


def process_video(
    model: YOLO,
    input_path: str,
    output_path: str,
    confidence: float,
    image_size: int,
    device: str,
    frame_skip: int = 1,
) -> tuple[int, float]:
    """Run detection frame by frame and write an annotated MP4."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("The video has invalid frame dimensions.")

    # MP4V is broadly available with OpenCV; output is written as MP4.
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(1, frame_skip),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the output video. Try another video format.")

    progress = st.progress(0, text="Processing video…")
    status = st.empty()
    processed = 0
    frame_index = 0
    start = time.time()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_index % max(1, frame_skip) == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                prediction = model.predict(
                    source=frame_rgb,
                    conf=confidence,
                    imgsz=image_size,
                    device=device,
                    verbose=False,
                )
                annotated_bgr = prediction[0].plot()
                if annotated_bgr.shape[1] != width or annotated_bgr.shape[0] != height:
                    annotated_bgr = cv2.resize(annotated_bgr, (width, height))
                writer.write(annotated_bgr)
                processed += 1

            frame_index += 1
            if frame_count > 0:
                pct = min(int(frame_index * 100 / frame_count), 100)
                progress.progress(pct, text=f"Processing video… {pct}%")
                status.caption(
                    f"Frames read: {frame_index:,}/{frame_count:,} · "
                    f"Frames analyzed: {processed:,}"
                )

        progress.progress(100, text="Video processing complete.")
        status.caption(
            f"Analyzed {processed:,} frames in {time.time() - start:.1f} seconds."
        )
    finally:
        cap.release()
        writer.release()

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Processing finished, but no output video was created.")

    return processed, time.time() - start


def main():
    st.title("🚗 Vehicle Detection App")
    st.write(
        "Detect vehicles and other trained classes in images or videos using your "
        "custom-trained YOLO11n model. Preview the result and download the annotated file."
    )

    with st.sidebar:
        st.header("Model & Settings")
        detected_default = find_default_model()
        default_model_path = str(detected_default) if detected_default else "best_vehicle_yolo11n.pt"
        model_path = st.text_input(
            "Model (.pt) path",
            value=os.environ.get("MODEL_PATH", default_model_path),
            help="On Streamlit Cloud, keep best_vehicle_yolo11n.pt beside app.py in your GitHub repository.",
        )
        confidence = st.slider(
            "Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05
        )
        image_size = st.select_slider(
            "Inference image size",
            options=[320, 416, 512, 640, 768, 960, 1280],
            value=640,
        )
        # Streamlit Community Cloud normally runs on CPU, so keep inference on CPU.
        device = "cpu"
        st.caption("Inference device: CPU (Streamlit Community Cloud)")
        st.caption("Lower confidence detects more objects but may add false positives.")

    if not model_path.strip():
        st.warning("Enter the path to your trained .pt model in the sidebar.")
        st.stop()

    if not Path(model_path).expanduser().is_file():
        st.error(
            "Model file not found. Place `best_vehicle_yolo11n.pt` beside `app.py`, "
            "or enter its full path in the sidebar."
        )
        st.info(
            "Upload `best_vehicle_yolo11n.pt` to the same GitHub repository folder as `app.py`. "
            "The filename must match exactly."
        )
        st.stop()

    try:
        model = load_model(model_path.strip())
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        st.stop()

    names = get_model_names(model)
    with st.expander("Model classes", expanded=False):
        st.write(", ".join(str(v) for v in names.values()))

    image_tab, video_tab = st.tabs(["🖼️ Image detection", "🎬 Video detection"])

    with image_tab:
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=SUPPORTED_IMAGE_TYPES,
            key="image_uploader",
        )
        if uploaded_image is not None:
            try:
                pil_image = Image.open(uploaded_image).convert("RGB")
                image_rgb = np.array(pil_image)
                left, right = st.columns(2)
                with left:
                    st.subheader("Original image")
                    st.image(image_rgb, use_container_width=True)
                with st.spinner("Detecting objects…"):
                    annotated_rgb, detections = render_detection_image(
                        model, image_rgb, confidence, int(image_size), device
                    )
                with right:
                    st.subheader("Detection result")
                    st.image(annotated_rgb, use_container_width=True)

                st.subheader(f"Detections ({len(detections)})")
                if detections:
                    st.dataframe(detections, use_container_width=True, hide_index=True)
                else:
                    st.info("No objects were detected. Try lowering the confidence threshold.")

                result_image = Image.fromarray(annotated_rgb)
                import io
                buffer = io.BytesIO()
                result_image.save(buffer, format="JPEG", quality=95)
                st.download_button(
                    "⬇️ Download annotated image",
                    data=buffer.getvalue(),
                    file_name=f"{Path(uploaded_image.name).stem}_detected.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Image processing failed: {exc}")

    with video_tab:
        st.write("Upload a video to annotate it. Longer or high-resolution videos take more time.")
        uploaded_video = st.file_uploader(
            "Upload a video",
            type=SUPPORTED_VIDEO_TYPES,
            key="video_uploader",
        )
        frame_skip = st.selectbox(
            "Analyze frames",
            options=[1, 2, 3, 5],
            index=0,
            format_func=lambda x: (
                "Every frame (best detail)" if x == 1 else f"Every {x} frames (faster, less detail)"
            ),
            help="Skipping frames makes processing faster but can miss brief detections.",
        )

        if uploaded_video is not None:
            st.video(uploaded_video)
            if st.button("▶️ Run video detection", type="primary", use_container_width=True):
                suffix = Path(uploaded_video.name).suffix or ".mp4"
                with tempfile.TemporaryDirectory(prefix="vehicle_detection_") as temp_dir:
                    input_path = os.path.join(temp_dir, f"input{suffix}")
                    output_path = os.path.join(temp_dir, "detected_video.mp4")

                    with open(input_path, "wb") as f:
                        f.write(uploaded_video.getbuffer())

                    try:
                        with st.spinner("Running YOLO detection on video frames…"):
                            frames, elapsed = process_video(
                                model=model,
                                input_path=input_path,
                                output_path=output_path,
                                confidence=confidence,
                                image_size=int(image_size),
                                device=device,
                                frame_skip=int(frame_skip),
                            )

                        st.success(f"Video processed successfully: {frames:,} frames analyzed in {elapsed:.1f}s.")
                        with open(output_path, "rb") as f:
                            video_bytes = f.read()

                        st.video(video_bytes)
                        st.download_button(
                            "⬇️ Download annotated video (MP4)",
                            data=video_bytes,
                            file_name=f"{Path(uploaded_video.name).stem}_detected.mp4",
                            mime="video/mp4",
                            type="primary",
                            use_container_width=True,
                        )
                    except Exception as exc:
                        st.error(f"Video processing failed: {exc}")
                        st.info(
                            "If you selected GPU but CUDA is unavailable, choose CPU. "
                            "If the video is very large, try analyzing every 2 or 3 frames."
                        )

    st.divider()
    st.caption(
        "Model metrics from your reported test run: Precision 69.50% · Recall 64.79% · "
        "mAP@50 69.20% · mAP@50–95 48.19%. Results on new videos may differ."
    )


if __name__ == "__main__":
    main()
