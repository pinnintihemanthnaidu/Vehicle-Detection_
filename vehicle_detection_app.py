from pathlib import Path
import time

import streamlit as st
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from ultralytics import YOLO

YOLO_WEIGHTS = Path(r"D:\CarAI\runs\vehicle_detection\weights\best.pt")
FASTER_WEIGHTS = Path(r"D:\CarAI\runs\faster_rcnn\best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 640

st.set_page_config(page_title="Aerial Vehicle Detection", page_icon="🚗", layout="wide")
st.title("🚗 Real-Time Vehicle Detection in Aerial Imagery")
st.caption("Compare YOLO11n and Faster R-CNN using your trained models.")

with st.sidebar:
    st.header("Detection settings")
    model_choice = st.selectbox("Model", ["YOLO11n", "Faster R-CNN"])
    confidence = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    st.write(f"**Device:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    st.write(f"**Image size:** {IMAGE_SIZE}px")

@st.cache_resource
def load_yolo():
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
    return YOLO(str(YOLO_WEIGHTS))

@st.cache_resource
def load_faster_rcnn():
    if not FASTER_WEIGHTS.exists():
        raise FileNotFoundError(f"Faster R-CNN weights not found: {FASTER_WEIGHTS}")
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, 2
    )
    model.transform.min_size = (IMAGE_SIZE,)
    model.transform.max_size = IMAGE_SIZE
    model.roi_heads.detections_per_img = 300
    checkpoint = torch.load(FASTER_WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    return model.to(DEVICE).eval()

def sync_device():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

def run_yolo(image, threshold):
    model = load_yolo()
    sync_device()
    start = time.perf_counter()
    result = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=threshold,
        device=0 if DEVICE.type == "cuda" else "cpu",
        verbose=False,
    )[0]
    sync_device()
    elapsed_ms = (time.perf_counter() - start) * 1000
    annotated = Image.fromarray(result.plot()[:, :, ::-1])
    count = 0 if result.boxes is None else len(result.boxes)
    return annotated, count, elapsed_ms

def run_faster(image, threshold):
    model = load_faster_rcnn()
    tensor = to_tensor(image).to(DEVICE)
    sync_device()
    start = time.perf_counter()
    with torch.inference_mode():
        result = model([tensor])[0]
    sync_device()
    elapsed_ms = (time.perf_counter() - start) * 1000

    boxes = result["boxes"].detach().cpu()
    scores = result["scores"].detach().cpu()
    labels = result["labels"].detach().cpu()
    keep = (scores >= threshold) & (labels == 1)
    boxes, scores = boxes[keep], scores[keep]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(float, box.tolist())
        draw.rectangle((x1, y1, x2, y2), outline="red", width=max(2, image.width // 400))
        draw.text((x1, max(0, y1 - 16)), f"car {float(score):.2f}", fill="red")
    return annotated, len(boxes), elapsed_ms

uploaded = st.file_uploader("Upload an aerial image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is None:
    st.info("Upload an aerial image to detect cars. The model will draw bounding boxes and report the number of detections and inference time.")
    st.markdown("### Models available")
    st.write("- **YOLO11n:** lightweight single-stage detector.")
    st.write("- **Faster R-CNN:** two-stage detector with a ResNet-50 FPN backbone.")
else:
    image = Image.open(uploaded).convert("RGB")
    st.subheader("Input image")
    st.image(image, use_container_width=True)

    if st.button("Detect vehicles", type="primary"):
        try:
            with st.spinner(f"Running {model_choice} on the uploaded image..."):
                if model_choice == "YOLO11n":
                    annotated, count, elapsed_ms = run_yolo(image, confidence)
                else:
                    annotated, count, elapsed_ms = run_faster(image, confidence)

            left, right = st.columns([3, 1])
            with left:
                st.subheader("Detection result")
                st.image(annotated, use_container_width=True)
            with right:
                st.subheader("Summary")
                st.metric("Detected cars", count)
                st.metric("Inference time", f"{elapsed_ms:.1f} ms")
                st.metric("Approx. model FPS", f"{1000 / elapsed_ms:.1f}" if elapsed_ms > 0 else "N/A")
                st.caption("FPS is calculated from this single model-call timing. It is not end-to-end application FPS.")
                import io
                image_buffer = io.BytesIO()
                annotated.save(image_buffer, format="PNG")
                st.download_button(
                    "Download annotated image",
                    data=image_buffer.getvalue(),
                    file_name="vehicle_detection_result.png",
                    mime="image/png",
                )
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            st.info("Confirm that the model checkpoint files exist at the paths configured at the top of app.py.")
