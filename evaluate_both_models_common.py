"""
Common evaluator for YOLO11n and Faster R-CNN on the same aerial-car validation set.
Writes a comparable JSON summary, CSV table, and annotated predictions.
"""

from pathlib import Path
import csv
import json
import time

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from ultralytics import YOLO

ROOT = Path(r"D:\CarAI\dataset")
YOLO_WEIGHTS = Path(r"D:\CarAI\runs\vehicle_detection\weights\best.pt")
FASTER_WEIGHTS = Path(r"D:\CarAI\runs\faster_rcnn\best.pth")
OUT = Path(r"D:\CarAI\runs\common_evaluation")
PRED_ROOT = OUT / "predictions"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 640


class ValidationDataset(Dataset):
    def __init__(self):
        self.image_dir = ROOT / "images" / "val"
        self.label_dir = ROOT / "labels" / "val"
        self.images = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not self.images:
            raise RuntimeError(f"No validation images found: {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        image = Image.open(path).convert("RGB")
        w, h = image.size
        boxes = []
        labels = []
        label_path = self.label_dir / f"{path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")
        text = label_path.read_text(encoding="utf-8").strip()
        if text:
            for line in text.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid YOLO label in {label_path}: {line}")
                cls, xc, yc, bw, bh = map(float, parts)
                if int(cls) != 0:
                    continue
                x1 = max(0.0, (xc - bw / 2) * w)
                y1 = max(0.0, (yc - bh / 2) * h)
                x2 = min(float(w), (xc + bw / 2) * w)
                y2 = min(float(h), (yc + bh / 2) * h)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(1)  # normalize the car class to label 1 for both models
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return path.name, image, target


def build_faster_rcnn():
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


def iou_against_gt(box, gt_boxes):
    if len(gt_boxes) == 0:
        return torch.zeros(0)
    x1 = torch.maximum(box[0], gt_boxes[:, 0])
    y1 = torch.maximum(box[1], gt_boxes[:, 1])
    x2 = torch.minimum(box[2], gt_boxes[:, 2])
    y2 = torch.minimum(box[3], gt_boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area2 = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
    return inter / (area1 + area2 - inter).clamp(min=1e-9)


def count_matches(boxes, scores, gt_boxes):
    order = torch.argsort(scores, descending=True)
    matched = set()
    tp = fp = 0
    for i in order.tolist():
        if len(gt_boxes):
            ious = iou_against_gt(boxes[i], gt_boxes)
            best_iou, best_idx = ious.max(dim=0)
            if best_iou.item() >= IOU_THRESHOLD and int(best_idx) not in matched:
                tp += 1
                matched.add(int(best_idx))
            else:
                fp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched)
    return tp, fp, fn


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def get_yolo_prediction(model, pil_image):
    # Run at low threshold so mAP sees a broad set of scored predictions.
    result = model.predict(
        source=pil_image,
        imgsz=IMAGE_SIZE,
        conf=0.001,
        iou=0.7,
        max_det=300,
        device=0 if DEVICE.type == "cuda" else "cpu",
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        empty = torch.zeros((0, 4), dtype=torch.float32)
        return {"boxes": empty, "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)}
    boxes = result.boxes.xyxy.detach().cpu().float()
    scores = result.boxes.conf.detach().cpu().float()
    classes = result.boxes.cls.detach().cpu().long()
    car = classes == 0
    return {"boxes": boxes[car], "scores": scores[car],
            "labels": torch.ones(int(car.sum()), dtype=torch.int64)}


def get_faster_prediction(model, pil_image):
    tensor = to_tensor(pil_image).to(DEVICE)
    with torch.inference_mode():
        result = model([tensor])[0]
    boxes = result["boxes"].detach().cpu().float()
    scores = result["scores"].detach().cpu().float()
    labels = result["labels"].detach().cpu().long()
    car = labels == 1
    return {"boxes": boxes[car], "scores": scores[car],
            "labels": torch.ones(int(car.sum()), dtype=torch.int64)}


def calculate_model(name, model, dataset, predictor, map_metric):
    print(f"\nEvaluating {name}...")
    pred_dir = PRED_ROOT / ("yolo" if name == "YOLO11n" else "faster_rcnn")
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Warm up on up to three images; exclude warm-up from timing.
    for i in range(min(3, len(dataset))):
        predictor(model, dataset[i][1])
    sync()

    rows = []
    total_tp = total_fp = total_fn = 0
    times = []

    for idx in range(len(dataset)):
        filename, image, target = dataset[idx]
        sync()
        t0 = time.perf_counter()
        prediction = predictor(model, image)
        sync()
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)

        # Common fixed threshold for P/R/F1.
        keep = prediction["scores"] >= CONF_THRESHOLD
        boxes = prediction["boxes"][keep]
        scores = prediction["scores"][keep]
        tp, fp, fn = count_matches(boxes, scores, target["boxes"])
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Same mAP implementation for both; use all scored predictions.
        map_metric.update(
            [{
                "boxes": prediction["boxes"],
                "scores": prediction["scores"],
                "labels": prediction["labels"],
            }],
            [{
                "boxes": target["boxes"],
                "labels": target["labels"],
            }],
        )

        # Save a visual sample for manual inspection.
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(float, box.tolist())
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((x1, max(0, y1 - 14)), f"car {float(score):.2f}", fill="red")
        annotated.save(pred_dir / filename)

        rows.append({
            "model": name,
            "image": filename,
            "ground_truth_cars": len(target["boxes"]),
            "predictions_conf_ge_0.25": len(boxes),
            "true_positives_iou_ge_0.50": tp,
            "false_positives_iou_ge_0.50": fp,
            "false_negatives_iou_ge_0.50": fn,
            "inference_ms": round(ms, 3),
        })
        print(f"{filename}: {len(boxes)} detections, {ms:.1f} ms")

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    map_result = map_metric.compute()
    summary = {
        "model": name,
        "validation_images": len(dataset),
        "confidence_threshold_for_precision_recall_f1": CONF_THRESHOLD,
        "iou_threshold_for_precision_recall_f1": IOU_THRESHOLD,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map_50": float(map_result["map_50"]),
        "map_50_95": float(map_result["map"]),
        "mean_inference_ms_per_image_excluding_warmup": sum(times) / len(times),
        "timing_note": "Model forward/inference call only; common warm-up and synchronization used. Pre/postprocessing and image I/O are excluded.",
    }
    print(f"{name} summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary, rows


def main():
    for path in [YOLO_WEIGHTS, FASTER_WEIGHTS]:
        if not path.exists():
            raise FileNotFoundError(path)
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        # Ensure the COCO backend is installed before starting a long run.
        test_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        del test_metric
    except Exception as e:
        raise RuntimeError(
            "Could not initialize TorchMetrics mAP. Run: "
            r"D:\CarAI\.venv\Scripts\python.exe -m pip install torchmetrics pycocotools"
        ) from e

    OUT.mkdir(parents=True, exist_ok=True)
    dataset = ValidationDataset()
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Validation images: {len(dataset)}")

    yolo = YOLO(str(YOLO_WEIGHTS))
    faster = build_faster_rcnn()

    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    # Separate metric instance for each model.
    yolo_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    faster_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    yolo_summary, yolo_rows = calculate_model("YOLO11n", yolo, dataset, get_yolo_prediction, yolo_metric)
    faster_summary, faster_rows = calculate_model("Faster R-CNN ResNet50 FPN", faster, dataset, get_faster_prediction, faster_metric)

    summaries = [yolo_summary, faster_summary]
    (OUT / "common_metrics.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with (OUT / "model_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["model", "validation_images", "precision", "recall", "f1", "map_50", "map_50_95",
                  "mean_inference_ms_per_image_excluding_warmup"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k) for k in fields})
    with (OUT / "per_image_results.csv").open("w", newline="", encoding="utf-8") as f:
        all_rows = yolo_rows + faster_rows
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n=== COMMON EVALUATION COMPLETE ===")
    print(f"Summary JSON: {OUT / 'common_metrics.json'}")
    print(f"Comparison CSV: {OUT / 'model_comparison.csv'}")
    print(f"Per-image CSV: {OUT / 'per_image_results.csv'}")
    print(f"YOLO predictions: {PRED_ROOT / 'yolo'}")
    print(f"Faster R-CNN predictions: {PRED_ROOT / 'faster_rcnn'}")


if __name__ == "__main__":
    main()
