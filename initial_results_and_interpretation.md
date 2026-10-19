# Real-Time Vehicle Detection in Aerial Imagery — Initial Model Comparison

## Dataset and setup
- 102 labeled images total.
- Train/validation split: 81 training images and 21 validation images.
- Hardware: NVIDIA GeForce RTX 3050 Laptop GPU, 6 GB VRAM.
- YOLO model: YOLO11n.
- Two-stage detector: Faster R-CNN ResNet50 FPN.
- Both models were evaluated on the same held-out validation split.

## Results

| Metric | YOLO11n | Faster R-CNN ResNet50 FPN |
|---|---:|---:|
| Precision | 98.0% | 60.0%* |
| Recall | 86.3% | 94.5%* |
| mAP@50 | 94.4% | 95.5% |
| mAP@50–95 | 44.8% | 43.4% |
| Inference time | 8.1 ms/image | 56.2 ms/image |

*Precision/recall are not directly comparable: YOLO values came from Ultralytics validation, while Faster R-CNN precision/recall use a custom one-to-one matching procedure at confidence >= 0.25 and IoU >= 0.50. For a rigorous report, calculate both models' precision/recall with the same evaluator and confidence/IoU rules.

## Initial interpretation
- Faster R-CNN has a slightly higher mAP@50 in this run (95.5% vs 94.4%).
- YOLO11n has a slightly higher mAP@50–95 (44.8% vs 43.4%).
- YOLO11n is substantially faster in the reported inference timings: 8.1 ms/image versus 56.2 ms/image.
- Faster R-CNN's custom evaluation indicates higher recall but lower precision at the chosen thresholds; verify using a shared evaluator before making a definitive precision/recall claim.

## Limitations
- The validation set has only 21 images, so results are preliminary and may not generalize.
- Training for 10 epochs on this small dataset may not be sufficient to optimize Faster R-CNN.
- The timing values are framework-reported/locally measured under different pipelines; benchmark both models using the same warm-up, batch size, image size, device, and timing procedure before publishing a strict speed comparison.
- A high mAP@50 and lower mAP@50–95 can indicate that detections are often roughly correct but bounding-box localization is less consistent at stricter IoU thresholds.

## Recommended next steps
1. Inspect the annotated prediction images from both models.
2. Evaluate both models using a common metric implementation and identical confidence/IoU settings.
3. Benchmark both models using a shared timing script.
4. If time permits, increase the dataset size and use a fixed, documented train/validation split.
5. Use these results as preliminary experimental findings, not as proof of broad real-world performance.
