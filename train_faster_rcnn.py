"""
Faster R-CNN training for the Car Detection project.
Dataset expected:
D:/CarAI/dataset/images/train, val
D:/CarAI/dataset/labels/train, val
YOLO labels: class_id x_center y_center width height (normalized).
"""

from pathlib import Path
import time
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

ROOT = Path(r"D:\CarAI\dataset")
OUTPUT_DIR = Path(r"D:\CarAI\runs\faster_rcnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 10
BATCH_SIZE = 2
NUM_WORKERS = 0  # Safer on Windows
LEARNING_RATE = 0.005
NUM_CLASSES = 2  # background + car
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class YoloCarDataset(Dataset):
    def __init__(self, split):
        self.image_dir = ROOT / "images" / split
        self.label_dir = ROOT / "labels" / split
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.images = sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )
        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

        missing = [p.name for p in self.images if not (self.label_dir / f"{p.stem}.txt").exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} label files are missing. First few: {missing[:5]}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_tensor = to_tensor(image)

        boxes, labels, areas, crowds = [], [], [], []
        content = label_path.read_text(encoding="utf-8").strip()

        if content:
            for line_number, line in enumerate(content.splitlines(), start=1):
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(
                        f"Bad YOLO label in {label_path}, line {line_number}: {line}"
                    )
                class_id, xc, yc, bw, bh = map(float, parts)
                if int(class_id) != 0:
                    continue  # This project currently uses one class: car.

                x1 = max(0.0, (xc - bw / 2) * width)
                y1 = max(0.0, (yc - bh / 2) * height)
                x2 = min(float(width), (xc + bw / 2) * width)
                y2 = min(float(height), (yc + bh / 2) * height)
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(1)  # 0 is reserved for background in Faster R-CNN.
                areas.append((x2 - x1) * (y2 - y1))
                crowds.append(0)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(crowds, dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # Lower resolution helps fit the 6 GB laptop GPU.
    model.transform.min_size = (640,)
    model.transform.max_size = 640
    model.roi_heads.detections_per_img = 100
    return model


def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = YoloCarDataset("train")
    val_dataset = YoloCarDataset("val")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print("Loading pretrained Faster R-CNN weights (first run may download weights)...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = build_model().to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_loss = float("inf")
    history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        start = time.time()

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, step {step}: {loss}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_loss += loss.item()

            if step % 10 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"step {step}/{len(train_loader)} | loss {loss.item():.4f}"
                )

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))
        elapsed = time.time() - start
        history.append({"epoch": epoch + 1, "train_loss": avg_loss, "seconds": elapsed})
        print(
            f"Epoch {epoch + 1}/{EPOCHS} complete | "
            f"average loss: {avg_loss:.4f} | time: {elapsed:.1f}s"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "class_names": ["background", "car"],
            "train_loss": avg_loss,
        }
        torch.save(checkpoint, OUTPUT_DIR / "last.pth")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, OUTPUT_DIR / "best.pth")

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    import csv
    history_path = OUTPUT_DIR / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "seconds"])
        writer.writeheader()
        writer.writerows(history)

    print("\nFaster R-CNN training completed.")
    print(f"Best checkpoint: {OUTPUT_DIR / 'best.pth'}")
    print(f"Last checkpoint: {OUTPUT_DIR / 'last.pth'}")
    print(f"Training history: {history_path}")
    print("Note: training loss is not the same as validation mAP. Evaluate on the same validation set before comparing with YOLO.")


if __name__ == "__main__":
    main()
