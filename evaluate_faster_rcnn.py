from pathlib import Path
import csv, time, json
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

ROOT = Path(r"D:\CarAI\dataset")
MODEL_PATH = Path(r"D:\CarAI\runs\faster_rcnn\best.pth")
OUT = Path(r"D:\CarAI\runs\faster_rcnn_eval")
PRED_DIR = OUT / "predictions"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF = 0.25
IOU_THR = 0.50

class ValDataset(Dataset):
    def __init__(self):
        self.images = sorted(p for p in (ROOT / "images" / "val").iterdir()
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        self.label_dir = ROOT / "labels" / "val"
        if not self.images:
            raise RuntimeError("No validation images found.")
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        p = self.images[i]
        im = Image.open(p).convert("RGB")
        w, h = im.size
        boxes, labels = [], []
        lp = self.label_dir / f"{p.stem}.txt"
        if not lp.exists():
            raise FileNotFoundError(lp)
        text = lp.read_text(encoding="utf-8").strip()
        if text:
            for line in text.splitlines():
                vals = line.split()
                if len(vals) != 5:
                    continue
                cls, xc, yc, bw, bh = map(float, vals)
                if int(cls) != 0:
                    continue
                x1, y1 = max(0, (xc-bw/2)*w), max(0, (yc-bh/2)*h)
                x2, y2 = min(w, (xc+bw/2)*w), min(h, (yc+bh/2)*h)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1,y1,x2,y2]); labels.append(1)
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1,4),
                  "labels": torch.tensor(labels, dtype=torch.int64)}
        return p.name, to_tensor(im), target, im.copy()

def collate(batch):
    return tuple(zip(*batch))

def build_model():
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.transform.min_size = (640,)
    model.transform.max_size = 640
    model.roi_heads.detections_per_img = 100
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    return model

def greedy_counts(pred_boxes, pred_scores, gt_boxes):
    order = torch.argsort(pred_scores, descending=True)
    matched, tp, fp = set(), 0, 0
    for ix in order.tolist():
        if len(gt_boxes):
            b = pred_boxes[ix]
            xx1 = torch.maximum(b[0], gt_boxes[:,0]); yy1 = torch.maximum(b[1], gt_boxes[:,1])
            xx2 = torch.minimum(b[2], gt_boxes[:,2]); yy2 = torch.minimum(b[3], gt_boxes[:,3])
            inter = (xx2-xx1).clamp(min=0)*(yy2-yy1).clamp(min=0)
            a = (b[2]-b[0]).clamp(min=0)*(b[3]-b[1]).clamp(min=0)
            ga = (gt_boxes[:,2]-gt_boxes[:,0]).clamp(min=0)*(gt_boxes[:,3]-gt_boxes[:,1]).clamp(min=0)
            ious = inter/(a+ga-inter).clamp(min=1e-9)
            best_iou, best_idx = ious.max(dim=0)
            if best_iou.item() >= IOU_THR and int(best_idx) not in matched:
                tp += 1; matched.add(int(best_idx))
            else:
                fp += 1
        else:
            fp += 1
    return tp, fp, len(gt_boxes)-len(matched)

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    OUT.mkdir(parents=True, exist_ok=True); PRED_DIR.mkdir(parents=True, exist_ok=True)
    ds = ValDataset()
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    model = build_model().to(DEVICE).eval()
    print("Device:", DEVICE)
    if DEVICE.type == "cuda": print("GPU:", torch.cuda.get_device_name(0))
    print("Validation images:", len(ds))
    with torch.inference_mode():
        model([ds[0][1].to(DEVICE)])  # warm-up
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        totals = [0,0,0]; times=[]; rows=[]
        metric = None
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
            metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        except Exception as e:
            print("mAP metric unavailable; basic P/R/F1 will still be calculated:", str(e))
        for names, images, targets, pil_images in loader:
            name, image, target, pil = names[0], images[0], targets[0], pil_images[0]
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t0=time.perf_counter(); output=model([image.to(DEVICE)])[0]
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            ms=(time.perf_counter()-t0)*1000; times.append(ms)
            boxes=output["boxes"].detach().cpu(); scores=output["scores"].detach().cpu()
            labels=output["labels"].detach().cpu()
            keep=(scores>=CONF)&(labels==1)
            pb, ps=boxes[keep], scores[keep]
            tp,fp,fn=greedy_counts(pb,ps,target["boxes"])
            totals[0]+=tp; totals[1]+=fp; totals[2]+=fn
            if metric is not None:
                metric.update([{"boxes":boxes[labels==1], "scores":scores[labels==1],
                                "labels":torch.ones(int((labels==1).sum()),dtype=torch.int64)}],
                              [{"boxes":target["boxes"], "labels":target["labels"]}])
            draw=ImageDraw.Draw(pil)
            for b,s in zip(pb,ps):
                x1,y1,x2,y2=map(float,b.tolist())
                draw.rectangle((x1,y1,x2,y2),outline="red",width=3)
                draw.text((x1,max(0,y1-14)),f"car {float(s):.2f}",fill="red")
            pil.save(PRED_DIR/name)
            rows.append({"image":name,"ground_truth_cars":len(target["boxes"]),
                         "predicted_cars_conf_0.25":len(pb),"true_positives":tp,
                         "false_positives":fp,"false_negatives":fn,"inference_ms":round(ms,3)})
            print(f"{name}: {len(pb)} detections, {ms:.1f} ms")
    tp,fp,fn=totals
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    summary={"model":"Faster R-CNN ResNet50 FPN","validation_images":len(ds),
             "confidence_threshold":CONF,"matching_iou_threshold":IOU_THR,
             "true_positives":tp,"false_positives":fp,"false_negatives":fn,
             "precision_at_conf_0.25_iou_0.50":precision,
             "recall_at_conf_0.25_iou_0.50":recall,"f1_at_conf_0.25_iou_0.50":f1,
             "mean_inference_ms_per_image":sum(times)/len(times),
             "note":"P/R/F1 use one-to-one greedy matching at IoU 0.50 and confidence >= 0.25; not directly identical to Ultralytics validation metrics."}
    if metric is not None:
        ap=metric.compute()
        summary["map_50"]=float(ap["map_50"]); summary["map_50_95"]=float(ap["map"])
    (OUT/"metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (OUT/"per_image_results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print("\n=== Faster R-CNN evaluation ===")
    for k,v in summary.items(): print(f"{k}: {v}")
    print("Predictions:",PRED_DIR)
    print("Metrics:",OUT/"metrics.json")
    print("Per-image results:",OUT/"per_image_results.csv")

if __name__ == "__main__":
    main()
