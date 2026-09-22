"""
yolo_evaluate.py

Runs a YOLO object detector on RetinexNet's enhanced images and reports
detection metrics (mAP, Precision, Recall, F1). This script is deliberately
SEPARATE from RetinexNet training — RetinexNet is optimized only for image
restoration (PSNR/SSIM/LPIPS), never for detection accuracy directly.

Because this same script's interface (enhanced-image folder + YOLO
weights + label folder -> metrics) is reused for MSRCR, URWKV, and
EnlightenGAN outputs, always pointing at the SAME YOLO weights/config,
your four-model comparison is guaranteed to be apples-to-apples: the only
thing that changes between runs is which enhancement folder is passed in.

Requires the `ultralytics` package (YOLOv8/YOLO11) and a YOLO-format
dataset (images + labels + a data.yaml describing class names and paths).

Usage:
    python yolo_evaluate.py --enhanced_dir results/retinexnet \
        --data_yaml yolo_dataset/data.yaml \
        --yolo_weights yolov8n.pt \
        --output_csv results/retinexnet_yolo_metrics.csv

Note: `data.yaml` must point its validation image path at `enhanced_dir`
(or a copy of it) with matching YOLO-format label files already present —
this script does not generate labels, it only runs inference/evaluation
against existing ground-truth annotations.
"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO detector on RetinexNet-enhanced images"
    )
    parser.add_argument("--enhanced_dir", type=str, required=True,
                         help="Folder of enhanced images to run detection on "
                              "(must be referenced by data_yaml's val path).")
    parser.add_argument("--data_yaml", type=str, required=True,
                         help="YOLO dataset YAML (class names + image/label paths).")
    parser.add_argument("--yolo_weights", type=str, required=True,
                         help="Path to YOLO weights (.pt), e.g. yolov8n.pt or a custom-trained model. "
                              "Use the SAME weights across MSRCR/RetinexNet/URWKV/EnlightenGAN runs.")
    parser.add_argument("--output_csv", type=str, default="results/retinexnet_yolo_metrics.csv")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "The 'ultralytics' package is required for YOLO evaluation. "
            "Install it with: pip install ultralytics"
        ) from e

    print(f"Loading YOLO weights from: {args.yolo_weights}")
    model = YOLO(args.yolo_weights)

    print(f"Running validation on enhanced images described in: {args.data_yaml}")
    metrics = model.val(data=args.data_yaml, imgsz=args.imgsz, conf=args.conf)

    # Ultralytics' `metrics.box` exposes standard detection metrics.
    map50 = float(metrics.box.map50)      # mAP at IoU=0.50
    map50_95 = float(metrics.box.map)     # mAP averaged over IoU=0.50:0.95
    precision = float(metrics.box.mp)     # mean precision across classes
    recall = float(metrics.box.mr)        # mean recall across classes
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("\n===== YOLO Detection Metrics (RetinexNet-enhanced images) =====")
    print(f"mAP@0.50:      {map50:.4f}")
    print(f"mAP@0.50:0.95: {map50_95:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1-score:      {f1:.4f}")

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["mAP50", f"{map50:.4f}"])
        writer.writerow(["mAP50-95", f"{map50_95:.4f}"])
        writer.writerow(["precision", f"{precision:.4f}"])
        writer.writerow(["recall", f"{recall:.4f}"])
        writer.writerow(["f1", f"{f1:.4f}"])

    print(f"\nMetrics written to: {args.output_csv}")
    print("Re-run this exact script with --enhanced_dir pointed at MSRCR / URWKV / "
          "EnlightenGAN outputs (same --yolo_weights and --data_yaml) for a fair comparison.")


if __name__ == "__main__":
    main()
