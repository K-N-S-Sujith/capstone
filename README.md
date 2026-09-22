# RetinexNet for Low-Light Image Restoration (LOL-v2-Real)

Part of the project **"Beyond Pixel Fidelity: Task-Guided Low-Light Image
Restoration for Improved Object Detection"** — this repository implements
ONLY the RetinexNet branch (one of four models compared: MSRCR,
RetinexNet, URWKV, EnlightenGAN).

## 1. Theory Primer (for your viva)

### What is Retinex theory?
Retinex theory (from "retina" + "cortex") models how humans perceive
consistent object color regardless of lighting. It assumes any image `I`
can be split into two physically meaningful components:

```
I = R × L
```

- **R (Reflectance):** the intrinsic color/texture of surfaces — what the
  scene would look like under perfectly even, unit-intensity light.
  Reflectance does not change when the lighting changes.
- **L (Illumination):** how much light is falling on each pixel. This DOES
  change between a dark photo and a well-lit photo of the same scene.

### Why can low-light images be decomposed this way?
A photo taken in darkness and the "same" photo taken with the lights on
show the SAME objects (same reflectance) under DIFFERENT lighting (different
illumination). If we can separate these two factors, we can enhance a
low-light image by only boosting `L` while keeping `R` (the actual scene
content) untouched — avoiding color distortion or texture loss.

### What does Decom-Net do?
Decom-Net takes a single RGB image and predicts its reflectance (3
channels) and illumination (1 channel). It is trained on PAIRS of
low/high-light images so it learns that both images of a pair should
decompose to nearly the SAME reflectance, with all the visual difference
pushed into illumination.

### What does Enhance-Net do?
Enhance-Net takes the (reflectance, illumination) from a LOW-light image
and predicts a BRIGHTENED illumination map. It never touches reflectance —
so the final image keeps the same colors/texture, just relit.

### How is the enhanced image reconstructed?
```
Enhanced Image = Reflectance × Enhanced Illumination
```
This is literally the Retinex equation run in reverse, using the enhanced
`L` in place of the original dark `L`.

### What do the losses do?
See `losses/retinex_losses.py` docstring for the full math. In one line
each:
- **Reconstruction loss:** makes sure R × L actually reproduces the input image.
- **Reflectance consistency loss:** forces the low/high pair to share the same reflectance.
- **Illumination smoothness loss:** keeps illumination spatially smooth, except at real object edges.
- **Illumination enhancement loss (Stage 2):** directly supervises the brightened illumination to match the ground-truth illumination.

### How does training work?
Two stages, trained separately:
1. Train Decom-Net on low+high pairs (reconstruction + reflectance + smoothness losses).
2. Freeze Decom-Net. Train Enhance-Net to brighten low-light illumination toward high-light illumination.

### How does testing work?
Load both trained networks, run a low-light image through Decom-Net (get
R, L), then through Enhance-Net (get enhanced L), then multiply R by
enhanced L to get the final image.

### How does this fit the 4-model comparison?
RetinexNet's enhanced test images are saved to `results/retinexnet/` in
the exact same format/filenames as the other three methods will produce,
so `yolo_evaluate.py` can be pointed at any of the four folders using the
SAME YOLO weights — giving a fair, apples-to-apples downstream detection
comparison.

## 2. Project Structure

```
RetinexNet_Project/
├── data/
├── models/
│   ├── __init__.py
│   ├── decom_net.py
│   ├── enhance_net.py
│   └── retinexnet.py
├── losses/
│   ├── __init__.py
│   └── retinex_losses.py
├── datasets/
│   ├── __init__.py
│   └── lol_dataset.py
├── utils/
│   ├── metrics.py
│   ├── visualization.py
│   └── checkpoint.py
├── results/
├── checkpoints/
├── train.py
├── test.py
├── evaluate.py
├── yolo_evaluate.py
├── config.py
├── requirements.txt
└── README.md
```

## 3. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Dataset Layout

Point `--data_root` at a folder shaped like:

```
dataset/
├── train/
│   ├── low/
│   └── high/
├── val/
│   ├── low/
│   └── high/
└── test/
    ├── low/
    └── high/
```

Low/high images must share filenames within each split (LOL-v2-Real's
native format).

## 5. Training

```bash
python train.py \
    --data_root dataset \
    --batch_size 8 \
    --patch_size 384 \
    --epochs_decom 100 \
    --epochs_enhance 100 \
    --checkpoint_dir checkpoints \
    --sample_dir results/val_samples
```

This runs Stage 1 (Decom-Net) to completion, then Stage 2 (Enhance-Net).
Best checkpoints are saved as `checkpoints/decom_best.pth` and
`checkpoints/enhance_best.pth`. Periodic checkpoints (`decom_epochN.pth`,
`enhance_epochN.pth`) are saved every `--save_every` epochs (default 10).

To resume an interrupted run:
```bash
python train.py --resume_decom checkpoints/decom_epoch50.pth ...
```

## 6. Testing (generate enhanced images)

```bash
python test.py \
    --test_low_dir dataset/test/low \
    --decom_checkpoint checkpoints/decom_best.pth \
    --enhance_checkpoint checkpoints/enhance_best.pth \
    --output_dir results/retinexnet
```

Add `--save_components` to also dump reflectance/illumination/enhanced
illumination maps per image (saved under `results/retinexnet/components/`).

## 7. Evaluation (PSNR / SSIM / LPIPS)

```bash
python evaluate.py \
    --enhanced_dir results/retinexnet \
    --gt_dir dataset/test/high \
    --output_csv results/retinexnet_metrics.csv
```

Produces a CSV: `filename,psnr,ssim,lpips` plus printed averages.

## 8. YOLO Downstream Evaluation (optional, separate from RetinexNet training)

```bash
python yolo_evaluate.py \
    --enhanced_dir results/retinexnet \
    --data_yaml yolo_dataset/data.yaml \
    --yolo_weights yolov8n.pt \
    --output_csv results/retinexnet_yolo_metrics.csv
```

Re-run with `--enhanced_dir` pointed at MSRCR/URWKV/EnlightenGAN outputs
(same `--yolo_weights`, same `--data_yaml`) for a fair 4-way comparison.

## 9. Expected Folder Structure After Training/Testing

```
checkpoints/
├── decom_best.pth
├── decom_epoch10.pth ... decom_epoch100.pth
├── enhance_best.pth
└── enhance_epoch10.pth ... enhance_epoch100.pth
results/
├── val_samples/
│   └── epoch_N_<filename>.png     # 6-panel visual comparison
├── retinexnet/
│   ├── <filename>.png             # final enhanced test images
│   └── components/                # only if --save_components used
├── retinexnet_metrics.csv
└── retinexnet_yolo_metrics.csv
```

## 10. Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Dataset folder not found` | Wrong `--data_root` / folder names | Ensure `train/low`, `train/high`, etc. exist under the given root |
| `Mismatched pair counts` | Low/high folders have different numbers of images | LOL-v2-Real requires exact 1:1 pairs; check for missing files |
| CUDA out of memory | `--batch_size` or `--patch_size` too large for your GPU | Lower `--batch_size` (e.g. 4) or `--patch_size` (e.g. 256) |
| LPIPS shows `N/A` / `nan` | `lpips` package not installed | `pip install lpips` |
| `ultralytics` ImportError in `yolo_evaluate.py` | Package not installed | `pip install ultralytics` |
| Enhanced images look nearly identical to input | Enhance-Net undertrained or Decom-Net checkpoint not loaded | Verify `--decom_checkpoint`/`--enhance_checkpoint` paths point to *_best.pth, check training loss curves |

## 11. One-Paragraph Viva Summary

"RetinexNet enhances low-light images by first decomposing them into a
reflectance map (the scene's true, illumination-invariant color/texture)
and an illumination map (how bright each region is), based on the
physical assumption that an image equals reflectance times illumination.
A network called Decom-Net learns this split by training on paired
dark/well-lit photos of the same scene, learning that both should share
one reflectance but differ in illumination. A second network, Enhance-Net,
then learns to brighten the illumination map toward what a well-lit
photo's illumination looks like. Multiplying the original reflectance by
this brightened illumination gives the final enhanced image — colors and
texture are preserved exactly, only the lighting changes. This output
feeds into the same YOLO detector used for the other three enhancement
methods in my comparison, so I can measure whether RetinexNet's
restoration actually helps downstream object detection, not just whether
it looks good to the human eye."
