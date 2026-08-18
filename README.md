# Age Prediction Pipeline (local)

Trains one model that predicts age two ways from the same face image —
classification (exact-age class) and regression (continuous age) — and
reports both side by side, using your `20-50` and `age_prediction_up`
datasets in `D:\NPN`.

Everything here runs on your own machine (no cloud upload of the images
needed). A GPU speeds things up a lot but everything falls back to CPU
automatically.

## 1. Setup

```
python -m venv venv
venv\Scripts\activate
```

**GPU (RTX 5060 Ti / other RTX 50-series, Blackwell):** install the CUDA
build of PyTorch *before* the rest of `requirements.txt`, using the CUDA 12.8
wheel index — Blackwell (sm_120) support has been in stable PyTorch since
2.7.0, no nightly build needed:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

(A cu130 index is also available if you want the newer CUDA 13 wheels —
either works for this GPU. Check https://pytorch.org/get-started/locally/ if
you're on a different/older GPU and need a different CUDA version.)

Verify it picked up the GPU before training:

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

should print `True` and `NVIDIA GeForce RTX 5060 Ti`.

**CPU only:** just `pip install -r requirements.txt` — training will work,
just much slower on tens of thousands of images.

**Face detection model:** face alignment (step 3) uses MediaPipe's
FaceLandmarker, which needs its model bundle downloaded once:

```
curl -L -o models\face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

*Troubleshooting (Windows N/KN editions only):* if `import cv2` (a mediapipe
dependency) fails with `ImportError: DLL load failed while importing cv2`,
it's because N editions of Windows ship without the Media Feature Pack.
Fix: `pip uninstall opencv-contrib-python -y && pip install opencv-python-headless`
(the headless build doesn't need Media Foundation).

## 2. Build the label manifest

```
python build_manifest.py --narrow-root "D:\NPN\20-50\20-50" --full-root "D:\NPN\age_prediction_up\age_prediction" --out manifest.csv
```

Walks both dataset folders and writes `manifest.csv` (columns: `filepath,
age, split, source_dataset, is_duplicate, keep`). Doesn't touch your original
images. Prints per-age counts so you can see class imbalance up front.

## 3. Align faces

```
python align_faces.py --manifest manifest.csv --aligned-dir aligned --out manifest_aligned.csv
```

Runs face detection + landmark alignment (MediaPipe FaceLandmarker) on every image and writes
aligned 128x128 copies under `aligned\` (originals untouched). Images where
no face is detected fall back to a resized (unaligned) copy and get
`alignment_ok=False` in `manifest_aligned.csv` so you can filter or inspect
them later instead of losing them silently.

This step is the slow one (a detector pass per image). To test the rest of
the pipeline first, run it on a subset:

```
python align_faces.py --manifest manifest.csv --aligned-dir aligned --out manifest_aligned_sample.csv --limit 1000
```

It's resumable — if you stop it partway through, re-running skips images
that already have an aligned output on disk.

## 4. Train

```
python train.py --manifest manifest_aligned.csv --epochs 30 --out-dir runs\exp1
```

Trains a shared backbone (default ResNet18) with two heads (classification +
regression), combined loss, early stopping on validation loss. Useful flags:

- `--dataset-scope {full,narrow,combined}` — default `combined` (both
  datasets, de-duplicated; `age_prediction_up` is treated as the base,
  `20-50` fills in extra density for that range).
- `--backbone {resnet18,resnet50,efficientnet_b0}` — default `resnet18`.
  `resnet50` is a bigger, usually more accurate backbone (slower to train,
  more VRAM); `efficientnet_b0` is a lighter alternative.
- `--class-loss-weight` (0-1, default 0.5) — how much the combined loss
  favors the classification head vs the regression head.
- `--batch-size`, `--epochs`, `--lr`, `--patience`

Saves `runs\exp1\best_model.pt`, `training_history.csv`, and
`test_split.csv` (the exact held-out test rows used, so evaluation matches
training).

**To compare backbones**, train each into its own `--out-dir` instead of
overwriting the same one:

```
python train.py --manifest manifest_aligned.csv --backbone resnet18 --out-dir runs\resnet18
python train.py --manifest manifest_aligned.csv --backbone resnet50 --out-dir runs\resnet50
```

Then evaluate each (step 5) with its matching `--checkpoint`/`--test-split`,
and combine the results:

```
python compare_backbones.py --run-dirs runs\resnet18 runs\resnet50 --out backbone_comparison.csv
```

This writes one table with classification and regression metrics for every
backbone side by side, and prints which backbone got the lowest MAE for
each approach.

## 5. Evaluate — classification vs regression, side by side

```
python evaluate.py --checkpoint runs\exp1\best_model.pt --test-split runs\exp1\test_split.csv
```

Writes to `runs\exp1\`:
- `summary_metrics.csv` — classification vs regression metrics in one table
  (exact accuracy, within-3-years, within-5-years, MAE, RMSE).
- `comparison_report.csv` — per-image: true age, both predictions, both
  errors, and how much the two methods disagree with each other.
- `predicted_vs_actual.png` — scatter plot for both methods.

## 6. Predict on a single image

```
python predict.py --checkpoint runs\exp1\best_model.pt --image "C:\path\to\face.jpg"
```

Prints both predictions (classification top-5 + confidence, and the
regression estimate) for that one image.

## Notes / things worth knowing

- **Dataset scope default:** `combined` uses both datasets. If you'd rather
  train on just one, pass `--dataset-scope narrow` or `full` to `train.py`
  (no need to rebuild the manifest).
- **Class imbalance:** ages at the edges of the range typically have fewer
  images. Watch the per-age counts `build_manifest.py` prints — if some ages
  are very sparse, accuracy at those ages will lag; consider capping how
  many images per age you use, or weighting the loss, if that shows up.
- **Alignment failures:** by default, images that failed alignment are still
  used (resized, not aligned) — pass `--drop-failed-alignment` to `train.py`
  to exclude them instead once you've seen how many there are.
- **Time estimate:** roughly proportional to dataset size × epochs; on a
  modern GPU expect low hours for the full combined dataset, meaningfully
  longer on CPU. Use `--limit` in step 3 and a small `--epochs` in step 4 to
  do a fast end-to-end dry run first and catch issues early.
