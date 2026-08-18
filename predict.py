"""
Step 5 — blend predictions from multiple trained models on a single image.

Each model contributes its own classification + regression blend (see
--regression-weight); those per-model blends are then averaged into one
final ensemble age estimate.

Usage:
    python predict.py --image path\\to\\face.jpg
    python predict.py --image path\\to\\face.jpg --checkpoints runs/resnet50/best_model.pt runs/efficientnet_b0/best_model.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from align_faces import align_image
from data import build_transforms
from face_detector import detect_landmarks
from model import DualHeadAgeModel

DEFAULT_CHECKPOINTS = [
    Path("runs/resnet50/best_model.pt"),
    Path("runs/efficientnet_b0/best_model.pt"),
    Path("runs/mobilenet_v2/best_model.pt"),
]


def predict_with_model(ckpt, x, device, regression_weight: float):
    min_age = ckpt["min_age"]
    model = DualHeadAgeModel(num_classes=ckpt["num_classes"], backbone=ckpt["backbone"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        class_logits, reg_out = model(x)
        probs = F.softmax(class_logits, dim=1)[0]

    top5 = torch.topk(probs, k=5)
    top5_ages = [(int(idx) + min_age, float(p)) for idx, p in zip(top5.indices, top5.values)]
    class_pred_age = top5_ages[0][0]
    reg_pred_age = float(reg_out.item())
    final_pred_age = regression_weight * reg_pred_age + (1 - regression_weight) * class_pred_age

    return {
        "backbone": ckpt["backbone"],
        "class_pred_age": class_pred_age,
        "class_confidence": top5_ages[0][1],
        "top5_ages": top5_ages,
        "reg_pred_age": reg_pred_age,
        "final_pred_age": final_pred_age,
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = Image.open(args.image).convert("RGB")
    landmarks = detect_landmarks(img)
    if landmarks is None:
        raise SystemExit(f"No face detected in {args.image}, cannot align for prediction.")

    checkpoints = [(path, torch.load(path, map_location=device)) for path in args.checkpoints]

    # Align once per distinct img_size, so face detection/alignment isn't repeated per model.
    aligned_cache = {}
    for _, ckpt in checkpoints:
        size = ckpt["img_size"]
        if size not in aligned_cache:
            aligned = align_image(img, landmarks, size=size)
            transform = build_transforms(train=False, img_size=size)
            aligned_cache[size] = transform(aligned).unsqueeze(0).to(device)

    print(f"\nImage: {args.image}")
    per_model = []
    for path, ckpt in checkpoints:
        result = predict_with_model(ckpt, aligned_cache[ckpt["img_size"]], device, args.regression_weight)
        per_model.append(result)
        print(f"\n[{result['backbone']}] ({path})")
        print(f"  Classification: {result['class_pred_age']} years (confidence {result['class_confidence']:.1%})")
        print("    Top-5: " + ", ".join(f"{a} ({p:.1%})" for a, p in result["top5_ages"]))
        print(f"  Regression:     {result['reg_pred_age']:.1f} years")
        print(f"  Model blend ({args.regression_weight:.0%} reg / {1 - args.regression_weight:.0%} class): "
              f"{result['final_pred_age']:.1f} years")

    ensemble_age = sum(r["final_pred_age"] for r in per_model) / len(per_model)
    spread = max(r["final_pred_age"] for r in per_model) - min(r["final_pred_age"] for r in per_model)
    print(f"\n=== Ensemble across {len(per_model)} models: {ensemble_age:.1f} years "
          f"(spread {spread:.1f} years) ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path, nargs="+", default=DEFAULT_CHECKPOINTS,
                     help="Model checkpoints to blend together.")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--regression-weight", type=float, default=0.7,
                     help="Weight on each model's regression head within its own blend; "
                          "the classification head gets (1 - this). Default favors regression, "
                          "which scored lower MAE than classification in evaluation.")
    args = ap.parse_args()
    main(args)
