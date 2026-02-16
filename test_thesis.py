import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_default_config
from thesis_model import Enhanced_STF_Detector
from train_thesis import DeepfakePairDataset


def parse_args() -> argparse.Namespace:
    cfg = get_default_config()
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    test_cfg = cfg["test"]

    parser = argparse.ArgumentParser("Test Enhanced_STF_Detector")
    parser.add_argument("--data_root", type=str, default=data_cfg["test_root"], help="Test root dir with class folders.")
    parser.add_argument("--checkpoint", type=str, default=test_cfg["checkpoint"], help="Checkpoint path.")
    parser.add_argument("--output_csv", type=str, default=test_cfg["output_csv"])
    parser.add_argument("--video_output_csv", type=str, default=test_cfg["video_output_csv"], help="Optional video-level csv path.")
    parser.add_argument("--threshold", type=float, default=test_cfg["threshold"])

    parser.add_argument("--batch_size", type=int, default=test_cfg["batch_size"])
    parser.add_argument("--num_workers", type=int, default=test_cfg["num_workers"])
    parser.add_argument("--image_size", type=int, default=data_cfg["image_size"])
    parser.add_argument("--motion_stride", type=int, default=data_cfg["motion_stride"])
    parser.add_argument("--max_frames_per_video", type=int, default=data_cfg["max_frames_per_video"])

    # Optional overrides. If omitted, try loading from checkpoint config.
    parser.add_argument("--fusion_mode", type=str, default="", choices=["", "independent", "cross_attention"])
    parser.add_argument("--feature_dim", type=int, default=0)
    parser.add_argument("--hfri_mode", type=str, default="", choices=["", "fft", "dct"])
    parser.add_argument("--flowformer_repo", type=str, default="")
    parser.add_argument("--flowformer_ckpt", type=str, default="")

    return parser.parse_args()


def safe_float(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def infer_model_args(test_args: argparse.Namespace, ckpt_cfg: Dict) -> Dict:
    model_cfg = get_default_config()["model"]

    fusion_mode = test_args.fusion_mode if test_args.fusion_mode else ckpt_cfg.get("fusion_mode", model_cfg["fusion_mode"])
    feature_dim = test_args.feature_dim if test_args.feature_dim > 0 else int(ckpt_cfg.get("feature_dim", model_cfg["feature_dim"]))
    hfri_mode = test_args.hfri_mode if test_args.hfri_mode else ckpt_cfg.get("hfri_mode", model_cfg["hfri_mode"])
    require_flowformer = bool(ckpt_cfg.get("require_flowformer", model_cfg["require_flowformer"]))
    flowformer_repo = (
        test_args.flowformer_repo if test_args.flowformer_repo else ckpt_cfg.get("flowformer_repo", model_cfg["flowformer_repo"])
    )
    flowformer_ckpt = (
        test_args.flowformer_ckpt if test_args.flowformer_ckpt else ckpt_cfg.get("flowformer_ckpt", model_cfg["flowformer_ckpt"])
    )

    return {
        "fusion_mode": fusion_mode,
        "feature_dim": feature_dim,
        "hfri_mode": hfri_mode,
        "require_flowformer": require_flowformer,
        "flowformer_repo": flowformer_repo,
        "flowformer_ckpt": flowformer_ckpt,
    }


def compute_auc_ap(labels: List[float], probs: List[float]) -> Tuple[float, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(set(labels)) < 2:
            return float("nan"), float("nan")
        return safe_float(roc_auc_score(labels, probs)), safe_float(average_precision_score(labels, probs))
    except Exception:
        return float("nan"), float("nan")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model_args = infer_model_args(args, ckpt_cfg)
    model = Enhanced_STF_Detector(
        feature_dim=model_args["feature_dim"],
        fusion_mode=model_args["fusion_mode"],
        use_resnet_imagenet=False,
        hfri_mode=model_args["hfri_mode"],
        require_flowformer=model_args["require_flowformer"],
        flowformer_repo=model_args["flowformer_repo"],
        flowformer_ckpt=model_args["flowformer_ckpt"],
    ).to(device)

    load_msg = model.load_state_dict(state_dict, strict=True)
    print(
        f"[Checkpoint] loaded from {args.checkpoint}\n"
        f"  missing_keys={len(load_msg.missing_keys)} unexpected_keys={len(load_msg.unexpected_keys)}"
    )
    print(
        f"[Model] fusion_mode={model_args['fusion_mode']} feature_dim={model_args['feature_dim']} "
        f"hfri_mode={model_args['hfri_mode']}"
    )

    dataset = DeepfakePairDataset(
        root=args.data_root,
        image_size=args.image_size,
        split="val",
        motion_stride=args.motion_stride,
        max_frames_per_video=args.max_frames_per_video,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model.eval()
    frame_rows: List[Dict] = []
    frame_probs: List[float] = []
    frame_labels: List[float] = []
    video_pool: Dict[str, List[Tuple[float, float]]] = {}

    with torch.no_grad():
        for batch in loader:
            img_spatial = batch["img_spatial"].to(device, non_blocking=True)
            img_motion_1 = batch["img_motion_1"].to(device, non_blocking=True)
            img_motion_2 = batch["img_motion_2"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)  # [B]

            outputs = model(img_spatial, img_motion_1, img_motion_2)
            probs = torch.sigmoid(outputs["logits"]).squeeze(1)  # [B]

            probs_np = probs.detach().cpu().numpy().tolist()
            labels_np = labels.detach().cpu().numpy().tolist()
            preds_np = [1.0 if p >= args.threshold else 0.0 for p in probs_np]

            frame_probs.extend([float(p) for p in probs_np])
            frame_labels.extend([float(y) for y in labels_np])

            for path, vid, y, p, pred in zip(batch["path"], batch["video_id"], labels_np, probs_np, preds_np):
                frame_rows.append(
                    {
                        "path": path,
                        "video_id": vid,
                        "label": int(y),
                        "prob": float(p),
                        "pred": int(pred),
                    }
                )
                if vid not in video_pool:
                    video_pool[vid] = []
                video_pool[vid].append((float(p), float(y)))

    frame_pred = [1.0 if p >= args.threshold else 0.0 for p in frame_probs]
    frame_acc = float(np.mean([int(p == y) for p, y in zip(frame_pred, frame_labels)])) if frame_labels else 0.0
    frame_auc, frame_ap = compute_auc_ap(frame_labels, frame_probs)

    video_rows: List[Dict] = []
    video_probs: List[float] = []
    video_labels: List[float] = []
    for vid, items in video_pool.items():
        p = float(np.mean([x[0] for x in items]))
        y = float(round(np.mean([x[1] for x in items])))
        pred = 1.0 if p >= args.threshold else 0.0
        video_rows.append(
            {
                "video_id": vid,
                "label": int(y),
                "prob": p,
                "pred": int(pred),
                "num_frames": len(items),
            }
        )
        video_probs.append(p)
        video_labels.append(y)

    video_pred = [1.0 if p >= args.threshold else 0.0 for p in video_probs]
    video_acc = float(np.mean([int(p == y) for p, y in zip(video_pred, video_labels)])) if video_labels else 0.0
    video_auc, video_ap = compute_auc_ap(video_labels, video_probs)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "video_id", "label", "prob", "pred"])
        writer.writeheader()
        writer.writerows(frame_rows)

    if args.video_output_csv:
        video_csv = Path(args.video_output_csv)
    else:
        video_csv = out_csv.with_name(out_csv.stem + "_video.csv")
    video_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(video_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "label", "prob", "pred", "num_frames"])
        writer.writeheader()
        writer.writerows(video_rows)

    metrics = {
        "frame_acc": frame_acc,
        "frame_auc": frame_auc,
        "frame_ap": frame_ap,
        "video_acc": video_acc,
        "video_auc": video_auc,
        "video_ap": video_ap,
        "num_frames": len(frame_rows),
        "num_videos": len(video_rows),
        "threshold": args.threshold,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[Saved] frame-level csv: {out_csv}")
    print(f"[Saved] video-level csv: {video_csv}")


if __name__ == "__main__":
    main()
