import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from config import get_default_config
from thesis_model import Enhanced_STF_Detector

_CFG = get_default_config()
_COMMON_CFG = _CFG["common"]
_DATA_CFG = _CFG["data"]

IMAGE_EXTS = {str(x).lower() for x in _COMMON_CFG.get("image_exts", [".jpg", ".jpeg", ".png", ".bmp", ".webp"])}
REAL_CLASS_FALLBACKS = [str(x) for x in _DATA_CFG.get("real_class_names", ["0_real", "real"])]
FAKE_CLASS_FALLBACKS = [str(x) for x in _DATA_CFG.get("fake_class_names", ["1_fake", "fake"])]


def _torch_load_compat(path: str, map_location: str = "cpu", weights_only: bool = False):
    kwargs = {"map_location": map_location, "weights_only": weights_only}
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(path, **kwargs)


def parse_args() -> argparse.Namespace:
    cfg = get_default_config()
    data_cfg = cfg["data"]
    test_cfg = cfg["test"]

    parser = argparse.ArgumentParser("Test Enhanced_STF_Detector")
    parser.add_argument("--data_root", type=str, default=data_cfg["test_root"], help="Test root dir with class folders.")
    parser.add_argument("--checkpoint", type=str, default=test_cfg["checkpoint"], help="Checkpoint path.")
    parser.add_argument("--output_csv", type=str, default=test_cfg["output_csv"])
    parser.add_argument("--video_output_csv", type=str, default=test_cfg["video_output_csv"], help="Optional video-level csv path.")
    parser.add_argument("--threshold", type=float, default=test_cfg["threshold"])

    parser.add_argument("--batch_size", type=int, default=test_cfg["batch_size"])
    parser.add_argument("--num_workers", type=int, default=test_cfg["num_workers"])  # kept for interface compatibility
    parser.add_argument("--image_size", type=int, default=data_cfg["image_size"])
    parser.add_argument("--motion_stride", type=int, default=data_cfg["motion_stride"])
    parser.add_argument("--max_frames_per_video", type=int, default=data_cfg["max_frames_per_video"])

    parser.add_argument("--real_dir_name", type=str, default=test_cfg["real_dir_name"])
    parser.add_argument("--fake_dir_name", type=str, default=test_cfg["fake_dir_name"])
    parser.add_argument("--cache_dir", type=str, default=test_cfg["cache_dir"])
    rgb_q_choices = [int(x) for x in test_cfg.get("rgb_compress_quality_choices", [0, 100, 95, 90, 85, 80, 75, 70])]
    parser.add_argument(
        "--rgb_compress_quality",
        type=int,
        default=int(test_cfg.get("rgb_compress_quality", 0)),
        choices=rgb_q_choices,
        help="RGB JPEG compression quality for test. 0 means no compression, 1~100 means compressed.",
    )

    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if bool_action is not None:
        parser.add_argument(
            "--balanced_1to1",
            action=bool_action,
            default=test_cfg["balanced_1to1"],
            help="Match each fake-model group to same-count real videos sequentially.",
        )
        parser.add_argument(
            "--cache_video_predictions",
            action=bool_action,
            default=test_cfg["cache_video_predictions"],
            help="Cache per-video frame probabilities to reuse repeated videos.",
        )
        parser.add_argument(
            "--refresh_cache",
            action=bool_action,
            default=test_cfg["refresh_cache"],
            help="Ignore old cache and recompute.",
        )
    else:
        parser.add_argument("--balanced_1to1", dest="balanced_1to1", action="store_true")
        parser.add_argument("--no-balanced_1to1", dest="balanced_1to1", action="store_false")
        parser.set_defaults(balanced_1to1=test_cfg["balanced_1to1"])

        parser.add_argument("--cache_video_predictions", dest="cache_video_predictions", action="store_true")
        parser.add_argument("--no-cache_video_predictions", dest="cache_video_predictions", action="store_false")
        parser.set_defaults(cache_video_predictions=test_cfg["cache_video_predictions"])

        parser.add_argument("--refresh_cache", dest="refresh_cache", action="store_true")
        parser.add_argument("--no-refresh_cache", dest="refresh_cache", action="store_false")
        parser.set_defaults(refresh_cache=test_cfg["refresh_cache"])

    # Optional overrides. If omitted, try loading from checkpoint config.
    parser.add_argument("--fusion_mode", type=str, default=test_cfg["fusion_mode"], choices=["", "independent", "cross_attention"])
    parser.add_argument("--feature_dim", type=int, default=test_cfg["feature_dim"])
    parser.add_argument("--hfri_mode", type=str, default=test_cfg["hfri_mode"], choices=["", "fft", "dct"])
    parser.add_argument("--flowformer_repo", type=str, default=test_cfg["flowformer_repo"])
    parser.add_argument("--flowformer_ckpt", type=str, default=test_cfg["flowformer_ckpt"])

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


def list_image_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    return files


def is_video_frame_dir(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return True
    return False


def collect_video_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    if not root.exists():
        return out
    for d in sorted(root.rglob("*")):
        if d.is_dir() and is_video_frame_dir(d):
            out.append(d)
    # If root itself is a video frame dir, include it as well.
    if is_video_frame_dir(root):
        out = [root] + out

    # Deduplicate while preserving order.
    dedup: List[Path] = []
    seen = set()
    for p in out:
        key = str(p.resolve()).lower()
        if key in seen:
            continue
        dedup.append(p)
        seen.add(key)
    return dedup


def resolve_class_dir(data_root: Path, primary_name: str, fallback_names: List[str]) -> Path:
    candidates = [primary_name] + [x for x in fallback_names if x != primary_name]
    for name in candidates:
        p = data_root / name
        if p.exists() and p.is_dir():
            return p
    raise RuntimeError(
        f"Cannot find class directory under {data_root}. Tried: {candidates}"
    )


def collect_fake_groups(fake_root: Path) -> List[Tuple[str, List[Path]]]:
    groups: List[Tuple[str, List[Path]]] = []

    # Prefer first-level folders as generator groups.
    first_level = [d for d in sorted(fake_root.iterdir()) if d.is_dir()]
    for d in first_level:
        vids = collect_video_dirs(d)
        if vids:
            groups.append((d.name, vids))

    # Fallback: fake_root directly contains video folders.
    if not groups:
        vids = collect_video_dirs(fake_root)
        if vids:
            groups.append((fake_root.name, vids))
    return groups


def build_balanced_eval_plan(
    data_root: Path,
    real_dir_name: str,
    fake_dir_name: str,
    balanced_1to1: bool,
) -> List[Dict]:
    real_root = resolve_class_dir(data_root, real_dir_name, REAL_CLASS_FALLBACKS)
    fake_root = resolve_class_dir(data_root, fake_dir_name, FAKE_CLASS_FALLBACKS)

    real_videos = collect_video_dirs(real_root)
    fake_groups = collect_fake_groups(fake_root)

    if not fake_groups:
        raise RuntimeError(f"No fake videos found under {fake_root}")
    if not real_videos:
        raise RuntimeError(f"No real videos found under {real_root}")

    plan: List[Dict] = []
    real_cursor = 0

    for group_name, fake_videos in fake_groups:
        n_fake = len(fake_videos)
        if n_fake == 0:
            continue

        # Always include fake videos.
        for i, video_dir in enumerate(fake_videos):
            plan.append(
                {
                    "group_model": group_name,
                    "role": "fake",
                    "label": 1,
                    "video_dir": video_dir,
                    "sample_id": f"{group_name}/fake/{i:06d}",
                }
            )

        # Match real videos 1:1 by sequential order if enabled.
        n_real = n_fake if balanced_1to1 else len(real_videos)
        for i in range(n_real):
            if balanced_1to1:
                real_video = real_videos[real_cursor % len(real_videos)]
                real_cursor += 1
            else:
                real_video = real_videos[i]
            plan.append(
                {
                    "group_model": group_name,
                    "role": "real",
                    "label": 0,
                    "video_dir": real_video,
                    "sample_id": f"{group_name}/real/{i:06d}",
                }
            )

    return plan


def build_cache_bucket(
    cache_root: Path,
    checkpoint: Path,
    model_args: Dict,
    image_size: int,
    motion_stride: int,
    max_frames_per_video: int,
    rgb_compress_quality: int,
) -> Path:
    ckpt_resolved = checkpoint.resolve()
    try:
        ckpt_mtime = ckpt_resolved.stat().st_mtime_ns
    except Exception:
        ckpt_mtime = 0

    signature_obj = {
        "checkpoint": str(ckpt_resolved),
        "checkpoint_mtime_ns": ckpt_mtime,
        "fusion_mode": model_args["fusion_mode"],
        "feature_dim": int(model_args["feature_dim"]),
        "hfri_mode": model_args["hfri_mode"],
        "flowformer_repo": model_args["flowformer_repo"],
        "flowformer_ckpt": model_args["flowformer_ckpt"],
        "image_size": int(image_size),
        "motion_stride": int(motion_stride),
        "max_frames_per_video": int(max_frames_per_video),
        "rgb_compress_quality": int(rgb_compress_quality),
    }
    sign_text = json.dumps(signature_obj, ensure_ascii=False, sort_keys=True)
    sign = hashlib.sha1(sign_text.encode("utf-8")).hexdigest()[:16]
    bucket = cache_root / sign
    bucket.mkdir(parents=True, exist_ok=True)
    return bucket


def get_video_cache_file(cache_bucket: Path, video_dir: Path) -> Path:
    key = str(video_dir.resolve()).lower()
    hid = hashlib.sha1(key.encode("utf-8")).hexdigest()
    sub = cache_bucket / hid[:2]
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{hid}.json"


def load_video_cache(cache_file: Path) -> Optional[Dict]:
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None
        frame_paths = obj.get("frame_paths", [])
        frame_probs = obj.get("frame_probs", [])
        if not isinstance(frame_paths, list) or not isinstance(frame_probs, list):
            return None
        if len(frame_paths) != len(frame_probs):
            return None
        return obj
    except Exception:
        return None


def save_video_cache(cache_file: Path, payload: Dict) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


class InferencePairTransform:
    def __init__(self, image_size: int, rgb_compress_quality: int = 0):
        self.image_size = int(image_size)
        self.rgb_compress_quality = int(rgb_compress_quality)

    @staticmethod
    def _apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
        q = max(1, min(100, int(quality)))
        with io.BytesIO() as buf:
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            out = Image.open(buf).convert("RGB")
            return out.copy()

    def __call__(self, img1: Image.Image, img2: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rgb_compress_quality > 0:
            img1 = self._apply_jpeg_compression(img1, self.rgb_compress_quality)
            img2 = self._apply_jpeg_compression(img2, self.rgb_compress_quality)
        img1 = TF.resize(img1, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        img2 = TF.resize(img2, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        return TF.to_tensor(img1), TF.to_tensor(img2)


def infer_one_video(
    model: torch.nn.Module,
    device: torch.device,
    video_dir: Path,
    image_size: int,
    motion_stride: int,
    max_frames_per_video: int,
    batch_size: int,
    rgb_compress_quality: int = 0,
) -> Dict:
    frames = list_image_files(video_dir)
    if max_frames_per_video > 0:
        frames = frames[: int(max_frames_per_video)]
    if not frames:
        return {
            "frame_paths": [],
            "frame_probs": [],
            "num_frames": 0,
            "video_prob": float("nan"),
        }

    transform = InferencePairTransform(image_size=image_size, rgb_compress_quality=rgb_compress_quality)
    n = len(frames)
    pairs: List[Tuple[Path, Path]] = []
    for i in range(n):
        j = 0 if n == 1 else min(n - 1, i + max(1, int(motion_stride)))
        pairs.append((frames[i], frames[j]))

    frame_paths: List[str] = []
    frame_probs: List[float] = []

    with torch.no_grad():
        for start in range(0, len(pairs), max(1, int(batch_size))):
            batch_pairs = pairs[start : start + max(1, int(batch_size))]
            x1_list: List[torch.Tensor] = []
            x2_list: List[torch.Tensor] = []
            p1_list: List[str] = []

            for p1, p2 in batch_pairs:
                img1 = Image.open(p1).convert("RGB")
                img2 = Image.open(p2).convert("RGB")
                x1, x2 = transform(img1, img2)
                x1_list.append(x1)
                x2_list.append(x2)
                p1_list.append(str(p1))

            x1_batch = torch.stack(x1_list, dim=0).to(device, non_blocking=True)  # [B,3,H,W]
            x2_batch = torch.stack(x2_list, dim=0).to(device, non_blocking=True)  # [B,3,H,W]

            outputs = model(x1_batch, x1_batch, x2_batch)
            probs = torch.sigmoid(outputs["logits"]).squeeze(1).detach().cpu().numpy().tolist()

            frame_paths.extend(p1_list)
            frame_probs.extend([float(p) for p in probs])

    video_prob = float(np.mean(frame_probs)) if frame_probs else float("nan")
    return {
        "frame_paths": frame_paths,
        "frame_probs": frame_probs,
        "num_frames": len(frame_probs),
        "video_prob": video_prob,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = _torch_load_compat(args.checkpoint, map_location="cpu", weights_only=False)
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
        f"hfri_mode={model_args['hfri_mode']} rgb_compress_quality={int(args.rgb_compress_quality)}"
    )

    data_root = Path(args.data_root)
    plan = build_balanced_eval_plan(
        data_root=data_root,
        real_dir_name=args.real_dir_name,
        fake_dir_name=args.fake_dir_name,
        balanced_1to1=bool(args.balanced_1to1),
    )
    if not plan:
        raise RuntimeError("No evaluation samples found.")

    n_fake = sum(1 for x in plan if x["label"] == 1)
    n_real = sum(1 for x in plan if x["label"] == 0)
    print(
        f"[Plan] samples={len(plan)} fake={n_fake} real={n_real} "
        f"balanced_1to1={bool(args.balanced_1to1)}"
    )

    cache_bucket = None
    if args.cache_video_predictions:
        cache_root = Path(args.cache_dir)
        cache_bucket = build_cache_bucket(
            cache_root=cache_root,
            checkpoint=Path(args.checkpoint),
            model_args=model_args,
            image_size=args.image_size,
            motion_stride=args.motion_stride,
            max_frames_per_video=args.max_frames_per_video,
            rgb_compress_quality=args.rgb_compress_quality,
        )
        print(f"[Cache] bucket: {cache_bucket}")

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.video_output_csv:
        video_csv = Path(args.video_output_csv)
    else:
        video_csv = out_csv.with_name(out_csv.stem + "_video.csv")
    video_csv.parent.mkdir(parents=True, exist_ok=True)

    frame_fieldnames = ["path", "video_id", "source_video", "group_model", "role", "label", "prob", "pred", "cache_hit"]
    video_fieldnames = ["video_id", "source_video", "group_model", "role", "label", "prob", "pred", "num_frames", "cache_hit"]

    model.eval()
    frame_probs_all: List[float] = []
    frame_labels_all: List[float] = []
    video_probs_all: List[float] = []
    video_labels_all: List[float] = []
    num_frames_written = 0
    num_videos_written = 0

    mem_cache: Dict[str, Dict] = {}
    cache_hit_mem = 0
    cache_hit_disk = 0
    cache_miss = 0
    skipped_empty = 0
    cache_group_flush_count = 0
    cache_group_flush_videos = 0

    current_group_for_cache: Optional[str] = None
    pending_group_cache_entries: Dict[str, Dict] = {}

    def flush_pending_group_cache(group_name: Optional[str]) -> None:
        nonlocal pending_group_cache_entries, cache_group_flush_count, cache_group_flush_videos
        if not args.cache_video_predictions:
            return
        if group_name is None:
            return
        if not pending_group_cache_entries:
            return
        assert cache_bucket is not None
        for entry in pending_group_cache_entries.values():
            cache_file = get_video_cache_file(cache_bucket, entry["video_dir"])
            save_video_cache(cache_file=cache_file, payload=entry["payload"])
        flushed = len(pending_group_cache_entries)
        cache_group_flush_count += 1
        cache_group_flush_videos += flushed
        print(f"[Cache] flushed group={group_name} videos={flushed}")
        pending_group_cache_entries = {}

    with open(out_csv, "w", newline="", encoding="utf-8") as frame_f, open(
        video_csv, "w", newline="", encoding="utf-8"
    ) as video_f:
        frame_writer = csv.DictWriter(frame_f, fieldnames=frame_fieldnames)
        video_writer = csv.DictWriter(video_f, fieldnames=video_fieldnames)
        frame_writer.writeheader()
        video_writer.writeheader()
        frame_f.flush()
        video_f.flush()

        for i, sample in enumerate(plan, start=1):
            video_dir: Path = sample["video_dir"]
            sample_id: str = sample["sample_id"]
            group_model: str = sample["group_model"]
            role: str = sample["role"]
            label: int = int(sample["label"])
            key = str(video_dir.resolve()).lower()

            pred_obj: Optional[Dict] = None
            cache_hit_flag = 0

            if args.cache_video_predictions:
                if current_group_for_cache is None:
                    current_group_for_cache = group_model
                elif group_model != current_group_for_cache:
                    # Delay disk writes until one generator-model group finishes.
                    flush_pending_group_cache(current_group_for_cache)
                    current_group_for_cache = group_model

            if args.cache_video_predictions:
                if key in mem_cache and not args.refresh_cache:
                    pred_obj = mem_cache[key]
                    cache_hit_mem += 1
                    cache_hit_flag = 1
                else:
                    assert cache_bucket is not None
                    cache_file = get_video_cache_file(cache_bucket, video_dir)
                    if (not args.refresh_cache) and cache_file.exists():
                        cached = load_video_cache(cache_file)
                        if cached is not None:
                            pred_obj = cached
                            mem_cache[key] = cached
                            cache_hit_disk += 1
                            cache_hit_flag = 1

            if pred_obj is None:
                cache_miss += 1
                pred_obj = infer_one_video(
                    model=model,
                    device=device,
                    video_dir=video_dir,
                    image_size=args.image_size,
                    motion_stride=args.motion_stride,
                    max_frames_per_video=args.max_frames_per_video,
                    batch_size=args.batch_size,
                    rgb_compress_quality=args.rgb_compress_quality,
                )
                mem_cache[key] = pred_obj

                if args.cache_video_predictions:
                    # Keep per-video cache in memory immediately.
                    # Write to disk later when current group_model is fully processed.
                    if key not in pending_group_cache_entries:
                        pending_group_cache_entries[key] = {
                            "video_dir": video_dir,
                            "payload": {
                                "video_dir": str(video_dir.resolve()),
                                "frame_paths": pred_obj["frame_paths"],
                                "frame_probs": pred_obj["frame_probs"],
                                "num_frames": int(pred_obj["num_frames"]),
                                "video_prob": float(pred_obj["video_prob"]),
                                "rgb_compress_quality": int(args.rgb_compress_quality),
                            },
                        }

            frame_paths = pred_obj.get("frame_paths", [])
            frame_probs = [float(x) for x in pred_obj.get("frame_probs", [])]
            num_frames = int(pred_obj.get("num_frames", len(frame_probs)))
            video_prob = float(pred_obj.get("video_prob", float("nan")))
            video_pred = int(video_prob >= args.threshold) if np.isfinite(video_prob) else 0
            source_video = str(video_dir.resolve())

            if num_frames <= 0:
                skipped_empty += 1
                print(f"[{i:04d}/{len(plan):04d}] {group_model} {role} skip (empty video): {source_video}")
                continue

            frame_batch_rows: List[Dict] = []
            for pth, prob in zip(frame_paths, frame_probs):
                pred = int(prob >= args.threshold)
                row = {
                    "path": pth,
                    "video_id": sample_id,
                    "source_video": source_video,
                    "group_model": group_model,
                    "role": role,
                    "label": label,
                    "prob": float(prob),
                    "pred": pred,
                    "cache_hit": cache_hit_flag,
                }
                frame_batch_rows.append(row)
                frame_probs_all.append(float(prob))
                frame_labels_all.append(float(label))

            video_row = {
                "video_id": sample_id,
                "source_video": source_video,
                "group_model": group_model,
                "role": role,
                "label": label,
                "prob": float(video_prob),
                "pred": int(video_pred),
                "num_frames": int(num_frames),
                "cache_hit": cache_hit_flag,
            }
            video_probs_all.append(float(video_prob))
            video_labels_all.append(float(label))

            # Save immediately after each processed video.
            frame_writer.writerows(frame_batch_rows)
            video_writer.writerow(video_row)
            frame_f.flush()
            video_f.flush()
            num_frames_written += len(frame_batch_rows)
            num_videos_written += 1

            print(
                f"[{i:04d}/{len(plan):04d}] {group_model} {role} "
                f"frames={num_frames} prob={video_prob:.4f} cache_hit={cache_hit_flag} saved=1"
            )

        # Flush final group cache after all its videos are done.
        flush_pending_group_cache(current_group_for_cache)

    frame_pred = [1.0 if p >= args.threshold else 0.0 for p in frame_probs_all]
    frame_acc = float(np.mean([int(p == y) for p, y in zip(frame_pred, frame_labels_all)])) if frame_labels_all else 0.0
    frame_auc, frame_ap = compute_auc_ap(frame_labels_all, frame_probs_all)

    video_pred = [1.0 if p >= args.threshold else 0.0 for p in video_probs_all]
    video_acc = float(np.mean([int(p == y) for p, y in zip(video_pred, video_labels_all)])) if video_labels_all else 0.0
    video_auc, video_ap = compute_auc_ap(video_labels_all, video_probs_all)

    metrics = {
        "frame_acc": frame_acc,
        "frame_auc": frame_auc,
        "frame_ap": frame_ap,
        "video_acc": video_acc,
        "video_auc": video_auc,
        "video_ap": video_ap,
        "num_frames": int(num_frames_written),
        "num_videos": int(num_videos_written),
        "num_fake": int(n_fake),
        "num_real": int(n_real),
        "threshold": args.threshold,
        "rgb_compress_quality": int(args.rgb_compress_quality),
        "cache_hit_mem": int(cache_hit_mem),
        "cache_hit_disk": int(cache_hit_disk),
        "cache_miss": int(cache_miss),
        "cache_group_flush_count": int(cache_group_flush_count),
        "cache_group_flush_videos": int(cache_group_flush_videos),
        "skipped_empty": int(skipped_empty),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[Saved] frame-level csv: {out_csv}")
    print(f"[Saved] video-level csv: {video_csv}")


if __name__ == "__main__":
    main()
