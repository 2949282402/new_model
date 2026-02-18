import argparse
import csv
import hashlib
import io
import json
from collections import OrderedDict
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


def resolve_abs(path_text: str) -> Path:
    p = Path(path_text)
    if not p.is_absolute():
        p = (Path(__file__).resolve().parent / p).resolve()
    return p


def normalize_motion_offsets(motion_stride: int, motion_multi_offsets: Optional[List[int]]) -> List[int]:
    offsets: List[int] = []
    if motion_multi_offsets:
        for x in motion_multi_offsets:
            try:
                v = int(x)
            except Exception:
                continue
            if v > 0:
                offsets.append(v)
    if not offsets:
        offsets = [max(1, int(motion_stride))]

    dedup: List[int] = []
    seen = set()
    for v in offsets:
        if v in seen:
            continue
        seen.add(v)
        dedup.append(v)
    return dedup


def normalize_two_weights(weight_rgb: float, weight_flow: float) -> Tuple[float, float]:
    w_rgb = max(0.0, float(weight_rgb))
    w_flow = max(0.0, float(weight_flow))
    s = w_rgb + w_flow
    if s <= 0:
        return 0.5, 0.5
    return w_rgb / s, w_flow / s


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
    parser.add_argument(
        "--video_agg_method",
        type=str,
        default=test_cfg.get("video_agg_method", "topk_mean"),
        choices=["mean", "topk_mean", "hybrid_topk_mean"],
        help="Video-level aggregation method for frame probabilities.",
    )
    parser.add_argument("--video_agg_topk_ratio", type=float, default=float(test_cfg.get("video_agg_topk_ratio", 0.1)))
    parser.add_argument("--video_agg_topk_min", type=int, default=int(test_cfg.get("video_agg_topk_min", 3)))
    parser.add_argument("--video_agg_topk_max", type=int, default=int(test_cfg.get("video_agg_topk_max", 32)))
    parser.add_argument("--video_agg_hybrid_alpha", type=float, default=float(test_cfg.get("video_agg_hybrid_alpha", 0.7)))
    parser.add_argument(
        "--rgb_video_agg_method",
        type=str,
        default=str(test_cfg.get("rgb_video_agg_method", test_cfg.get("video_agg_method", "topk_mean"))),
        choices=["mean", "topk_mean", "hybrid_topk_mean"],
        help="Independent mode RGB branch video aggregation method.",
    )
    parser.add_argument(
        "--rgb_video_agg_topk_ratio",
        type=float,
        default=float(test_cfg.get("rgb_video_agg_topk_ratio", test_cfg.get("video_agg_topk_ratio", 0.1))),
    )
    parser.add_argument(
        "--rgb_video_agg_topk_min",
        type=int,
        default=int(test_cfg.get("rgb_video_agg_topk_min", test_cfg.get("video_agg_topk_min", 3))),
    )
    parser.add_argument(
        "--rgb_video_agg_topk_max",
        type=int,
        default=int(test_cfg.get("rgb_video_agg_topk_max", test_cfg.get("video_agg_topk_max", 32))),
    )
    parser.add_argument(
        "--rgb_video_agg_hybrid_alpha",
        type=float,
        default=float(test_cfg.get("rgb_video_agg_hybrid_alpha", test_cfg.get("video_agg_hybrid_alpha", 0.7))),
    )
    parser.add_argument(
        "--flow_video_agg_method",
        type=str,
        default=str(test_cfg.get("flow_video_agg_method", test_cfg.get("video_agg_method", "topk_mean"))),
        choices=["mean", "topk_mean", "hybrid_topk_mean"],
        help="Independent mode Flow branch video aggregation method.",
    )
    parser.add_argument(
        "--flow_video_agg_topk_ratio",
        type=float,
        default=float(test_cfg.get("flow_video_agg_topk_ratio", test_cfg.get("video_agg_topk_ratio", 0.1))),
    )
    parser.add_argument(
        "--flow_video_agg_topk_min",
        type=int,
        default=int(test_cfg.get("flow_video_agg_topk_min", test_cfg.get("video_agg_topk_min", 3))),
    )
    parser.add_argument(
        "--flow_video_agg_topk_max",
        type=int,
        default=int(test_cfg.get("flow_video_agg_topk_max", test_cfg.get("video_agg_topk_max", 32))),
    )
    parser.add_argument(
        "--flow_video_agg_hybrid_alpha",
        type=float,
        default=float(test_cfg.get("flow_video_agg_hybrid_alpha", test_cfg.get("video_agg_hybrid_alpha", 0.7))),
    )
    parser.add_argument("--branch_fuse_weight_rgb", type=float, default=float(test_cfg.get("branch_fuse_weight_rgb", 0.5)))
    parser.add_argument("--branch_fuse_weight_flow", type=float, default=float(test_cfg.get("branch_fuse_weight_flow", 0.5)))

    parser.add_argument("--batch_size", type=int, default=test_cfg["batch_size"])
    parser.add_argument("--num_workers", type=int, default=test_cfg["num_workers"])  # kept for interface compatibility
    parser.add_argument("--image_size", type=int, default=data_cfg["image_size"])
    parser.add_argument("--motion_stride", type=int, default=data_cfg["motion_stride"])
    parser.add_argument(
        "--motion_multi_offsets",
        type=int,
        nargs="+",
        default=[int(x) for x in data_cfg.get("motion_multi_offsets", [1, 2, 4])],
        help="Motion multi-interval offsets, e.g. 1 2 4.",
    )
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
            "--independent_branch_agg",
            action=bool_action,
            default=bool(test_cfg.get("independent_branch_agg", False)),
            help="In independent mode, aggregate RGB/Flow branches separately and then fuse.",
        )
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
        parser.add_argument("--independent_branch_agg", dest="independent_branch_agg", action="store_true")
        parser.add_argument("--no-independent_branch_agg", dest="independent_branch_agg", action="store_false")
        parser.set_defaults(independent_branch_agg=bool(test_cfg.get("independent_branch_agg", False)))

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
    parser.add_argument("--motion_token_cache_dir", type=str, default=test_cfg.get("motion_token_cache_dir", "./data/exp/_shared/flowformer_token_cache"))
    parser.add_argument("--motion_token_cache_dtype", type=str, default=test_cfg.get("motion_token_cache_dtype", "float16"), choices=["float16", "float32"])
    parser.add_argument("--motion_token_cache_mem_videos", type=int, default=int(test_cfg.get("motion_token_cache_mem_videos", 128)))
    parser.add_argument("--motion_token_cache_batch_size", type=int, default=int(test_cfg.get("motion_token_cache_batch_size", 16)))
    parser.add_argument("--motion_token_cache_print_freq", type=int, default=int(test_cfg.get("motion_token_cache_print_freq", 20)))

    if bool_action is not None:
        parser.add_argument(
            "--use_motion_token_cache",
            action=bool_action,
            default=bool(test_cfg.get("use_motion_token_cache", True)),
            help="Use shared FlowFormer token cache in test inference.",
        )
        parser.add_argument(
            "--motion_token_cache_refresh",
            action=bool_action,
            default=bool(test_cfg.get("motion_token_cache_refresh", False)),
            help="Ignore token cache and recompute.",
        )
    else:
        parser.add_argument("--use_motion_token_cache", dest="use_motion_token_cache", action="store_true")
        parser.add_argument("--no-use_motion_token_cache", dest="use_motion_token_cache", action="store_false")
        parser.set_defaults(use_motion_token_cache=bool(test_cfg.get("use_motion_token_cache", True)))

        parser.add_argument("--motion_token_cache_refresh", dest="motion_token_cache_refresh", action="store_true")
        parser.add_argument("--no-motion_token_cache_refresh", dest="motion_token_cache_refresh", action="store_false")
        parser.set_defaults(motion_token_cache_refresh=bool(test_cfg.get("motion_token_cache_refresh", False)))

    return parser.parse_args()


def safe_float(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _calc_topk_k(num_frames: int, ratio: float, k_min: int, k_max: int) -> int:
    n = max(0, int(num_frames))
    if n <= 0:
        return 0
    r = float(ratio)
    if not np.isfinite(r):
        r = 0.1
    r = min(max(r, 0.0), 1.0)
    k = int(np.ceil(n * r))
    k = max(int(k_min), k)
    if int(k_max) > 0:
        k = min(int(k_max), k)
    k = min(n, max(1, k))
    return int(k)


def aggregate_video_probability(
    frame_probs: List[float],
    method: str,
    topk_ratio: float,
    topk_min: int,
    topk_max: int,
    hybrid_alpha: float,
) -> float:
    if not frame_probs:
        return float("nan")
    arr = np.asarray(frame_probs, dtype=np.float32)
    if arr.size == 0:
        return float("nan")

    all_mean = float(np.mean(arr))
    agg_method = str(method).strip().lower()
    if agg_method == "mean":
        return all_mean

    k = _calc_topk_k(
        num_frames=int(arr.size),
        ratio=float(topk_ratio),
        k_min=int(topk_min),
        k_max=int(topk_max),
    )
    if k <= 0:
        return all_mean
    topk_vals = np.sort(arr)[-k:]
    topk_mean = float(np.mean(topk_vals))
    if agg_method == "topk_mean":
        return topk_mean

    alpha = float(hybrid_alpha)
    if not np.isfinite(alpha):
        alpha = 0.7
    alpha = min(max(alpha, 0.0), 1.0)
    return float(alpha * topk_mean + (1.0 - alpha) * all_mean)


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
    motion_multi_offsets: Optional[List[int]],
    max_frames_per_video: int,
    rgb_compress_quality: int,
) -> Path:
    ckpt_resolved = checkpoint.resolve()
    try:
        ckpt_mtime = ckpt_resolved.stat().st_mtime_ns
    except Exception:
        ckpt_mtime = 0

    signature_obj = {
        "cache_schema": "video_cache_v2_with_branch_probs",
        "checkpoint": str(ckpt_resolved),
        "checkpoint_mtime_ns": ckpt_mtime,
        "fusion_mode": model_args["fusion_mode"],
        "feature_dim": int(model_args["feature_dim"]),
        "hfri_mode": model_args["hfri_mode"],
        "flowformer_repo": model_args["flowformer_repo"],
        "flowformer_ckpt": model_args["flowformer_ckpt"],
        "image_size": int(image_size),
        "motion_stride": int(motion_stride),
        "motion_multi_offsets": [int(x) for x in normalize_motion_offsets(motion_stride, motion_multi_offsets)],
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


class MotionTokenCacheManager:
    def __init__(
        self,
        motion_encoder: torch.nn.Module,
        device: torch.device,
        cache_dir: str,
        image_size: int,
        motion_stride: int,
        motion_multi_offsets: Optional[List[int]],
        max_frames_per_video: int,
        rgb_compress_quality: int,
        flowformer_repo: str,
        flowformer_ckpt: str,
        save_dtype: str = "float16",
        refresh_cache: bool = False,
        mem_videos: int = 128,
        compute_batch_size: int = 16,
        print_freq: int = 20,
    ):
        self.motion_encoder = motion_encoder
        self.device = device
        self.cache_root = resolve_abs(cache_dir)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self.image_size = int(image_size)
        self.motion_stride = max(1, int(motion_stride))
        self.motion_offsets = normalize_motion_offsets(
            motion_stride=self.motion_stride,
            motion_multi_offsets=motion_multi_offsets,
        )
        self.max_frames_per_video = max(0, int(max_frames_per_video))
        self.rgb_compress_quality = int(rgb_compress_quality)
        self.flowformer_repo = str(resolve_abs(flowformer_repo))
        self.flowformer_ckpt = str(resolve_abs(flowformer_ckpt))
        self.save_dtype_name = str(save_dtype).lower()
        self.save_dtype = torch.float16 if self.save_dtype_name == "float16" else torch.float32
        self.refresh_cache = bool(refresh_cache)
        self.mem_videos = max(0, int(mem_videos))
        self.compute_batch_size = max(1, int(compute_batch_size))
        self.print_freq = max(1, int(print_freq))

        self.token_dim = int(getattr(motion_encoder.align, "in_features", 256))
        self.cache_bucket = self._build_cache_bucket()
        self.transform = InferencePairTransform(
            image_size=self.image_size,
            rgb_compress_quality=self.rgb_compress_quality,
        )

        self.mem_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.hit_mem = 0
        self.hit_disk = 0
        self.miss_compute = 0
        self.compute_calls = 0

    def _build_cache_bucket(self) -> Path:
        ckpt_path = Path(self.flowformer_ckpt)
        try:
            ckpt_mtime = ckpt_path.stat().st_mtime_ns
        except Exception:
            ckpt_mtime = 0

        signature_obj = {
            "script": "cache_flowformer_tokens_v1",
            "flowformer_repo": self.flowformer_repo,
            "flowformer_ckpt": self.flowformer_ckpt,
            "flowformer_ckpt_mtime_ns": int(ckpt_mtime),
            "image_size": int(self.image_size),
            "motion_stride": int(self.motion_stride),
            "motion_multi_offsets": [int(x) for x in self.motion_offsets],
            "max_frames_per_video": int(self.max_frames_per_video),
            "rgb_compress_quality": int(self.rgb_compress_quality),
            "token_dim": int(self.token_dim),
            "save_dtype": self.save_dtype_name,
        }
        sign_text = json.dumps(signature_obj, ensure_ascii=False, sort_keys=True)
        sign = hashlib.sha1(sign_text.encode("utf-8")).hexdigest()[:16]
        bucket = self.cache_root / sign
        bucket.mkdir(parents=True, exist_ok=True)
        with open(bucket / "signature.json", "w", encoding="utf-8") as f:
            json.dump(signature_obj, f, ensure_ascii=False, indent=2)
        return bucket

    @staticmethod
    def _video_key(video_dir: Path) -> str:
        return str(video_dir.resolve()).lower()

    def _cache_file(self, video_dir: Path) -> Path:
        key = self._video_key(video_dir)
        hid = hashlib.sha1(key.encode("utf-8")).hexdigest()
        sub = self.cache_bucket / hid[:2]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{hid}.pt"

    def _mem_get(self, key: str) -> Optional[torch.Tensor]:
        if key not in self.mem_cache:
            return None
        tokens = self.mem_cache.pop(key)
        self.mem_cache[key] = tokens
        return tokens

    def _mem_set(self, key: str, tokens: torch.Tensor) -> None:
        if self.mem_videos <= 0:
            return
        if key in self.mem_cache:
            self.mem_cache.pop(key)
        self.mem_cache[key] = tokens
        while len(self.mem_cache) > self.mem_videos:
            self.mem_cache.popitem(last=False)

    def _load_video_tokens(self, cache_file: Path) -> Optional[torch.Tensor]:
        if not cache_file.exists():
            return None
        try:
            obj = _torch_load_compat(str(cache_file), map_location="cpu", weights_only=False)
            if not isinstance(obj, dict):
                return None
            tokens = obj.get("motion_tokens", None)
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
                return None
            return tokens.to(dtype=self.save_dtype, device="cpu")
        except Exception:
            return None

    def _save_video_tokens(
        self,
        cache_file: Path,
        video_dir: Path,
        frames: List[Path],
        pairs: List[Tuple[int, int]],
        tokens: torch.Tensor,
    ) -> None:
        payload = {
            "video_dir": str(video_dir.resolve()),
            "num_frames": int(len(frames)),
            "num_pairs": int(len(pairs)),
            "motion_stride": int(self.motion_stride),
            "motion_multi_offsets": [int(x) for x in self.motion_offsets],
            "image_size": int(self.image_size),
            "rgb_compress_quality": int(self.rgb_compress_quality),
            "frame_paths": [str(p.resolve()) for p in frames],
            "pair_indices": [(int(i), int(j)) for i, j in pairs],
            "token_dim": int(tokens.shape[1]) if tokens.ndim == 2 else int(self.token_dim),
            "save_dtype": self.save_dtype_name,
            "motion_tokens": tokens,
        }
        tmp_file = cache_file.with_suffix(".tmp")
        torch.save(payload, str(tmp_file))
        tmp_file.replace(cache_file)

    def _compute_video_tokens(self, video_dir: Path) -> torch.Tensor:
        frames = list_image_files(video_dir)
        if self.max_frames_per_video > 0:
            frames = frames[: self.max_frames_per_video]
        if not frames:
            return torch.empty((0, int(self.token_dim)), dtype=self.save_dtype)

        n = len(frames)
        pair_entries: List[Tuple[int, int, int]] = []
        for i in range(n):
            for off in self.motion_offsets:
                j = 0 if n == 1 else min(n - 1, i + int(off))
                pair_entries.append((i, j, int(off)))
        pairs = [(int(i), int(j)) for i, j, _ in pair_entries]

        token_sum = torch.zeros((n, int(self.token_dim)), dtype=torch.float32)
        token_count = torch.zeros((n, 1), dtype=torch.float32)
        was_training = bool(self.motion_encoder.training)
        self.motion_encoder.eval()
        try:
            for start in range(0, len(pair_entries), self.compute_batch_size):
                batch_pairs = pair_entries[start : start + self.compute_batch_size]
                x1_list: List[torch.Tensor] = []
                x2_list: List[torch.Tensor] = []
                for i, j, _ in batch_pairs:
                    with Image.open(frames[i]) as img1_raw, Image.open(frames[j]) as img2_raw:
                        x1, x2 = self.transform(img1_raw.convert("RGB"), img2_raw.convert("RGB"))
                    x1_list.append(x1)
                    x2_list.append(x2)
                x1_batch = torch.stack(x1_list, dim=0).to(self.device, non_blocking=True)
                x2_batch = torch.stack(x2_list, dim=0).to(self.device, non_blocking=True)
                with torch.no_grad():
                    tokens = self.motion_encoder.extract_motion_tokens(x1_batch, x2_batch)
                tokens_cpu = tokens.detach().cpu().to(dtype=torch.float32)
                for local_idx, (i, _, _) in enumerate(batch_pairs):
                    token_sum[int(i)] += tokens_cpu[local_idx]
                    token_count[int(i), 0] += 1.0
        finally:
            self.motion_encoder.train(was_training)

        token_count = token_count.clamp_min(1.0)
        video_tokens = (token_sum / token_count).to(dtype=self.save_dtype)

        cache_file = self._cache_file(video_dir)
        self._save_video_tokens(
            cache_file=cache_file,
            video_dir=video_dir,
            frames=frames,
            pairs=pairs,
            tokens=video_tokens,
        )
        self.miss_compute += 1
        self.compute_calls += 1
        if self.compute_calls % self.print_freq == 0:
            print(
                f"[MotionTokenCache] computed={self.miss_compute} hit_mem={self.hit_mem} "
                f"hit_disk={self.hit_disk} bucket={self.cache_bucket}"
            )
        return video_tokens

    def ensure_video_tokens(self, video_dir: Path) -> torch.Tensor:
        key = self._video_key(video_dir)

        tokens = self._mem_get(key)
        if tokens is not None:
            self.hit_mem += 1
            return tokens

        cache_file = self._cache_file(video_dir)
        if cache_file.exists() and (not self.refresh_cache):
            tokens = self._load_video_tokens(cache_file)
            if tokens is not None:
                self.hit_disk += 1
                self._mem_set(key, tokens)
                return tokens

        tokens = self._compute_video_tokens(video_dir)
        self._mem_set(key, tokens)
        return tokens


def infer_one_video(
    model: torch.nn.Module,
    device: torch.device,
    video_dir: Path,
    image_size: int,
    motion_stride: int,
    motion_multi_offsets: Optional[List[int]],
    max_frames_per_video: int,
    batch_size: int,
    rgb_compress_quality: int = 0,
    video_agg_method: str = "topk_mean",
    video_agg_topk_ratio: float = 0.1,
    video_agg_topk_min: int = 3,
    video_agg_topk_max: int = 32,
    video_agg_hybrid_alpha: float = 0.7,
    independent_branch_agg: bool = False,
    rgb_video_agg_method: str = "topk_mean",
    rgb_video_agg_topk_ratio: float = 0.1,
    rgb_video_agg_topk_min: int = 3,
    rgb_video_agg_topk_max: int = 32,
    rgb_video_agg_hybrid_alpha: float = 0.7,
    flow_video_agg_method: str = "topk_mean",
    flow_video_agg_topk_ratio: float = 0.1,
    flow_video_agg_topk_min: int = 3,
    flow_video_agg_topk_max: int = 32,
    flow_video_agg_hybrid_alpha: float = 0.7,
    branch_fuse_weight_rgb: float = 0.5,
    branch_fuse_weight_flow: float = 0.5,
    motion_token_cache: Optional[MotionTokenCacheManager] = None,
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
    center_indices = list(range(n))
    motion_offsets = normalize_motion_offsets(motion_stride=motion_stride, motion_multi_offsets=motion_multi_offsets)

    video_tokens: Optional[torch.Tensor] = None
    if motion_token_cache is not None:
        video_tokens = motion_token_cache.ensure_video_tokens(video_dir)

    frame_paths: List[str] = []
    frame_probs: List[float] = []
    frame_probs_rgb: List[float] = []
    frame_probs_flow: List[float] = []
    w_rgb, w_flow = normalize_two_weights(branch_fuse_weight_rgb, branch_fuse_weight_flow)
    branch_agg_enabled = bool(independent_branch_agg)

    with torch.no_grad():
        for start in range(0, len(center_indices), max(1, int(batch_size))):
            idx_list = center_indices[start : start + max(1, int(batch_size))]
            x1_list: List[torch.Tensor] = []
            p1_list: List[str] = []

            for i in idx_list:
                p1 = frames[i]
                with Image.open(p1) as img1_raw:
                    img1 = img1_raw.convert("RGB")
                x1, _ = transform(img1, img1)
                x1_list.append(x1)
                p1_list.append(str(p1))

            x1_batch = torch.stack(x1_list, dim=0).to(device, non_blocking=True)  # [B,3,H,W]
            if motion_token_cache is not None:
                if video_tokens is None or video_tokens.ndim != 2 or video_tokens.size(0) <= 0:
                    token_dim = int(getattr(motion_token_cache, "token_dim", 256))
                    motion_tokens_batch = torch.zeros((len(idx_list), token_dim), dtype=torch.float32, device=device)
                else:
                    safe_indices = [min(max(int(i), 0), int(video_tokens.size(0) - 1)) for i in idx_list]
                    motion_tokens_batch = video_tokens[safe_indices].to(device, non_blocking=True)
                outputs = model(x1_batch, motion_tokens=motion_tokens_batch)
            else:
                if len(motion_offsets) == 1:
                    off = int(motion_offsets[0])
                    x2_list: List[torch.Tensor] = []
                    for i in idx_list:
                        j = 0 if n == 1 else min(n - 1, i + off)
                        with Image.open(frames[j]) as img2_raw:
                            img2 = img2_raw.convert("RGB")
                        _, x2 = transform(img2, img2)
                        x2_list.append(x2)
                    x2_batch = torch.stack(x2_list, dim=0).to(device, non_blocking=True)
                    outputs = model(x1_batch, x1_batch, x2_batch)
                else:
                    token_sum = None
                    for off in motion_offsets:
                        x2_list: List[torch.Tensor] = []
                        for i in idx_list:
                            j = 0 if n == 1 else min(n - 1, i + int(off))
                            with Image.open(frames[j]) as img2_raw:
                                img2 = img2_raw.convert("RGB")
                            _, x2 = transform(img2, img2)
                            x2_list.append(x2)
                        x2_batch = torch.stack(x2_list, dim=0).to(device, non_blocking=True)
                        tokens_k = model.motion_encoder.extract_motion_tokens(x1_batch, x2_batch)
                        token_sum = tokens_k if token_sum is None else (token_sum + tokens_k)
                    motion_tokens_batch = token_sum / float(len(motion_offsets))
                    outputs = model(x1_batch, motion_tokens=motion_tokens_batch)
            probs = torch.sigmoid(outputs["logits"]).squeeze(1).detach().cpu().numpy().tolist()
            if "aux_logits" in outputs:
                logit_rgb, logit_flow = outputs["aux_logits"]
                probs_rgb = torch.sigmoid(logit_rgb).squeeze(1).detach().cpu().numpy().tolist()
                probs_flow = torch.sigmoid(logit_flow).squeeze(1).detach().cpu().numpy().tolist()
            else:
                probs_rgb = []
                probs_flow = []

            frame_paths.extend(p1_list)
            frame_probs.extend([float(p) for p in probs])
            if probs_rgb and probs_flow:
                frame_probs_rgb.extend([float(p) for p in probs_rgb])
                frame_probs_flow.extend([float(p) for p in probs_flow])

    use_branch_agg = (
        branch_agg_enabled
        and (len(frame_probs_rgb) == len(frame_probs))
        and (len(frame_probs_flow) == len(frame_probs))
        and (len(frame_probs) > 0)
    )
    video_prob_rgb = float("nan")
    video_prob_flow = float("nan")
    if use_branch_agg:
        video_prob_rgb = aggregate_video_probability(
            frame_probs=frame_probs_rgb,
            method=rgb_video_agg_method,
            topk_ratio=rgb_video_agg_topk_ratio,
            topk_min=rgb_video_agg_topk_min,
            topk_max=rgb_video_agg_topk_max,
            hybrid_alpha=rgb_video_agg_hybrid_alpha,
        )
        video_prob_flow = aggregate_video_probability(
            frame_probs=frame_probs_flow,
            method=flow_video_agg_method,
            topk_ratio=flow_video_agg_topk_ratio,
            topk_min=flow_video_agg_topk_min,
            topk_max=flow_video_agg_topk_max,
            hybrid_alpha=flow_video_agg_hybrid_alpha,
        )
        video_prob = float(w_rgb * video_prob_rgb + w_flow * video_prob_flow)
    else:
        video_prob = aggregate_video_probability(
            frame_probs=frame_probs,
            method=video_agg_method,
            topk_ratio=video_agg_topk_ratio,
            topk_min=video_agg_topk_min,
            topk_max=video_agg_topk_max,
            hybrid_alpha=video_agg_hybrid_alpha,
        )
    return {
        "frame_paths": frame_paths,
        "frame_probs": frame_probs,
        "frame_probs_rgb": frame_probs_rgb,
        "frame_probs_flow": frame_probs_flow,
        "num_frames": len(frame_probs),
        "video_prob": video_prob,
        "video_prob_rgb": video_prob_rgb,
        "video_prob_flow": video_prob_flow,
        "use_branch_agg": bool(use_branch_agg),
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
        f"hfri_mode={model_args['hfri_mode']} rgb_compress_quality={int(args.rgb_compress_quality)} "
        f"video_agg={args.video_agg_method}(ratio={float(args.video_agg_topk_ratio):.3f},"
        f"min={int(args.video_agg_topk_min)},max={int(args.video_agg_topk_max)},"
        f"alpha={float(args.video_agg_hybrid_alpha):.2f})"
    )
    if bool(args.independent_branch_agg):
        w_rgb, w_flow = normalize_two_weights(args.branch_fuse_weight_rgb, args.branch_fuse_weight_flow)
        print(
            f"[BranchAgg] rgb={args.rgb_video_agg_method}(ratio={float(args.rgb_video_agg_topk_ratio):.3f},"
            f"min={int(args.rgb_video_agg_topk_min)},max={int(args.rgb_video_agg_topk_max)},"
            f"alpha={float(args.rgb_video_agg_hybrid_alpha):.2f}) "
            f"flow={args.flow_video_agg_method}(ratio={float(args.flow_video_agg_topk_ratio):.3f},"
            f"min={int(args.flow_video_agg_topk_min)},max={int(args.flow_video_agg_topk_max)},"
            f"alpha={float(args.flow_video_agg_hybrid_alpha):.2f}) "
            f"fuse_w=({w_rgb:.3f},{w_flow:.3f})"
        )

    motion_token_cache: Optional[MotionTokenCacheManager] = None
    if bool(args.use_motion_token_cache):
        motion_token_cache = MotionTokenCacheManager(
            motion_encoder=model.motion_encoder,
            device=device,
            cache_dir=args.motion_token_cache_dir,
            image_size=args.image_size,
            motion_stride=args.motion_stride,
            motion_multi_offsets=args.motion_multi_offsets,
            max_frames_per_video=args.max_frames_per_video,
            rgb_compress_quality=args.rgb_compress_quality,
            flowformer_repo=model_args["flowformer_repo"],
            flowformer_ckpt=model_args["flowformer_ckpt"],
            save_dtype=args.motion_token_cache_dtype,
            refresh_cache=bool(args.motion_token_cache_refresh),
            mem_videos=args.motion_token_cache_mem_videos,
            compute_batch_size=args.motion_token_cache_batch_size,
            print_freq=args.motion_token_cache_print_freq,
        )
        print(
            f"[MotionTokenCache] enabled bucket={motion_token_cache.cache_bucket} "
            f"dtype={args.motion_token_cache_dtype} refresh={bool(args.motion_token_cache_refresh)} "
            f"offsets={motion_token_cache.motion_offsets}"
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
            motion_multi_offsets=args.motion_multi_offsets,
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
    real_repeat_total = 0
    real_repeat_cache_hit = 0

    current_group_for_cache: Optional[str] = None
    pending_group_cache_entries: Dict[str, Dict] = {}
    seen_real_keys: set = set()

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
            is_repeated_real = (role == "real") and (key in seen_real_keys)
            if is_repeated_real:
                real_repeat_total += 1

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
                # Always prioritize in-memory reuse for repeated videos.
                if key in mem_cache:
                    pred_obj = mem_cache[key]
                    cache_hit_mem += 1
                    cache_hit_flag = 1
                else:
                    assert cache_bucket is not None
                    cache_file = get_video_cache_file(cache_bucket, video_dir)
                    # Even when refresh_cache=True, repeated real videos must reuse cache.
                    allow_disk_cache = (not args.refresh_cache) or is_repeated_real
                    if allow_disk_cache and cache_file.exists():
                        cached = load_video_cache(cache_file)
                        if cached is not None:
                            pred_obj = cached
                            mem_cache[key] = cached
                            cache_hit_disk += 1
                            cache_hit_flag = 1

            if is_repeated_real and pred_obj is None:
                raise RuntimeError(
                    f"[CachePolicy] Repeated real video must use cache, but cache miss occurred: {video_dir}"
                )

            if pred_obj is None:
                cache_miss += 1
                pred_obj = infer_one_video(
                    model=model,
                    device=device,
                    video_dir=video_dir,
                    image_size=args.image_size,
                    motion_stride=args.motion_stride,
                    motion_multi_offsets=args.motion_multi_offsets,
                    max_frames_per_video=args.max_frames_per_video,
                    batch_size=args.batch_size,
                    rgb_compress_quality=args.rgb_compress_quality,
                    video_agg_method=args.video_agg_method,
                    video_agg_topk_ratio=args.video_agg_topk_ratio,
                    video_agg_topk_min=args.video_agg_topk_min,
                    video_agg_topk_max=args.video_agg_topk_max,
                    video_agg_hybrid_alpha=args.video_agg_hybrid_alpha,
                    independent_branch_agg=bool(args.independent_branch_agg),
                    rgb_video_agg_method=args.rgb_video_agg_method,
                    rgb_video_agg_topk_ratio=args.rgb_video_agg_topk_ratio,
                    rgb_video_agg_topk_min=args.rgb_video_agg_topk_min,
                    rgb_video_agg_topk_max=args.rgb_video_agg_topk_max,
                    rgb_video_agg_hybrid_alpha=args.rgb_video_agg_hybrid_alpha,
                    flow_video_agg_method=args.flow_video_agg_method,
                    flow_video_agg_topk_ratio=args.flow_video_agg_topk_ratio,
                    flow_video_agg_topk_min=args.flow_video_agg_topk_min,
                    flow_video_agg_topk_max=args.flow_video_agg_topk_max,
                    flow_video_agg_hybrid_alpha=args.flow_video_agg_hybrid_alpha,
                    branch_fuse_weight_rgb=args.branch_fuse_weight_rgb,
                    branch_fuse_weight_flow=args.branch_fuse_weight_flow,
                    motion_token_cache=motion_token_cache,
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
                                "frame_probs_rgb": pred_obj.get("frame_probs_rgb", []),
                                "frame_probs_flow": pred_obj.get("frame_probs_flow", []),
                                "num_frames": int(pred_obj["num_frames"]),
                                "video_prob": float(pred_obj["video_prob"]),
                                "video_prob_rgb": float(pred_obj.get("video_prob_rgb", float("nan"))),
                                "video_prob_flow": float(pred_obj.get("video_prob_flow", float("nan"))),
                                "use_branch_agg": bool(pred_obj.get("use_branch_agg", False)),
                                "rgb_compress_quality": int(args.rgb_compress_quality),
                            },
                        }
            elif is_repeated_real and cache_hit_flag == 1:
                real_repeat_cache_hit += 1

            if role == "real":
                seen_real_keys.add(key)

            frame_paths = pred_obj.get("frame_paths", [])
            frame_probs = [float(x) for x in pred_obj.get("frame_probs", [])]
            frame_probs_rgb = [float(x) for x in pred_obj.get("frame_probs_rgb", [])]
            frame_probs_flow = [float(x) for x in pred_obj.get("frame_probs_flow", [])]
            num_frames = int(pred_obj.get("num_frames", len(frame_probs)))
            use_branch_agg = (
                bool(args.independent_branch_agg)
                and (len(frame_probs_rgb) == len(frame_probs))
                and (len(frame_probs_flow) == len(frame_probs))
                and (len(frame_probs) > 0)
            )
            if use_branch_agg:
                video_prob_rgb = aggregate_video_probability(
                    frame_probs=frame_probs_rgb,
                    method=args.rgb_video_agg_method,
                    topk_ratio=args.rgb_video_agg_topk_ratio,
                    topk_min=args.rgb_video_agg_topk_min,
                    topk_max=args.rgb_video_agg_topk_max,
                    hybrid_alpha=args.rgb_video_agg_hybrid_alpha,
                )
                video_prob_flow = aggregate_video_probability(
                    frame_probs=frame_probs_flow,
                    method=args.flow_video_agg_method,
                    topk_ratio=args.flow_video_agg_topk_ratio,
                    topk_min=args.flow_video_agg_topk_min,
                    topk_max=args.flow_video_agg_topk_max,
                    hybrid_alpha=args.flow_video_agg_hybrid_alpha,
                )
                w_rgb, w_flow = normalize_two_weights(args.branch_fuse_weight_rgb, args.branch_fuse_weight_flow)
                video_prob = float(w_rgb * video_prob_rgb + w_flow * video_prob_flow)
            else:
                video_prob = aggregate_video_probability(
                    frame_probs=frame_probs,
                    method=args.video_agg_method,
                    topk_ratio=args.video_agg_topk_ratio,
                    topk_min=args.video_agg_topk_min,
                    topk_max=args.video_agg_topk_max,
                    hybrid_alpha=args.video_agg_hybrid_alpha,
                )
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
        "motion_multi_offsets": [int(x) for x in normalize_motion_offsets(args.motion_stride, args.motion_multi_offsets)],
        "video_agg_method": str(args.video_agg_method),
        "video_agg_topk_ratio": float(args.video_agg_topk_ratio),
        "video_agg_topk_min": int(args.video_agg_topk_min),
        "video_agg_topk_max": int(args.video_agg_topk_max),
        "video_agg_hybrid_alpha": float(args.video_agg_hybrid_alpha),
        "independent_branch_agg": bool(args.independent_branch_agg),
        "rgb_video_agg_method": str(args.rgb_video_agg_method),
        "rgb_video_agg_topk_ratio": float(args.rgb_video_agg_topk_ratio),
        "rgb_video_agg_topk_min": int(args.rgb_video_agg_topk_min),
        "rgb_video_agg_topk_max": int(args.rgb_video_agg_topk_max),
        "rgb_video_agg_hybrid_alpha": float(args.rgb_video_agg_hybrid_alpha),
        "flow_video_agg_method": str(args.flow_video_agg_method),
        "flow_video_agg_topk_ratio": float(args.flow_video_agg_topk_ratio),
        "flow_video_agg_topk_min": int(args.flow_video_agg_topk_min),
        "flow_video_agg_topk_max": int(args.flow_video_agg_topk_max),
        "flow_video_agg_hybrid_alpha": float(args.flow_video_agg_hybrid_alpha),
        "branch_fuse_weight_rgb": float(args.branch_fuse_weight_rgb),
        "branch_fuse_weight_flow": float(args.branch_fuse_weight_flow),
        "rgb_compress_quality": int(args.rgb_compress_quality),
        "cache_hit_mem": int(cache_hit_mem),
        "cache_hit_disk": int(cache_hit_disk),
        "cache_miss": int(cache_miss),
        "cache_group_flush_count": int(cache_group_flush_count),
        "cache_group_flush_videos": int(cache_group_flush_videos),
        "real_repeat_total": int(real_repeat_total),
        "real_repeat_cache_hit": int(real_repeat_cache_hit),
        "skipped_empty": int(skipped_empty),
        "use_motion_token_cache": bool(args.use_motion_token_cache),
        "motion_token_cache_bucket": str(motion_token_cache.cache_bucket) if motion_token_cache is not None else "",
        "motion_token_cache_hit_mem": int(motion_token_cache.hit_mem) if motion_token_cache is not None else 0,
        "motion_token_cache_hit_disk": int(motion_token_cache.hit_disk) if motion_token_cache is not None else 0,
        "motion_token_cache_miss_compute": int(motion_token_cache.miss_compute) if motion_token_cache is not None else 0,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[Saved] frame-level csv: {out_csv}")
    print(f"[Saved] video-level csv: {video_csv}")


if __name__ == "__main__":
    main()
