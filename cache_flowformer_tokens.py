import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from config import get_default_config
from thesis_model import MotionEncoder


def parse_args() -> argparse.Namespace:
    cfg = get_default_config()
    data_cfg = cfg["data"]
    common_cfg = cfg["common"]
    flow_cfg = cfg.get("flow_cache", {})

    default_roots = flow_cfg.get(
        "roots",
        [data_cfg.get("train_root", ""), data_cfg.get("val_root", ""), data_cfg.get("test_root", "")],
    )
    default_roots = [str(x) for x in default_roots if str(x)]

    parser = argparse.ArgumentParser("Cache FlowFormer motion tokens for all video-frame folders")
    parser.add_argument("--roots", nargs="+", default=default_roots, help="Input roots to recursively scan.")
    parser.add_argument("--cache_dir", type=str, default=flow_cfg.get("cache_dir", "./data/exp/_shared/flowformer_token_cache"))
    parser.add_argument("--image_size", type=int, default=int(flow_cfg.get("image_size", data_cfg.get("image_size", 224))))
    parser.add_argument("--motion_stride", type=int, default=int(flow_cfg.get("motion_stride", data_cfg.get("motion_stride", 2))))
    parser.add_argument(
        "--motion_multi_offsets",
        type=int,
        nargs="+",
        default=[int(x) for x in flow_cfg.get("motion_multi_offsets", data_cfg.get("motion_multi_offsets", [1, 2, 4]))],
        help="Motion multi-interval offsets, e.g. 1 2 4.",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=int(flow_cfg.get("max_frames_per_video", data_cfg.get("max_frames_per_video", 0))),
    )
    parser.add_argument("--batch_size", type=int, default=int(flow_cfg.get("batch_size", 16)))
    parser.add_argument("--rgb_compress_quality", type=int, default=int(flow_cfg.get("rgb_compress_quality", 0)))
    parser.add_argument("--flowformer_repo", type=str, default=str(flow_cfg.get("flowformer_repo", "./FlowFormerPlusPlus-main")))
    parser.add_argument("--flowformer_ckpt", type=str, default=str(flow_cfg.get("flowformer_ckpt", "./checkpoints/things.pth")))
    parser.add_argument("--save_dtype", type=str, default=str(flow_cfg.get("save_dtype", "float16")), choices=["float16", "float32"])
    parser.add_argument("--print_freq", type=int, default=int(flow_cfg.get("print_freq", 20)))
    parser.add_argument(
        "--image_exts",
        nargs="+",
        default=[str(x).lower() for x in common_cfg.get("image_exts", [".jpg", ".jpeg", ".png", ".bmp", ".webp"])],
        help="Valid frame image extensions.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if bool_action is not None:
        parser.add_argument(
            "--require_flowformer",
            action=bool_action,
            default=bool(flow_cfg.get("require_flowformer", True)),
            help="Require FlowFormer load success.",
        )
        parser.add_argument(
            "--refresh_cache",
            action=bool_action,
            default=bool(flow_cfg.get("refresh_cache", False)),
            help="Ignore old cache and recompute.",
        )
    else:
        parser.add_argument("--require_flowformer", dest="require_flowformer", action="store_true")
        parser.add_argument("--no-require_flowformer", dest="require_flowformer", action="store_false")
        parser.set_defaults(require_flowformer=bool(flow_cfg.get("require_flowformer", True)))

        parser.add_argument("--refresh_cache", dest="refresh_cache", action="store_true")
        parser.add_argument("--no-refresh_cache", dest="refresh_cache", action="store_false")
        parser.set_defaults(refresh_cache=bool(flow_cfg.get("refresh_cache", False)))

    return parser.parse_args()


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
    seen: Set[int] = set()
    for v in offsets:
        if v in seen:
            continue
        seen.add(v)
        dedup.append(v)
    return dedup


def list_image_files(folder: Path, image_exts: Set[str]) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
    files.sort()
    return files


def is_video_frame_dir(folder: Path, image_exts: Set[str]) -> bool:
    if not folder.is_dir():
        return False
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in image_exts:
            return True
    return False


def collect_video_dirs(root: Path, image_exts: Set[str]) -> List[Path]:
    out: List[Path] = []
    if not root.exists():
        return out
    for d in sorted(root.rglob("*")):
        if d.is_dir() and is_video_frame_dir(d, image_exts):
            out.append(d)
    if is_video_frame_dir(root, image_exts):
        out = [root] + out

    dedup: List[Path] = []
    seen: Set[str] = set()
    for p in out:
        key = str(p.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup


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


def build_cache_bucket(args: argparse.Namespace, token_dim: int) -> Path:
    cache_root = resolve_abs(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_abs(args.flowformer_ckpt)
    try:
        ckpt_mtime = ckpt_path.stat().st_mtime_ns
    except Exception:
        ckpt_mtime = 0

    signature_obj = {
        "script": "cache_flowformer_tokens_v1",
        "flowformer_repo": str(resolve_abs(args.flowformer_repo)),
        "flowformer_ckpt": str(ckpt_path),
        "flowformer_ckpt_mtime_ns": int(ckpt_mtime),
        "image_size": int(args.image_size),
        "motion_stride": int(args.motion_stride),
        "motion_multi_offsets": [int(x) for x in normalize_motion_offsets(args.motion_stride, args.motion_multi_offsets)],
        "max_frames_per_video": int(args.max_frames_per_video),
        "rgb_compress_quality": int(args.rgb_compress_quality),
        "token_dim": int(token_dim),
        "save_dtype": str(args.save_dtype),
    }
    text = json.dumps(signature_obj, ensure_ascii=False, sort_keys=True)
    sign = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    bucket = cache_root / sign
    bucket.mkdir(parents=True, exist_ok=True)
    with open(bucket / "signature.json", "w", encoding="utf-8") as f:
        json.dump(signature_obj, f, ensure_ascii=False, indent=2)
    return bucket


def get_cache_file(cache_bucket: Path, video_dir: Path) -> Path:
    key = str(video_dir.resolve()).lower()
    hid = hashlib.sha1(key.encode("utf-8")).hexdigest()
    sub = cache_bucket / hid[:2]
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{hid}.pt"


def cache_one_video(
    motion_encoder: MotionEncoder,
    device: torch.device,
    video_dir: Path,
    cache_file: Path,
    image_exts: Set[str],
    image_size: int,
    motion_stride: int,
    motion_multi_offsets: Optional[List[int]],
    max_frames_per_video: int,
    batch_size: int,
    rgb_compress_quality: int,
    save_dtype: torch.dtype,
    token_dim: int,
) -> Dict[str, int]:
    frames = list_image_files(video_dir, image_exts=image_exts)
    if max_frames_per_video > 0:
        frames = frames[: int(max_frames_per_video)]
    if not frames:
        return {"num_frames": 0, "num_pairs": 0}

    transform = InferencePairTransform(image_size=image_size, rgb_compress_quality=rgb_compress_quality)
    n = len(frames)
    motion_offsets = normalize_motion_offsets(motion_stride, motion_multi_offsets)
    pair_entries: List[Tuple[int, int, int]] = []
    for i in range(n):
        for off in motion_offsets:
            j = 0 if n == 1 else min(n - 1, i + int(off))
            pair_entries.append((i, j, int(off)))
    pairs = [(int(i), int(j)) for i, j, _ in pair_entries]

    token_sum = torch.zeros((n, int(token_dim)), dtype=torch.float32)
    token_count = torch.zeros((n, 1), dtype=torch.float32)
    frame_paths: List[str] = [str(p.resolve()) for p in frames]
    pair_indices: List[Tuple[int, int]] = []

    step = max(1, int(batch_size))
    for start in range(0, len(pair_entries), step):
        batch_pairs = pair_entries[start : start + step]
        x1_list: List[torch.Tensor] = []
        x2_list: List[torch.Tensor] = []

        for i, j, _ in batch_pairs:
            with Image.open(frames[i]) as im1, Image.open(frames[j]) as im2:
                x1, x2 = transform(im1.convert("RGB"), im2.convert("RGB"))
            x1_list.append(x1)
            x2_list.append(x2)
            pair_indices.append((int(i), int(j)))

        x1_batch = torch.stack(x1_list, dim=0).to(device, non_blocking=True)  # [B, 3, H, W]
        x2_batch = torch.stack(x2_list, dim=0).to(device, non_blocking=True)  # [B, 3, H, W]

        with torch.no_grad():
            tokens = motion_encoder.extract_motion_tokens(x1_batch, x2_batch)  # [B, D]
        tokens_cpu = tokens.detach().cpu().to(dtype=torch.float32)
        for local_idx, (i, _, _) in enumerate(batch_pairs):
            token_sum[int(i)] += tokens_cpu[local_idx]
            token_count[int(i), 0] += 1.0

    token_count = token_count.clamp_min(1.0)
    motion_tokens = (token_sum / token_count).to(dtype=save_dtype)  # [N, D]

    payload = {
        "video_dir": str(video_dir.resolve()),
        "num_frames": int(n),
        "num_pairs": int(len(pairs)),
        "motion_stride": int(max(1, int(motion_stride))),
        "motion_multi_offsets": [int(x) for x in motion_offsets],
        "image_size": int(image_size),
        "rgb_compress_quality": int(rgb_compress_quality),
        "frame_paths": frame_paths,
        "pair_indices": pair_indices,
        "token_dim": int(motion_tokens.shape[1]) if motion_tokens.ndim == 2 else int(token_dim),
        "save_dtype": str(save_dtype).replace("torch.", ""),
        "motion_tokens": motion_tokens,  # [N, D]
    }

    tmp_file = cache_file.with_suffix(".tmp")
    torch.save(payload, str(tmp_file))
    tmp_file.replace(cache_file)
    return {"num_frames": int(n), "num_pairs": int(len(pairs))}


def main() -> None:
    args = parse_args()
    image_exts = {str(x).lower() for x in args.image_exts}
    roots = [resolve_abs(p) for p in args.roots if str(p).strip()]
    if not roots:
        raise RuntimeError("No valid roots provided. Set flow_cache.roots in config.py or pass --roots.")

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[Warn] CUDA is not available, fallback to CPU.")

    motion_encoder = MotionEncoder(
        out_dim=512,
        flowformer_repo=args.flowformer_repo,
        flowformer_ckpt=args.flowformer_ckpt,
        require_flowformer=bool(args.require_flowformer),
    ).to(device)
    motion_encoder.eval()

    if motion_encoder.use_flowformer:
        token_dim = int(motion_encoder.flowformer_token_dim)
    else:
        token_dim = 256
    save_dtype = torch.float16 if str(args.save_dtype).lower() == "float16" else torch.float32

    cache_bucket = build_cache_bucket(args, token_dim=token_dim)
    print(f"[Cache] bucket={cache_bucket}")
    print(
        f"[Config] image_size={int(args.image_size)} motion_stride={int(args.motion_stride)} "
        f"motion_multi_offsets={normalize_motion_offsets(args.motion_stride, args.motion_multi_offsets)} "
        f"max_frames_per_video={int(args.max_frames_per_video)} batch_size={int(args.batch_size)} "
        f"rgb_compress_quality={int(args.rgb_compress_quality)} save_dtype={str(args.save_dtype)}"
    )

    all_videos: List[Path] = []
    for root in roots:
        if not root.exists():
            print(f"[SkipRoot] not found: {root}")
            continue
        vids = collect_video_dirs(root, image_exts=image_exts)
        print(f"[Scan] root={root} videos={len(vids)}")
        all_videos.extend(vids)

    dedup_videos: List[Path] = []
    seen: Set[str] = set()
    for p in all_videos:
        k = str(p.resolve()).lower()
        if k in seen:
            continue
        seen.add(k)
        dedup_videos.append(p)
    all_videos = dedup_videos

    if not all_videos:
        raise RuntimeError("No frame folders found. Please check roots and image_exts.")

    total = len(all_videos)
    processed = 0
    skipped_exists = 0
    empty_videos = 0
    failed = 0
    total_pairs = 0
    total_frames = 0

    errors: List[Dict[str, str]] = []
    for idx, video_dir in enumerate(all_videos, start=1):
        cache_file = get_cache_file(cache_bucket, video_dir)
        if cache_file.exists() and (not args.refresh_cache):
            skipped_exists += 1
        else:
            try:
                info = cache_one_video(
                    motion_encoder=motion_encoder,
                    device=device,
                    video_dir=video_dir,
                    cache_file=cache_file,
                    image_exts=image_exts,
                    image_size=int(args.image_size),
                    motion_stride=int(args.motion_stride),
                    motion_multi_offsets=args.motion_multi_offsets,
                    max_frames_per_video=int(args.max_frames_per_video),
                    batch_size=int(args.batch_size),
                    rgb_compress_quality=int(args.rgb_compress_quality),
                    save_dtype=save_dtype,
                    token_dim=token_dim,
                )
                if int(info["num_frames"]) <= 0:
                    empty_videos += 1
                else:
                    processed += 1
                    total_frames += int(info["num_frames"])
                    total_pairs += int(info["num_pairs"])
            except Exception as e:
                failed += 1
                errors.append({"video_dir": str(video_dir), "error": str(e)})

        if (idx % max(1, int(args.print_freq)) == 0) or (idx == total):
            print(
                f"[Progress] {idx}/{total} processed={processed} skipped_exists={skipped_exists} "
                f"empty={empty_videos} failed={failed}"
            )

    summary = {
        "total_videos": int(total),
        "processed": int(processed),
        "skipped_exists": int(skipped_exists),
        "empty_videos": int(empty_videos),
        "failed": int(failed),
        "total_frames": int(total_frames),
        "total_pairs": int(total_pairs),
        "cache_bucket": str(cache_bucket),
        "device": str(device),
        "flowformer_repo": str(resolve_abs(args.flowformer_repo)),
        "flowformer_ckpt": str(resolve_abs(args.flowformer_ckpt)),
        "image_size": int(args.image_size),
        "motion_stride": int(args.motion_stride),
        "motion_multi_offsets": [int(x) for x in normalize_motion_offsets(args.motion_stride, args.motion_multi_offsets)],
        "max_frames_per_video": int(args.max_frames_per_video),
        "rgb_compress_quality": int(args.rgb_compress_quality),
        "save_dtype": str(args.save_dtype),
        "refresh_cache": bool(args.refresh_cache),
    }
    with open(cache_bucket / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if errors:
        with open(cache_bucket / "errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[Done] summary: {cache_bucket / 'summary.json'}")
    if errors:
        print(f"[Done] errors: {cache_bucket / 'errors.json'}")


if __name__ == "__main__":
    main()
