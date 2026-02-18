import argparse
import csv
import hashlib
import io
import json
import math
import random
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from config import get_default_config
from thesis_model import Enhanced_STF_Detector

_CFG = get_default_config()
_COMMON_CFG = _CFG["common"]
_DATA_CFG = _CFG["data"]

IMAGE_EXTS = {str(x).lower() for x in _COMMON_CFG.get("image_exts", [".jpg", ".jpeg", ".png", ".bmp", ".webp"])}
REAL_CLASS_NAMES = [str(x).lower() for x in _DATA_CFG.get("real_class_names", ["0_real", "real"])]
FAKE_CLASS_NAMES = [str(x).lower() for x in _DATA_CFG.get("fake_class_names", ["1_fake", "fake"])]


def _torch_load_compat(path: str, map_location: str = "cpu", weights_only: Optional[bool] = None):
    kwargs = {"map_location": map_location}
    if weights_only is not None:
        kwargs["weights_only"] = weights_only
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(path, **kwargs)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_image_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    return files


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


def infer_class_dirs(root: Path) -> List[Tuple[Path, int]]:
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError(f"No class directories found in {root}")

    mapping: Dict[str, int] = {}
    for name in REAL_CLASS_NAMES:
        mapping[name] = 0
    for name in FAKE_CLASS_NAMES:
        mapping[name] = 1
    out: List[Tuple[Path, int]] = []

    for d in candidates:
        name = d.name.lower()
        if name in mapping:
            out.append((d, mapping[name]))

    if out:
        return out

    # fallback: two folders by alphabetical order
    candidates = sorted(candidates, key=lambda x: x.name.lower())
    if len(candidates) != 2:
        real_hint = "/".join(REAL_CLASS_NAMES) if REAL_CLASS_NAMES else "0_real/real"
        fake_hint = "/".join(FAKE_CLASS_NAMES) if FAKE_CLASS_NAMES else "1_fake/fake"
        raise RuntimeError(
            f"Unable to infer labels. Use class dirs named one of [{real_hint}] / [{fake_hint}], "
            f"or provide exactly two class folders. Got {len(candidates)}."
        )
    return [(candidates[0], 0), (candidates[1], 1)]


class PairTransform:
    def __init__(self, image_size: int, train: bool):
        self.image_size = int(image_size)
        self.train = train
        self.resize_size = int(round(self.image_size * 1.14))

    def __call__(self, img1: Image.Image, img2: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            img1 = TF.resize(img1, [self.resize_size, self.resize_size], interpolation=InterpolationMode.BILINEAR)
            img2 = TF.resize(img2, [self.resize_size, self.resize_size], interpolation=InterpolationMode.BILINEAR)

            i, j, h, w = torch.randint(
                low=0,
                high=max(1, self.resize_size - self.image_size + 1),
                size=(2,),
            ).tolist() + [self.image_size, self.image_size]
            i = int(i)
            j = int(j)
            img1 = TF.crop(img1, i, j, h, w)
            img2 = TF.crop(img2, i, j, h, w)

            if random.random() < 0.5:
                img1 = TF.hflip(img1)
                img2 = TF.hflip(img2)
        else:
            img1 = TF.resize(img1, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
            img2 = TF.resize(img2, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)

        x1 = TF.to_tensor(img1)  # [3,H,W], range [0,1]
        x2 = TF.to_tensor(img2)  # [3,H,W], range [0,1]
        return x1, x2


class DeterministicMotionTransform:
    """用于 motion token 缓存的确定性预处理（与随机增强解耦）。"""

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
    """
    训练阶段 token 缓存管理：
    1. 先查内存
    2. 再查磁盘
    3. 缓存不存在时在线计算并落盘
    """

    def __init__(
        self,
        motion_encoder: nn.Module,
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
        mem_videos: int = 64,
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
        self.transform = DeterministicMotionTransform(
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
                f"[MotionCache] computed={self.miss_compute} hit_mem={self.hit_mem} "
                f"hit_disk={self.hit_disk} bucket={self.cache_bucket}"
            )
        return video_tokens

    def ensure_video_tokens(self, video_dir: str) -> torch.Tensor:
        video_path = Path(video_dir)
        key = self._video_key(video_path)

        tokens = self._mem_get(key)
        if tokens is not None:
            self.hit_mem += 1
            return tokens

        cache_file = self._cache_file(video_path)
        if cache_file.exists() and (not self.refresh_cache):
            tokens = self._load_video_tokens(cache_file)
            if tokens is not None:
                self.hit_disk += 1
                self._mem_set(key, tokens)
                return tokens

        tokens = self._compute_video_tokens(video_path)
        self._mem_set(key, tokens)
        return tokens

    @staticmethod
    def _to_int_list(x) -> List[int]:
        if torch.is_tensor(x):
            return [int(v) for v in x.detach().cpu().tolist()]
        if isinstance(x, (list, tuple)):
            return [int(v) for v in x]
        return [int(x)]

    @staticmethod
    def _to_str_list(x) -> List[str]:
        if isinstance(x, (list, tuple)):
            return [str(v) for v in x]
        return [str(x)]

    def get_batch_tokens(self, video_dirs, frame_indices) -> torch.Tensor:
        video_dirs_list = self._to_str_list(video_dirs)
        frame_indices_list = self._to_int_list(frame_indices)
        if len(video_dirs_list) != len(frame_indices_list):
            raise RuntimeError(
                f"Batch meta length mismatch: video_dirs={len(video_dirs_list)} "
                f"frame_indices={len(frame_indices_list)}"
            )

        out_tokens: List[torch.Tensor] = []
        for video_dir, idx in zip(video_dirs_list, frame_indices_list):
            video_tokens = self.ensure_video_tokens(video_dir)
            if video_tokens.size(0) <= 0:
                out_tokens.append(torch.zeros((self.token_dim,), dtype=self.save_dtype))
                continue
            safe_idx = min(max(int(idx), 0), int(video_tokens.size(0) - 1))
            out_tokens.append(video_tokens[safe_idx])

        batch_tokens = torch.stack(out_tokens, dim=0)  # [B, D]
        return batch_tokens.to(self.device, non_blocking=True)


class DeepfakePairDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 224,
        split: str = "train",
        motion_stride: int = 2,
        max_frames_per_video: int = 0,
        deterministic_motion_pair: bool = False,
        load_motion_images: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.motion_stride = max(1, int(motion_stride))
        self.max_frames_per_video = max(0, int(max_frames_per_video))
        self.deterministic_motion_pair = bool(deterministic_motion_pair)
        self.load_motion_images = bool(load_motion_images)
        self.transform = PairTransform(image_size=image_size, train=(split == "train"))
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Dict]:
        if not self.root.exists():
            raise RuntimeError(f"Dataset root not found: {self.root}")

        class_dirs = infer_class_dirs(self.root)
        samples: List[Dict] = []

        for class_dir, label in class_dirs:
            subdirs = [d for d in class_dir.iterdir() if d.is_dir()]
            videos = subdirs if subdirs else [class_dir]

            for video_dir in videos:
                frames = list_image_files(video_dir)
                if not frames:
                    continue

                if self.max_frames_per_video > 0:
                    frames = frames[: self.max_frames_per_video]

                video_id = f"{class_dir.name}/{video_dir.name}"
                n = len(frames)
                for i, frame_i in enumerate(frames):
                    samples.append(
                        {
                            "frames": frames,
                            "idx1": i,
                            "label": float(label),
                            "video_id": video_id,
                            "path": str(frame_i),
                            "video_dir": str(video_dir.resolve()),
                        }
                    )

        if not samples:
            raise RuntimeError(f"No valid image samples found under {self.root}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        frames: List[Path] = sample["frames"]
        i = int(sample["idx1"])
        n = len(frames)

        if n == 1:
            j = 0
        elif self.split == "train" and (not self.deterministic_motion_pair):
            max_j = min(n - 1, i + self.motion_stride)
            if max_j <= i:
                j = n - 1
            else:
                j = random.randint(i + 1, max_j)
        else:
            j = min(n - 1, i + self.motion_stride)

        with Image.open(frames[i]) as img1_raw:
            img1 = img1_raw.convert("RGB")
        if self.load_motion_images:
            with Image.open(frames[j]) as img2_raw:
                img2 = img2_raw.convert("RGB")
        else:
            img2 = img1.copy()
        x1, x2 = self.transform(img1, img2)

        return {
            "img_spatial": x1,  # [3,H,W]
            "img_motion_1": x1,  # [3,H,W]
            "img_motion_2": x2,  # [3,H,W]
            "label": torch.tensor(sample["label"], dtype=torch.float32),  # []
            "video_id": sample["video_id"],
            "path": sample["path"],
            "motion_video_dir": sample["video_dir"],
            "motion_idx1": i,
        }


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def compute_losses(
    outputs: Dict,
    targets: torch.Tensor,
    criterion: nn.Module,
    fusion_mode: str,
    aux_loss_weight: float,
    fusion_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # targets: [B,1]
    logits = outputs["logits"]  # [B,1]
    main_loss = criterion(logits, targets)

    if fusion_mode == "independent":
        logit_s, logit_t = outputs["aux_logits"]  # [B,1], [B,1]
        aux_loss = 0.5 * (criterion(logit_s, targets) + criterion(logit_t, targets))
        total_loss = main_loss + aux_loss_weight * aux_loss
        return total_loss, {
            "main_loss": float(main_loss.detach().cpu()),
            "aux_loss": float(aux_loss.detach().cpu()),
            "fusion_reg": 0.0,
        }

    if "fusion_loss" not in outputs:
        raise KeyError(
            "[Loss] fusion_mode='cross_attention' but model outputs has no 'fusion_loss'. "
            "Please check that training args and model fusion_mode are consistent."
        )
    fusion_reg = outputs["fusion_loss"]
    total_loss = main_loss + fusion_loss_weight * fusion_reg
    return total_loss, {
        "main_loss": float(main_loss.detach().cpu()),
        "aux_loss": 0.0,
        "fusion_reg": float(fusion_reg.detach().cpu()),
    }


def _calc_topk_k(num_frames: int, ratio: float, k_min: int, k_max: int) -> int:
    n = max(0, int(num_frames))
    if n <= 0:
        return 0
    r = float(ratio)
    if not math.isfinite(r):
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
    if not math.isfinite(alpha):
        alpha = 0.7
    alpha = min(max(alpha, 0.0), 1.0)
    return float(alpha * topk_mean + (1.0 - alpha) * all_mean)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    fusion_mode: str,
    aux_loss_weight: float,
    fusion_loss_weight: float,
    threshold: float = 0.5,
    video_agg_method: str = "topk_mean",
    video_agg_topk_ratio: float = 0.1,
    video_agg_topk_min: int = 3,
    video_agg_topk_max: int = 32,
    video_agg_hybrid_alpha: float = 0.7,
    motion_token_cache: Optional[MotionTokenCacheManager] = None,
) -> Tuple[Dict[str, float], List[Dict], List[Dict]]:
    model.eval()
    losses: List[float] = []
    probs_all: List[float] = []
    labels_all: List[float] = []
    video_pool: Dict[str, List[Tuple[float, float]]] = {}
    frame_rows: List[Dict] = []

    for batch in loader:
        img_spatial = batch["img_spatial"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)  # [B,1]

        if motion_token_cache is not None:
            motion_tokens = motion_token_cache.get_batch_tokens(
                video_dirs=batch["motion_video_dir"],
                frame_indices=batch["motion_idx1"],
            )
            outputs = model(img_spatial, motion_tokens=motion_tokens)
        else:
            img_motion_1 = batch["img_motion_1"].to(device, non_blocking=True)
            img_motion_2 = batch["img_motion_2"].to(device, non_blocking=True)
            outputs = model(img_spatial, img_motion_1, img_motion_2)
        loss, _ = compute_losses(
            outputs=outputs,
            targets=labels,
            criterion=criterion,
            fusion_mode=fusion_mode,
            aux_loss_weight=aux_loss_weight,
            fusion_loss_weight=fusion_loss_weight,
        )
        losses.append(float(loss.detach().cpu()))

        probs = torch.sigmoid(outputs["logits"]).squeeze(1)  # [B]
        probs_np = probs.detach().cpu().numpy().tolist()
        labels_np = labels.squeeze(1).detach().cpu().numpy().tolist()
        probs_all.extend(probs_np)
        labels_all.extend(labels_np)

        for path, vid, p, y in zip(batch["path"], batch["video_id"], probs_np, labels_np):
            if vid not in video_pool:
                video_pool[vid] = []
            video_pool[vid].append((float(p), float(y)))
            frame_rows.append(
                {
                    "path": path,
                    "video_id": vid,
                    "label": int(y),
                    "prob": float(p),
                    "pred": int(float(p) >= threshold),
                }
            )

    frame_preds = [1.0 if p >= threshold else 0.0 for p in probs_all]
    frame_acc = float(np.mean([int(p == y) for p, y in zip(frame_preds, labels_all)])) if labels_all else 0.0

    video_probs: List[float] = []
    video_labels: List[float] = []
    video_rows: List[Dict] = []
    for _, items in video_pool.items():
        p = aggregate_video_probability(
            frame_probs=[float(it[0]) for it in items],
            method=video_agg_method,
            topk_ratio=video_agg_topk_ratio,
            topk_min=video_agg_topk_min,
            topk_max=video_agg_topk_max,
            hybrid_alpha=video_agg_hybrid_alpha,
        )
        y = float(round(np.mean([it[1] for it in items])))
        video_probs.append(p)
        video_labels.append(y)
        video_rows.append(
            {
                "video_id": _,
                "label": int(y),
                "prob": float(p),
                "pred": int(float(p) >= threshold),
                "num_frames": len(items),
            }
        )
    video_preds = [1.0 if p >= threshold else 0.0 for p in video_probs]
    video_acc = float(np.mean([int(p == y) for p, y in zip(video_preds, video_labels)])) if video_labels else 0.0

    metrics: Dict[str, float] = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "frame_acc": frame_acc,
        "video_acc": video_acc,
    }

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(set(labels_all)) >= 2:
            metrics["frame_auc"] = float(roc_auc_score(labels_all, probs_all))
            metrics["frame_ap"] = float(average_precision_score(labels_all, probs_all))
        else:
            metrics["frame_auc"] = float("nan")
            metrics["frame_ap"] = float("nan")

        if len(set(video_labels)) >= 2:
            metrics["video_auc"] = float(roc_auc_score(video_labels, video_probs))
            metrics["video_ap"] = float(average_precision_score(video_labels, video_probs))
        else:
            metrics["video_auc"] = float("nan")
            metrics["video_ap"] = float("nan")
    except Exception:
        metrics["frame_auc"] = float("nan")
        metrics["frame_ap"] = float("nan")
        metrics["video_auc"] = float("nan")
        metrics["video_ap"] = float("nan")

    return metrics, frame_rows, video_rows


def is_finite_number(x: Optional[float]) -> bool:
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def select_monitor_value(val_metrics: Dict[str, float], metric: str) -> Tuple[Optional[float], str, bool]:
    """
    Returns:
        monitor_value, monitor_name, higher_is_better
    """
    if metric == "loss":
        value = val_metrics.get("loss", float("nan"))
        return (float(value) if is_finite_number(value) else None), "loss", False

    if metric == "acc":
        video_acc = val_metrics.get("video_acc", float("nan"))
        if is_finite_number(video_acc):
            return float(video_acc), "video_acc", True
        frame_acc = val_metrics.get("frame_acc", float("nan"))
        if is_finite_number(frame_acc):
            return float(frame_acc), "frame_acc", True
        return None, "acc", True

    # default: auc
    video_auc = val_metrics.get("video_auc", float("nan"))
    if is_finite_number(video_auc):
        return float(video_auc), "video_auc", True
    frame_auc = val_metrics.get("frame_auc", float("nan"))
    if is_finite_number(frame_auc):
        return float(frame_auc), "frame_auc", True

    # AUC is undefined on single-class validation set; fallback to ACC.
    video_acc = val_metrics.get("video_acc", float("nan"))
    if is_finite_number(video_acc):
        return float(video_acc), "video_acc(fallback_from_auc)", True
    frame_acc = val_metrics.get("frame_acc", float("nan"))
    if is_finite_number(frame_acc):
        return float(frame_acc), "frame_acc(fallback_from_auc)", True

    return None, "auc", True


def is_improved(
    current: float,
    best: Optional[float],
    higher_is_better: bool,
    min_delta: float,
) -> bool:
    if best is None:
        return True
    if higher_is_better:
        return current > (best + min_delta)
    return current < (best - min_delta)


def save_checkpoint(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def save_val_prediction_cache(
    save_dir: Path,
    epoch: int,
    frame_rows: List[Dict],
    video_rows: List[Dict],
) -> Tuple[Path, Path]:
    cache_dir = save_dir / "val_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    frame_csv = cache_dir / f"epoch_{epoch:03d}_frame.csv"
    with open(frame_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "video_id", "label", "prob", "pred"])
        writer.writeheader()
        writer.writerows(frame_rows)

    video_csv = cache_dir / f"epoch_{epoch:03d}_video.csv"
    with open(video_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "label", "prob", "pred", "num_frames"])
        writer.writeheader()
        writer.writerows(video_rows)

    return frame_csv, video_csv


def parse_args() -> argparse.Namespace:
    cfg = get_default_config()
    common_cfg = cfg["common"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    parser = argparse.ArgumentParser("Train Enhanced_STF_Detector")
    parser.add_argument("--train_root", type=str, default=data_cfg["train_root"], help="Train root dir with class folders.")
    parser.add_argument("--val_root", type=str, default=data_cfg["val_root"], help="Validation root dir.")
    parser.add_argument("--save_dir", type=str, default=train_cfg["save_dir"])
    parser.add_argument("--resume", type=str, default=train_cfg["resume"], help="Checkpoint path to resume.")

    parser.add_argument("--epochs", type=int, default=train_cfg["epochs"])
    parser.add_argument("--batch_size", type=int, default=train_cfg["batch_size"])
    parser.add_argument("--num_workers", type=int, default=train_cfg["num_workers"])
    parser.add_argument("--lr", type=float, default=train_cfg["lr"])
    parser.add_argument("--weight_decay", type=float, default=train_cfg["weight_decay"])
    parser.add_argument("--print_freq", type=int, default=train_cfg["print_freq"])
    parser.add_argument(
        "--save_every",
        type=int,
        default=train_cfg["save_every"],
        help="Save epoch checkpoint every N epochs (latest.pth is still saved every epoch).",
    )
    parser.add_argument("--seed", type=int, default=common_cfg["seed"])

    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if bool_action is not None:
        parser.add_argument("--amp", action=bool_action, default=train_cfg["amp"], help="Enable mixed precision on CUDA.")
    else:
        parser.add_argument("--amp", dest="amp", action="store_true", help="Enable mixed precision on CUDA.")
        parser.add_argument("--no-amp", dest="amp", action="store_false")
        parser.set_defaults(amp=train_cfg["amp"])

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

    parser.add_argument(
        "--fusion_mode",
        type=str,
        default=model_cfg["fusion_mode"],
        choices=["independent", "cross_attention"],
    )
    parser.add_argument("--feature_dim", type=int, default=model_cfg["feature_dim"])

    if bool_action is not None:
        parser.add_argument("--use_resnet_imagenet", action=bool_action, default=model_cfg["use_resnet_imagenet"])
    else:
        parser.add_argument("--use_resnet_imagenet", dest="use_resnet_imagenet", action="store_true")
        parser.add_argument("--no-use_resnet_imagenet", dest="use_resnet_imagenet", action="store_false")
        parser.set_defaults(use_resnet_imagenet=model_cfg["use_resnet_imagenet"])
    if bool_action is not None:
        parser.add_argument(
            "--require_flowformer",
            action=bool_action,
            default=model_cfg["require_flowformer"],
            help="Require FlowFormer to load successfully; otherwise raise error.",
        )
    else:
        parser.add_argument("--require_flowformer", dest="require_flowformer", action="store_true")
        parser.add_argument("--no-require_flowformer", dest="require_flowformer", action="store_false")
        parser.set_defaults(require_flowformer=model_cfg["require_flowformer"])

    parser.add_argument("--hfri_mode", type=str, default=model_cfg["hfri_mode"], choices=["fft", "dct"])
    parser.add_argument("--aux_loss_weight", type=float, default=train_cfg["aux_loss_weight"])
    parser.add_argument("--fusion_loss_weight", type=float, default=train_cfg["fusion_loss_weight"])
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default=train_cfg["early_stop_metric"],
        choices=["auc", "acc", "loss"],
        help="Validation metric for early stopping.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=train_cfg["early_stop_patience"],
        help="Stop if no improvement for this many consecutive validation epochs.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=train_cfg["early_stop_min_delta"],
        help="Minimum metric improvement to reset patience.",
    )
    if bool_action is not None:
        parser.add_argument(
            "--cache_val_predictions",
            action=bool_action,
            default=train_cfg["cache_val_predictions"],
            help="Cache validation prediction data to CSV for threshold searching.",
        )
    else:
        parser.add_argument("--cache_val_predictions", dest="cache_val_predictions", action="store_true")
        parser.add_argument("--no-cache_val_predictions", dest="cache_val_predictions", action="store_false")
        parser.set_defaults(cache_val_predictions=train_cfg["cache_val_predictions"])
    parser.add_argument(
        "--cache_val_every",
        type=int,
        default=train_cfg["cache_val_every"],
        help="Cache validation predictions every N validation epochs.",
    )
    parser.add_argument(
        "--cache_val_threshold",
        type=float,
        default=train_cfg["cache_val_threshold"],
        help="Threshold used to generate pred column in cached CSV.",
    )
    parser.add_argument(
        "--val_video_agg_method",
        type=str,
        default=train_cfg.get("val_video_agg_method", "topk_mean"),
        choices=["mean", "topk_mean", "hybrid_topk_mean"],
        help="Video-level aggregation for validation metrics/cache.",
    )
    parser.add_argument(
        "--val_video_agg_topk_ratio",
        type=float,
        default=float(train_cfg.get("val_video_agg_topk_ratio", 0.1)),
    )
    parser.add_argument(
        "--val_video_agg_topk_min",
        type=int,
        default=int(train_cfg.get("val_video_agg_topk_min", 3)),
    )
    parser.add_argument(
        "--val_video_agg_topk_max",
        type=int,
        default=int(train_cfg.get("val_video_agg_topk_max", 32)),
    )
    parser.add_argument(
        "--val_video_agg_hybrid_alpha",
        type=float,
        default=float(train_cfg.get("val_video_agg_hybrid_alpha", 0.7)),
    )
    if bool_action is not None:
        parser.add_argument(
            "--use_motion_token_cache",
            action=bool_action,
            default=bool(train_cfg.get("use_motion_token_cache", True)),
            help="Use shared FlowFormer token cache for training/validation.",
        )
        parser.add_argument(
            "--motion_token_cache_refresh",
            action=bool_action,
            default=bool(train_cfg.get("motion_token_cache_refresh", False)),
            help="Ignore existing token cache and recompute.",
        )
        parser.add_argument(
            "--motion_token_cache_for_val",
            action=bool_action,
            default=bool(train_cfg.get("motion_token_cache_for_val", True)),
            help="Use token cache in validation.",
        )
        parser.add_argument(
            "--motion_token_cache_force_deterministic_pairs",
            action=bool_action,
            default=bool(train_cfg.get("motion_token_cache_force_deterministic_pairs", True)),
            help="Use deterministic i->i+stride pairs when token cache is enabled.",
        )
    else:
        parser.add_argument("--use_motion_token_cache", dest="use_motion_token_cache", action="store_true")
        parser.add_argument("--no-use_motion_token_cache", dest="use_motion_token_cache", action="store_false")
        parser.set_defaults(use_motion_token_cache=bool(train_cfg.get("use_motion_token_cache", True)))

        parser.add_argument("--motion_token_cache_refresh", dest="motion_token_cache_refresh", action="store_true")
        parser.add_argument("--no-motion_token_cache_refresh", dest="motion_token_cache_refresh", action="store_false")
        parser.set_defaults(motion_token_cache_refresh=bool(train_cfg.get("motion_token_cache_refresh", False)))

        parser.add_argument("--motion_token_cache_for_val", dest="motion_token_cache_for_val", action="store_true")
        parser.add_argument("--no-motion_token_cache_for_val", dest="motion_token_cache_for_val", action="store_false")
        parser.set_defaults(motion_token_cache_for_val=bool(train_cfg.get("motion_token_cache_for_val", True)))

        parser.add_argument(
            "--motion_token_cache_force_deterministic_pairs",
            dest="motion_token_cache_force_deterministic_pairs",
            action="store_true",
        )
        parser.add_argument(
            "--no-motion_token_cache_force_deterministic_pairs",
            dest="motion_token_cache_force_deterministic_pairs",
            action="store_false",
        )
        parser.set_defaults(
            motion_token_cache_force_deterministic_pairs=bool(
                train_cfg.get("motion_token_cache_force_deterministic_pairs", True)
            )
        )
    parser.add_argument(
        "--motion_token_cache_dir",
        type=str,
        default=str(train_cfg.get("motion_token_cache_dir", "./data/exp/_shared/flowformer_token_cache")),
    )
    parser.add_argument(
        "--motion_token_cache_dtype",
        type=str,
        default=str(train_cfg.get("motion_token_cache_dtype", "float16")),
        choices=["float16", "float32"],
    )
    parser.add_argument(
        "--motion_token_cache_rgb_compress_quality",
        type=int,
        default=int(train_cfg.get("motion_token_cache_rgb_compress_quality", 0)),
    )
    parser.add_argument(
        "--motion_token_cache_mem_videos",
        type=int,
        default=int(train_cfg.get("motion_token_cache_mem_videos", 64)),
        help="Max number of video token tensors kept in memory (LRU).",
    )
    parser.add_argument(
        "--motion_token_cache_batch_size",
        type=int,
        default=int(train_cfg.get("motion_token_cache_batch_size", 16)),
        help="Batch size used when computing token cache on miss.",
    )
    parser.add_argument(
        "--motion_token_cache_print_freq",
        type=int,
        default=int(train_cfg.get("motion_token_cache_print_freq", 20)),
        help="Print motion-cache stats every N miss computations.",
    )

    parser.add_argument("--flowformer_repo", type=str, default=model_cfg["flowformer_repo"])
    parser.add_argument("--flowformer_ckpt", type=str, default=model_cfg["flowformer_ckpt"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    use_motion_token_cache = bool(args.use_motion_token_cache)
    motion_offsets = normalize_motion_offsets(args.motion_stride, args.motion_multi_offsets)
    force_det_pairs = bool(use_motion_token_cache and args.motion_token_cache_force_deterministic_pairs)
    use_motion_cache_for_val = bool(use_motion_token_cache and args.motion_token_cache_for_val)
    if (not use_motion_token_cache) and len(motion_offsets) > 1:
        print(
            "[Warn] use_motion_token_cache=False while motion_multi_offsets has multiple values. "
            "Train no-cache path uses single pair sampling and will not perform full multi-interval fusion."
        )

    train_dataset = DeepfakePairDataset(
        root=args.train_root,
        image_size=args.image_size,
        split="train",
        motion_stride=args.motion_stride,
        max_frames_per_video=args.max_frames_per_video,
        deterministic_motion_pair=force_det_pairs,
        load_motion_images=(not use_motion_token_cache),
    )
    val_dataset: Optional[DeepfakePairDataset] = None
    if args.val_root:
        val_dataset = DeepfakePairDataset(
            root=args.val_root,
            image_size=args.image_size,
            split="val",
            motion_stride=args.motion_stride,
            max_frames_per_video=args.max_frames_per_video,
            deterministic_motion_pair=use_motion_cache_for_val,
            load_motion_images=(not use_motion_cache_for_val),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        if val_dataset is not None
        else None
    )

    model = Enhanced_STF_Detector(
        feature_dim=args.feature_dim,
        fusion_mode=args.fusion_mode,
        use_resnet_imagenet=args.use_resnet_imagenet,
        hfri_mode=args.hfri_mode,
        flowformer_repo=args.flowformer_repo,
        flowformer_ckpt=args.flowformer_ckpt,
        require_flowformer=args.require_flowformer,
    ).to(device)
    print(
        f"[Train] fusion_mode={args.fusion_mode} feature_dim={args.feature_dim} "
        f"hfri_mode={args.hfri_mode} flowformer_ckpt={args.flowformer_ckpt}"
    )

    motion_token_cache: Optional[MotionTokenCacheManager] = None
    if use_motion_token_cache:
        motion_token_cache = MotionTokenCacheManager(
            motion_encoder=model.motion_encoder,
            device=device,
            cache_dir=args.motion_token_cache_dir,
            image_size=args.image_size,
            motion_stride=args.motion_stride,
            motion_multi_offsets=args.motion_multi_offsets,
            max_frames_per_video=args.max_frames_per_video,
            rgb_compress_quality=args.motion_token_cache_rgb_compress_quality,
            flowformer_repo=args.flowformer_repo,
            flowformer_ckpt=args.flowformer_ckpt,
            save_dtype=args.motion_token_cache_dtype,
            refresh_cache=args.motion_token_cache_refresh,
            mem_videos=args.motion_token_cache_mem_videos,
            compute_batch_size=args.motion_token_cache_batch_size,
            print_freq=args.motion_token_cache_print_freq,
        )
        print(
            f"[MotionCache] enabled bucket={motion_token_cache.cache_bucket} "
            f"dtype={args.motion_token_cache_dtype} refresh={bool(args.motion_token_cache_refresh)} "
            f"det_pairs={force_det_pairs} val_cache={use_motion_cache_for_val} "
            f"offsets={motion_token_cache.motion_offsets}"
        )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_metric: Optional[float] = None
    best_metric_name = ""
    best_metric_higher_is_better = True
    early_stop_wait = 0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckpt = _torch_load_compat(args.resume, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        if isinstance(ckpt, dict):
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt and use_amp:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            if "best_metric" in ckpt and ckpt["best_metric"] is not None:
                try:
                    best_metric = float(ckpt["best_metric"])
                except Exception:
                    best_metric = None
            best_metric_name = str(ckpt.get("best_metric_name", best_metric_name))
            best_metric_higher_is_better = bool(
                ckpt.get("best_metric_higher_is_better", best_metric_higher_is_better)
            )
            early_stop_wait = int(ckpt.get("early_stop_wait", 0))

        best_metric_str = f"{best_metric:.4f}" if best_metric is not None else "None"
        print(
            f"[Resume] from {args.resume}, start_epoch={start_epoch}, "
            f"best_metric={best_metric_str}, wait={early_stop_wait}"
        )

    with open(save_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    if val_loader is None:
        print("[EarlyStop] Validation set is not provided. Early stopping is disabled.")
    else:
        print(
            f"[ValAgg] method={args.val_video_agg_method} "
            f"ratio={float(args.val_video_agg_topk_ratio):.3f} "
            f"min={int(args.val_video_agg_topk_min)} "
            f"max={int(args.val_video_agg_topk_max)} "
            f"alpha={float(args.val_video_agg_hybrid_alpha):.2f}"
        )

    if args.early_stop_metric == "loss" and best_metric is not None and best_metric < 0:
        # Compatibility with older checkpoints that initialized best_metric as -1.
        best_metric = None

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_main = 0.0
        running_aux = 0.0
        running_fusion_reg = 0.0

        for step, batch in enumerate(train_loader, start=1):
            img_spatial = batch["img_spatial"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)  # [B,1]
            motion_tokens = None
            img_motion_1 = None
            img_motion_2 = None

            if motion_token_cache is not None:
                motion_tokens = motion_token_cache.get_batch_tokens(
                    video_dirs=batch["motion_video_dir"],
                    frame_indices=batch["motion_idx1"],
                )
            else:
                img_motion_1 = batch["img_motion_1"].to(device, non_blocking=True)
                img_motion_2 = batch["img_motion_2"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                    amp_ctx = torch.amp.autocast(device_type="cuda", enabled=True)
                else:
                    amp_ctx = torch.cuda.amp.autocast(enabled=True)
            else:
                amp_ctx = nullcontext()

            with amp_ctx:
                if motion_tokens is not None:
                    outputs = model(img_spatial, motion_tokens=motion_tokens)
                else:
                    outputs = model(img_spatial, img_motion_1, img_motion_2)
                loss, parts = compute_losses(
                    outputs=outputs,
                    targets=labels,
                    criterion=criterion,
                    fusion_mode=args.fusion_mode,
                    aux_loss_weight=args.aux_loss_weight,
                    fusion_loss_weight=args.fusion_loss_weight,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu())
            running_main += parts["main_loss"]
            running_aux += parts["aux_loss"]
            running_fusion_reg += parts["fusion_reg"]

            if step % args.print_freq == 0 or step == len(train_loader):
                avg_loss = running_loss / step
                avg_main = running_main / step
                avg_aux = running_aux / step
                avg_reg = running_fusion_reg / step
                print(
                    f"[Epoch {epoch:03d}/{args.epochs:03d}] "
                    f"Step {step:04d}/{len(train_loader):04d} "
                    f"loss={avg_loss:.4f} main={avg_main:.4f} aux={avg_aux:.4f} reg={avg_reg:.3e}"
                )

        scheduler.step()

        val_metrics = None
        val_frame_rows: List[Dict] = []
        val_video_rows: List[Dict] = []
        monitor_value: Optional[float] = None
        monitor_name = "neg_train_loss"
        monitor_higher_is_better = True

        if val_loader is not None:
            val_metrics, val_frame_rows, val_video_rows = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                criterion=criterion,
                fusion_mode=args.fusion_mode,
                aux_loss_weight=args.aux_loss_weight,
                fusion_loss_weight=args.fusion_loss_weight,
                threshold=args.cache_val_threshold,
                video_agg_method=args.val_video_agg_method,
                video_agg_topk_ratio=args.val_video_agg_topk_ratio,
                video_agg_topk_min=args.val_video_agg_topk_min,
                video_agg_topk_max=args.val_video_agg_topk_max,
                video_agg_hybrid_alpha=args.val_video_agg_hybrid_alpha,
                motion_token_cache=(motion_token_cache if use_motion_cache_for_val else None),
            )
            print(
                f"[Val][Epoch {epoch:03d}] "
                f"loss={val_metrics['loss']:.4f} frame_acc={val_metrics['frame_acc']:.4f} "
                f"video_acc={val_metrics['video_acc']:.4f} frame_auc={val_metrics['frame_auc']:.4f} "
                f"video_auc={val_metrics['video_auc']:.4f}"
            )
            monitor_value, monitor_name, monitor_higher_is_better = select_monitor_value(
                val_metrics, args.early_stop_metric
            )
            if (
                args.cache_val_predictions
                and args.cache_val_every > 0
                and (epoch % args.cache_val_every == 0)
            ):
                frame_csv, video_csv = save_val_prediction_cache(
                    save_dir=save_dir,
                    epoch=epoch,
                    frame_rows=val_frame_rows,
                    video_rows=val_video_rows,
                )
                print(f"[Cache] Saved validation predictions: {frame_csv} | {video_csv}")
        else:
            train_loss = running_loss / max(1, len(train_loader))
            monitor_value = -train_loss
            monitor_name = "neg_train_loss"
            monitor_higher_is_better = True

        if motion_token_cache is not None:
            print(
                f"[MotionCache][Epoch {epoch:03d}] hit_mem={motion_token_cache.hit_mem} "
                f"hit_disk={motion_token_cache.hit_disk} miss_compute={motion_token_cache.miss_compute} "
                f"bucket={motion_token_cache.cache_bucket}"
            )

        if monitor_value is not None:
            improved = is_improved(
                current=monitor_value,
                best=best_metric,
                higher_is_better=monitor_higher_is_better,
                min_delta=args.early_stop_min_delta,
            )
        else:
            improved = False

        if improved:
            best_metric = monitor_value
            best_metric_name = monitor_name
            best_metric_higher_is_better = monitor_higher_is_better
            early_stop_wait = 0
            best_metric_str = f"{best_metric:.6f}" if best_metric is not None else "None"
            print(f"[Best] Updated best {best_metric_name}: {best_metric_str}")
        elif val_loader is not None:
            early_stop_wait += 1
            current_str = "None" if monitor_value is None else f"{monitor_value:.6f}"
            best_str = "None" if best_metric is None else f"{best_metric:.6f}"
            print(
                f"[EarlyStop] No improvement on {monitor_name}. "
                f"current={current_str}, best={best_str}, "
                f"wait={early_stop_wait}/{args.early_stop_patience}"
            )

        ckpt_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if use_amp else {},
            "best_metric": best_metric,
            "best_metric_name": best_metric_name,
            "best_metric_higher_is_better": best_metric_higher_is_better,
            "early_stop_wait": early_stop_wait,
            "config": vars(args),
            "val_metrics": val_metrics,
            "motion_token_cache_bucket": str(motion_token_cache.cache_bucket) if motion_token_cache is not None else "",
            "motion_token_cache_stats": {
                "hit_mem": int(motion_token_cache.hit_mem),
                "hit_disk": int(motion_token_cache.hit_disk),
                "miss_compute": int(motion_token_cache.miss_compute),
            }
            if motion_token_cache is not None
            else {},
        }

        if epoch % args.save_every == 0:
            save_checkpoint(save_dir / f"epoch_{epoch:03d}.pth", ckpt_payload)
        save_checkpoint(save_dir / "latest.pth", ckpt_payload)

        if improved:
            save_checkpoint(save_dir / "best.pth", ckpt_payload)
        if val_loader is not None and args.early_stop_patience > 0 and early_stop_wait >= args.early_stop_patience:
            print(
                f"[EarlyStop] Triggered at epoch {epoch}. "
                f"Metric={monitor_name}, patience={args.early_stop_patience}."
            )
            break

    best_metric_str = f"{best_metric:.6f}" if best_metric is not None else "None"
    print(
        f"Training finished. Best {best_metric_name or 'metric'}={best_metric_str}. "
        f"Checkpoints in: {save_dir}"
    )


if __name__ == "__main__":
    main()
