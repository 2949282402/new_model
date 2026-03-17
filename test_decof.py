#!/usr/bin/env python3
"""
DeCoF (arXiv 2024) 复现脚本 —— Detecting AI-Generated Video via Frame Consistency
使用 CLIP ViT-L/14 提取帧特征 + 时序 ViT 判别帧间一致性。

用法:
    python test_decof.py --threshold 0.5
    python test_decof.py --ckpt C:/hejulian/decof_weights.pth --threshold 0.5

默认复用 ./data/test 下的测试数据（与 test_thesis.py 同一测试集）。
需要训练好的时序 ViT 权重 (CLIP backbone 自动下载)。

输出格式与 test_thesis.py 完全一致:
  - test_frame_predictions.csv      帧级预测
  - test_frame_predictions_video.csv 视频级预测
  - test_metrics_per_group.csv      分生成器指标
  - test_metrics_summary.json       汇总 JSON

原理: CLIP 冻结提取帧特征 [8, 768] → 时序 ViT (2层 Transformer)
      学习帧间一致性差异 → 2分类 (real/fake)。
"""

import argparse
import csv
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DECOF_DIR = SCRIPT_DIR / "DeCoF"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 每个视频采样 8 帧 (DeCoF 固定设计)
N_FRAMES = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeCoF baseline 测试")
    parser.add_argument("--data_root", type=str, default=str(SCRIPT_DIR / "data" / "test"),
                        help="测试集根目录 (默认: ./data/test)")
    parser.add_argument("--real_dir", type=str, default="0_real", help="真实类目录名")
    parser.add_argument("--fake_dir", type=str, default="1_fake", help="伪造类目录名")
    parser.add_argument("--ckpt", type=str,
                        default=r"C:\hejulian\decof_weights.pth",
                        help="DeCoF 时序 ViT 权重路径")
    parser.add_argument("--batch_size", type=int, default=16, help="推理批大小")
    parser.add_argument("--output_dir", type=str, default=r"C:\hejulian\exp\decof_baseline",
                        help="结果输出目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="分类阈值")
    return parser.parse_args()


# ============ DeCoF Model Components ============

# --- Temporal ViT (from DeCoF/src/vit.py) ---

from torch import einsum
try:
    from einops import rearrange, repeat
except ImportError:
    print("[错误] 需要安装 einops: pip install einops")
    sys.exit(1)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TemporalViT(nn.Module):
    """DeCoF 时序 ViT: 接收 [B, 8, 768] CLIP 特征, 输出 [B, 2] 分类 logits。"""

    def __init__(self, num_classes=2, num_patches=8, dim=768, depth=2, heads=4,
                 mlp_dim=768, pool='mean', dropout=0.1, emb_dropout=0.1):
        super().__init__()
        assert pool in {'cls', 'mean'}
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        dim_head = dim // heads
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        if x.is_cuda:
            cls_tokens = cls_tokens.cuda()
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


# --- CLIP Backbone (use OpenAI CLIP) ---

def load_clip_model(device):
    """加载 CLIP ViT-L/14 (冻结, 仅做特征提取)。"""
    # 尝试使用 DeCoF 自带的 clip 模块
    decof_src = str(DECOF_DIR / "src")
    if decof_src not in sys.path:
        sys.path.insert(0, decof_src)

    try:
        from clip import clip
        model, _ = clip.load("ViT-L/14", device=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print("[DeCoF] CLIP ViT-L/14 加载完成 (from DeCoF/src/clip)")
        return model
    except Exception as e:
        print(f"[警告] DeCoF clip 模块加载失败: {e}")
        print("[尝试] 使用 pip install clip 或 openai-clip...")
        try:
            import clip
            model, _ = clip.load("ViT-L/14", device=device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            print("[DeCoF] CLIP ViT-L/14 加载完成 (from pip clip)")
            return model
        except Exception as e2:
            print(f"[错误] CLIP 加载失败: {e2}")
            print("请安装: pip install ftfy regex")
            sys.exit(1)


class DeCoFDetector(nn.Module):
    """DeCoF 完整检测器: CLIP (冻结) + Temporal ViT (可训练)。"""

    def __init__(self, clip_model, temporal_vit):
        super().__init__()
        self.clip_model = clip_model
        self.temporal_vit = temporal_vit

    @torch.no_grad()
    def extract_clip_features(self, frames):
        """
        Args:
            frames: [B, 8, 3, 224, 224]
        Returns:
            features: [B, 8, 768]
        """
        b, t, c, h, w = frames.shape
        features = torch.zeros(b, t, 768, device=frames.device)
        for i in range(b):
            img = frames[i]  # [8, 3, 224, 224]
            feat = self.clip_model.encode_image(img)  # [8, 768]
            features[i] = feat.float()
        return features

    def forward(self, frames):
        """
        Args:
            frames: [B, 8, 3, 224, 224]
        Returns:
            logits: [B, 2]
        """
        features = self.extract_clip_features(frames)
        logits = self.temporal_vit(features)
        return logits


# ============ 数据处理 ============

# DeCoF 使用 ImageNet normalization (不是 CLIP normalization)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def sample_n_frames(video_dir: Path, n: int = 8) -> Tuple[List[Path], List[Path]]:
    """
    从视频目录中均匀采样 n 帧。
    Returns:
        selected_frames: 采样的 n 帧路径
        all_frames: 所有帧路径
    """
    all_files = sorted(
        [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )

    if len(all_files) < n:
        return all_files, all_files

    # 均匀采样 (模拟 DeCoF 的 "从前 32 帧中均匀选 8 帧")
    pool = all_files[:min(32, len(all_files))]
    if len(pool) <= n:
        selected = pool
    else:
        indices = np.linspace(0, len(pool) - 1, n, dtype=int).tolist()
        selected = [pool[i] for i in indices]

    return selected, all_files


def load_frames_as_tensor(frame_paths: List[Path]) -> Optional[torch.Tensor]:
    """加载帧图像并堆叠为 [N, 3, 224, 224] tensor。"""
    tensors = []
    for p in frame_paths:
        try:
            img = Image.open(str(p)).convert("RGB")
            tensors.append(test_transform(img))
        except Exception as e:
            print(f"[警告] 帧加载失败 {p}: {e}")
            continue

    if not tensors:
        return None

    return torch.stack(tensors, dim=0)


# ============ 数据收集 (与其他 baseline 脚本一致) ============

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
    out = []
    if not root.exists():
        return out
    for d in sorted(root.rglob("*")):
        if d.is_dir() and is_video_frame_dir(d):
            out.append(d)
    if is_video_frame_dir(root):
        out = [root] + out
    seen = set()
    dedup = []
    for p in out:
        key = str(p.resolve()).lower()
        if key not in seen:
            dedup.append(p)
            seen.add(key)
    return dedup


def build_eval_plan(data_root: Path, real_dir: str, fake_dir: str) -> List[Dict]:
    """构建评估计划, 复用 test_thesis.py 的数据组织方式。"""
    real_root = data_root / real_dir
    fake_root = data_root / fake_dir

    for name in [real_dir, "0_real", "real", "0_reall"]:
        p = data_root / name
        if p.exists():
            real_root = p
            break

    for name in [fake_dir, "1_fake", "fake"]:
        p = data_root / name
        if p.exists():
            fake_root = p
            break

    real_videos = collect_video_dirs(real_root)
    if not real_videos:
        raise RuntimeError(f"未找到真实视频目录: {real_root}")

    fake_groups = []
    first_level = [d for d in sorted(fake_root.iterdir()) if d.is_dir()]
    for d in first_level:
        vids = collect_video_dirs(d)
        if vids:
            fake_groups.append((d.name, vids))
    if not fake_groups:
        vids = collect_video_dirs(fake_root)
        if vids:
            fake_groups.append((fake_root.name, vids))

    if not fake_groups:
        raise RuntimeError(f"未找到伪造视频目录: {fake_root}")

    plan = []
    real_cursor = 0
    for group_name, fake_videos in fake_groups:
        n_fake = len(fake_videos)
        for i, vdir in enumerate(fake_videos):
            plan.append({
                "group_model": group_name, "role": "fake", "label": 1,
                "video_dir": vdir,
                "video_id": f"{group_name}/fake/{i:06d}",
                "source_video": str(vdir),
            })
        for i in range(n_fake):
            real_video = real_videos[real_cursor % len(real_videos)]
            real_cursor += 1
            plan.append({
                "group_model": group_name, "role": "real", "label": 0,
                "video_dir": real_video,
                "video_id": f"{group_name}/real/{i:06d}",
                "source_video": str(real_video),
            })

    return plan


# ============ 指标计算 (与 test_thesis.py 一致) ============

def compute_auc_ap(labels, probs):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.asarray(labels, dtype=np.float32)
    p = np.asarray(probs, dtype=np.float32)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    try:
        auc = float(roc_auc_score(y, p))
        ap = float(average_precision_score(y, p))
    except Exception:
        auc, ap = float("nan"), float("nan")
    return auc, ap


def safe_float(v):
    if v is None:
        return float("nan")
    return float(v)


def detailed_metrics(labels_list, probs_list, threshold):
    """与 test_thesis.py 的 _detailed_metrics 完全一致。"""
    if not labels_list:
        return {k: float("nan") for k in [
            "auc", "ap", "acc", "precision", "recall", "f1",
            "fake_recall", "real_recall", "balanced_acc", "youden_j",
            "tp", "tn", "fp", "fn", "num_fake", "num_real", "num_total",
        ]}
    y = np.asarray(labels_list, dtype=np.float32)
    p = np.asarray(probs_list, dtype=np.float32)
    pred = (p >= threshold).astype(np.int32)
    yi = y.astype(np.int32)

    tp = int(np.sum((pred == 1) & (yi == 1)))
    tn = int(np.sum((pred == 0) & (yi == 0)))
    fp = int(np.sum((pred == 1) & (yi == 0)))
    fn = int(np.sum((pred == 0) & (yi == 1)))

    n_pos = tp + fn
    n_neg = tn + fp
    n_all = n_pos + n_neg

    acc = (tp + tn) / max(n_all, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12) if (precision + recall) > 0 else 0.0
    real_recall = tn / max(tn + fp, 1)
    balanced_acc = 0.5 * (recall + real_recall)
    youden_j = recall + real_recall - 1.0

    auc_val, ap_val = compute_auc_ap(labels_list, probs_list)

    return {
        "auc": safe_float(auc_val),
        "ap": safe_float(ap_val),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fake_recall": float(recall),
        "real_recall": float(real_recall),
        "balanced_acc": float(balanced_acc),
        "youden_j": float(youden_j),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "num_fake": float(n_pos),
        "num_real": float(n_neg),
        "num_total": float(n_all),
    }


# ============ 主函数 ============

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    # 检查权重文件
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[错误] 未找到 DeCoF 权重: {ckpt_path}")
        print(f"[提示] DeCoF 原始权重已丢失 (作者服务器迁移)。")
        print(f"[提示] 请从以下途径获取权重:")
        print(f"  1. AIGVDBench: https://github.com/LongMa-2025/AIGVDBench")
        print(f"  2. 联系 DeCoF 作者")
        print(f"  3. 使用 DeCoF/src/train.py 自行训练")
        print(f"[提示] 获取后放到: {ckpt_path}")
        sys.exit(1)

    # 加载 CLIP backbone (自动下载)
    print("[DeCoF] 加载 CLIP ViT-L/14 backbone...")
    clip_model = load_clip_model(device)

    # 加载时序 ViT
    print(f"[DeCoF] 加载时序 ViT 权重: {ckpt_path}")
    temporal_vit = TemporalViT(
        num_classes=2, num_patches=8, dim=768, depth=2, heads=4,
        mlp_dim=768, pool='mean', dropout=0.1, emb_dropout=0.1
    )

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    # 处理可能的 "module." 前缀 (DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "").replace("net_all.", "")
        new_state_dict[new_key] = v
    temporal_vit.load_state_dict(new_state_dict, strict=False)
    temporal_vit.eval()
    temporal_vit.to(device)
    print("[DeCoF] 时序 ViT 加载完成")

    # 组装完整检测器
    detector = DeCoFDetector(clip_model, temporal_vit)
    detector.eval()
    detector.to(device)

    # 构建评估计划
    data_root = Path(args.data_root)
    plan = build_eval_plan(data_root, args.real_dir, args.fake_dir)
    n_fake = sum(1 for x in plan if x["label"] == 1)
    n_real = sum(1 for x in plan if x["label"] == 0)
    print(f"[数据] 共 {len(plan)} 个视频 (fake={n_fake}, real={n_real})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = args.threshold

    # --- CSV ---
    out_csv = output_dir / "test_frame_predictions.csv"
    video_csv = output_dir / "test_frame_predictions_video.csv"

    frame_fieldnames = ["path", "video_id", "source_video", "group_model", "role", "label", "prob", "pred", "cache_hit"]
    video_fieldnames = ["video_id", "source_video", "group_model", "role", "label", "prob", "pred", "num_frames", "cache_hit"]

    frame_probs_all: List[float] = []
    frame_labels_all: List[float] = []
    video_probs_all: List[float] = []
    video_labels_all: List[float] = []
    group_video_data: Dict[str, Dict[str, List[float]]] = {}
    num_frames_written = 0
    num_videos_written = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as frame_f, \
         open(video_csv, "w", newline="", encoding="utf-8") as video_f:
        frame_writer = csv.DictWriter(frame_f, fieldnames=frame_fieldnames)
        video_writer = csv.DictWriter(video_f, fieldnames=video_fieldnames)
        frame_writer.writeheader()
        video_writer.writeheader()

        with torch.no_grad():
            for i, sample in enumerate(tqdm(plan, desc="DeCoF 推理")):
                video_dir = Path(sample["video_dir"])
                label = sample["label"]
                group_model = sample["group_model"]
                role = sample["role"]
                video_id = sample["video_id"]
                source_video = str(video_dir.resolve())

                # 采样 8 帧
                selected_frames, all_frames = sample_n_frames(video_dir, N_FRAMES)
                frames_tensor = load_frames_as_tensor(selected_frames)

                if frames_tensor is None or len(frames_tensor) == 0:
                    continue

                # 如果不足 8 帧, 重复填充到 8
                if frames_tensor.shape[0] < N_FRAMES:
                    repeats = N_FRAMES // frames_tensor.shape[0] + 1
                    frames_tensor = frames_tensor.repeat(repeats, 1, 1, 1)[:N_FRAMES]

                # [1, 8, 3, 224, 224]
                frames_tensor = frames_tensor.unsqueeze(0).to(device)

                # 推理
                logits = detector(frames_tensor)  # [1, 2]
                prob_fake = float(torch.softmax(logits, dim=1)[0, 1].cpu().item())
                video_pred = int(prob_fake >= threshold)

                num_frames = len(all_frames)

                # 帧级: DeCoF 是视频级方法, 所有帧共享同一概率
                for pth in all_frames:
                    pred = int(prob_fake >= threshold)
                    row = {
                        "path": str(pth),
                        "video_id": video_id,
                        "source_video": source_video,
                        "group_model": group_model,
                        "role": role,
                        "label": label,
                        "prob": float(prob_fake),
                        "pred": pred,
                        "cache_hit": 0,
                    }
                    frame_writer.writerow(row)
                    frame_probs_all.append(float(prob_fake))
                    frame_labels_all.append(float(label))

                # 视频级
                video_row = {
                    "video_id": video_id,
                    "source_video": source_video,
                    "group_model": group_model,
                    "role": role,
                    "label": label,
                    "prob": float(prob_fake),
                    "pred": video_pred,
                    "num_frames": num_frames,
                    "cache_hit": 0,
                }
                video_writer.writerow(video_row)
                video_probs_all.append(float(prob_fake))
                video_labels_all.append(float(label))

                if group_model not in group_video_data:
                    group_video_data[group_model] = {"probs": [], "labels": []}
                group_video_data[group_model]["probs"].append(float(prob_fake))
                group_video_data[group_model]["labels"].append(float(label))

                num_frames_written += num_frames
                num_videos_written += 1

                frame_f.flush()
                video_f.flush()

    # ------------------------------------------------------------------
    #  Overall metrics
    # ------------------------------------------------------------------
    overall_video = detailed_metrics(video_labels_all, video_probs_all, threshold)
    overall_frame = detailed_metrics(frame_labels_all, frame_probs_all, threshold)

    # ------------------------------------------------------------------
    #  Per-group metrics
    # ------------------------------------------------------------------
    group_metrics_list: List[Dict] = []
    for gname in group_video_data:
        gd = group_video_data[gname]
        gm = detailed_metrics(gd["labels"], gd["probs"], threshold)
        gm["group_model"] = gname
        group_metrics_list.append(gm)

    # ------------------------------------------------------------------
    #  Save per-group CSV
    # ------------------------------------------------------------------
    group_csv = output_dir / "test_metrics_per_group.csv"
    group_csv_fields = [
        "group_model", "num_total", "num_fake", "num_real",
        "auc", "ap", "acc", "precision", "recall", "f1",
        "fake_recall", "real_recall", "balanced_acc", "youden_j",
        "tp", "tn", "fp", "fn",
    ]
    with open(group_csv, "w", newline="", encoding="utf-8") as gf:
        gw = csv.DictWriter(gf, fieldnames=group_csv_fields, extrasaction="ignore")
        gw.writeheader()
        for gm in group_metrics_list:
            row_out = {}
            for k in group_csv_fields:
                v = gm.get(k, "")
                if isinstance(v, float) and k not in ("tp", "tn", "fp", "fn", "num_total", "num_fake", "num_real"):
                    row_out[k] = f"{v:.6f}" if np.isfinite(v) else ""
                elif isinstance(v, float):
                    row_out[k] = str(int(v))
                else:
                    row_out[k] = str(v)
            gw.writerow(row_out)
        # OVERALL row
        overall_row = dict(overall_video)
        overall_row["group_model"] = "OVERALL"
        row_out = {}
        for k in group_csv_fields:
            v = overall_row.get(k, "")
            if isinstance(v, float) and k not in ("tp", "tn", "fp", "fn", "num_total", "num_fake", "num_real"):
                row_out[k] = f"{v:.6f}" if np.isfinite(v) else ""
            elif isinstance(v, float):
                row_out[k] = str(int(v))
            else:
                row_out[k] = str(v)
        gw.writerow(row_out)

    # ------------------------------------------------------------------
    #  Save summary JSON
    # ------------------------------------------------------------------
    metrics = {
        "model": "DeCoF (arXiv 2024, CLIP ViT-L/14 + Temporal ViT)",
        "threshold": threshold,
        "overall_video": overall_video,
        "overall_frame": overall_frame,
        "per_group": {gm["group_model"]: gm for gm in group_metrics_list},
        "num_frames": int(num_frames_written),
        "num_videos": int(num_videos_written),
    }

    metrics_json = output_dir / "test_metrics_summary.json"
    with open(metrics_json, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    #  Console table
    # ------------------------------------------------------------------
    def _fmt(v: float, decimals: int = 4) -> str:
        if isinstance(v, float) and np.isfinite(v):
            return f"{v:.{decimals}f}"
        return "N/A"

    header = f"{'Group':<28s} {'#Total':>6s} {'#Fake':>6s} {'#Real':>6s} {'AUC':>7s} {'ACC':>7s} {'F1':>7s} {'Prec':>7s} {'FakeRec':>7s} {'RealRec':>7s}"
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(f"  DeCoF Test Results  (threshold={threshold:.4f})")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for gm in group_metrics_list:
        print(
            f"{gm['group_model']:<28s} "
            f"{int(gm['num_total']):>6d} {int(gm['num_fake']):>6d} {int(gm['num_real']):>6d} "
            f"{_fmt(gm['auc']):>7s} {_fmt(gm['acc']):>7s} {_fmt(gm['f1']):>7s} "
            f"{_fmt(gm['precision']):>7s} {_fmt(gm['fake_recall']):>7s} {_fmt(gm['real_recall']):>7s}"
        )
    print(sep)
    ov = overall_video
    print(
        f"{'OVERALL':<28s} "
        f"{int(ov['num_total']):>6d} {int(ov['num_fake']):>6d} {int(ov['num_real']):>6d} "
        f"{_fmt(ov['auc']):>7s} {_fmt(ov['acc']):>7s} {_fmt(ov['f1']):>7s} "
        f"{_fmt(ov['precision']):>7s} {_fmt(ov['fake_recall']):>7s} {_fmt(ov['real_recall']):>7s}"
    )
    print(f"{'='*len(header)}")
    print(f"\n[Saved] frame-level csv : {out_csv}")
    print(f"[Saved] video-level csv : {video_csv}")
    print(f"[Saved] per-group csv   : {group_csv}")
    print(f"[Saved] metrics summary : {metrics_json}")


if __name__ == "__main__":
    main()
