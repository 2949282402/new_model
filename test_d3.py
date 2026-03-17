#!/usr/bin/env python3
"""
D3 (ICCV 2025) 复现脚本 —— Training-Free AI-Generated Video Detection
使用二阶时序特征（Second-Order Features）检测 AI 生成视频。

用法:
    python test_d3.py
    python test_d3.py --encoder XCLIP-16 --loss l2
    python test_d3.py --encoder CLIP-16 --loss cos

默认复用 ./data/test 下的测试数据（与 test_thesis.py 同一测试集）。
无需训练权重，直接使用预训练编码器（自动从 HuggingFace 下载）。

输出格式与 test_thesis.py 完全一致:
  - test_frame_predictions.csv      帧级预测
  - test_frame_predictions_video.csv 视频级预测
  - test_metrics_per_group.csv      分生成器指标
  - test_metrics_summary.json       汇总 JSON

原理: AI 生成视频的帧间特征变化更均匀（二阶标准差低），
      真实视频帧间特征变化更不规则（二阶标准差高）。
      收集所有视频的 dis_2nd_std 后，通过 min-max 归一化转为 fake_prob。
"""

import argparse
import csv
import json
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
D3_DIR = SCRIPT_DIR / "D3"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D3 baseline 测试 (Training-Free)")
    parser.add_argument("--data_root", type=str, default=str(SCRIPT_DIR / "data" / "test"),
                        help="测试集根目录 (默认: ./data/test)")
    parser.add_argument("--real_dir", type=str, default="0_real", help="真实类目录名")
    parser.add_argument("--fake_dir", type=str, default="1_fake", help="伪造类目录名")
    parser.add_argument("--encoder", type=str, default="XCLIP-16",
                        choices=["CLIP-16", "CLIP-32", "XCLIP-16", "XCLIP-32",
                                 "DINO-base", "DINO-large", "ResNet-18"],
                        help="编码器类型 (默认: XCLIP-16)")
    parser.add_argument("--loss", type=str, default="l2", choices=["l2", "cos"],
                        help="距离度量方式 (默认: l2)")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="每个视频采样帧数 (默认: 16, 最少 8)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="推理批大小 (D3 按视频处理, 默认 1)")
    parser.add_argument("--output_dir", type=str, default=r"C:\hejulian\exp\d3_baseline",
                        help="结果输出目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="分类阈值")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="使用 FP16 推理以减少显存占用")
    parser.add_argument("--recalc", action="store_true",
                        help="跳过推理, 从已有 CSV 重算指标 (用于更换阈值)")
    return parser.parse_args()


# ============ D3 Model (from D3 repo) ============

Transformers_list = ['CLIP-16', 'CLIP-32', 'XCLIP-16', 'XCLIP-32', 'DINO-base', 'DINO-large']


class D3Model(nn.Module):
    """D3 模型: 使用预训练编码器提取帧特征, 计算二阶时序统计量。"""

    def __init__(self, encoder_type='XCLIP-16', loss_type='l2'):
        super().__init__()
        self.loss_type = loss_type
        self.encoder_type = encoder_type

        from transformers import CLIPVisionModel, XCLIPVisionModel, AutoModel
        import torchvision.models as models

        if encoder_type == 'CLIP-16':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        elif encoder_type == 'CLIP-32':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        elif encoder_type == 'XCLIP-16':
            self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16")
        elif encoder_type == 'XCLIP-32':
            self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")
        elif encoder_type == 'DINO-base':
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-base")
        elif encoder_type == 'DINO-large':
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-large")
        elif encoder_type == 'ResNet-18':
            resnet18 = models.resnet18(pretrained=True)
            modules = list(resnet18.children())[:-1]
            self.encoder = nn.Sequential(*modules).eval()
        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")

    def forward(self, x):
        """
        Args:
            x: [B, T, 3, H, W] 视频帧序列
        Returns:
            outputs: [B, T, D] 帧特征
            dis_2nd_avg: [B] 二阶差分均值
            dis_2nd_std: [B] 二阶差分标准差 (检测分数)
        """
        b, t, _, h, w = x.shape
        images = x.reshape(-1, 3, h, w)

        if self.encoder_type in Transformers_list:
            # output_hidden_states=False: 不保存所有层的中间状态，节省大量显存
            outputs = self.encoder(images)
            outputs = outputs.pooler_output
        else:
            outputs = self.encoder(images)

        outputs = outputs.reshape(b, t, -1)

        vec1 = outputs[:, :-1, :]
        vec2 = outputs[:, 1:, :]

        if self.loss_type == 'cos':
            dis_1st = F.cosine_similarity(vec1, vec2, dim=-1)
        elif self.loss_type == 'l2':
            dis_1st = torch.norm(vec1 - vec2, p=2, dim=-1)

        dis_2nd = dis_1st[:, 1:] - dis_1st[:, :-1]
        dis_2nd_avg = torch.mean(dis_2nd, dim=1)
        dis_2nd_std = torch.std(dis_2nd, dim=1)

        return outputs, dis_2nd_avg, dis_2nd_std


# ============ 数据处理 ============

def crop_center_by_percentage(image, percentage=0.1):
    """中心裁剪, 去除长边两侧各 percentage 比例的像素。"""
    height, width = image.shape[:2]
    if width > height:
        left = int(width * percentage)
        right = width - int(width * percentage)
        return image[:, left:right]
    else:
        top = int(height * percentage)
        bottom = height - int(height * percentage)
        return image[top:bottom, :]


def get_number_from_filename(filename):
    match = re.match(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')


def load_video_frames(video_dir: Path, max_frames: int = 16) -> Tuple[Optional[torch.Tensor], List[str]]:
    """
    加载视频帧并预处理。
    Returns:
        frames_tensor: [1, T, 3, 224, 224] or None if < 8 frames
        frame_paths: 选中的帧路径列表
    """
    all_files = sorted(
        [f for f in os.listdir(str(video_dir))
         if Path(f).suffix.lower() in IMAGE_EXTS],
        key=get_number_from_filename
    )

    if len(all_files) < 8:
        return None, [str(video_dir / f) for f in all_files]

    # 选择帧数: < 16 用 8, >= 16 用 max_frames
    set_frame = 8 if len(all_files) < 16 else min(max_frames, len(all_files))

    # 均匀采样
    if set_frame >= len(all_files):
        selected_indices = list(range(len(all_files)))
    else:
        selected_indices = np.linspace(0, len(all_files) - 1, set_frame, dtype=int).tolist()

    selected_files = [all_files[i] for i in selected_indices]
    selected_paths = [str(video_dir / f) for f in selected_files]

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    frames = []
    for fname in selected_files:
        fpath = str(video_dir / fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = crop_center_by_percentage(img, 0.1)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std  # BGR order but ImageNet norm
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        frames.append(img)

    if len(frames) < 8:
        return None, selected_paths

    frames_np = np.stack(frames, axis=0)  # [T, 3, 224, 224]
    frames_tensor = torch.from_numpy(frames_np).unsqueeze(0)  # [1, T, 3, 224, 224]
    return frames_tensor, selected_paths


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


# ============ 从已有 CSV 重算指标 ============

def recalc_metrics_from_csv(output_dir: Path, threshold: float, model_name: str):
    """从已有的 video/frame CSV 读取 prob, 用新阈值重算所有指标。"""
    import pandas as pd

    video_csv = output_dir / "test_frame_predictions_video.csv"
    frame_csv = output_dir / "test_frame_predictions.csv"

    if not video_csv.exists():
        print(f"[错误] 未找到视频级 CSV: {video_csv}")
        return
    print(f"[Recalc] 从已有 CSV 重算指标, 新阈值={threshold}")

    df_video = pd.read_csv(video_csv)
    video_probs_all = df_video["prob"].tolist()
    video_labels_all = df_video["label"].tolist()

    group_video_data: Dict[str, Dict[str, List[float]]] = {}
    for _, row in df_video.iterrows():
        gm = row["group_model"]
        if gm not in group_video_data:
            group_video_data[gm] = {"probs": [], "labels": []}
        group_video_data[gm]["probs"].append(float(row["prob"]))
        group_video_data[gm]["labels"].append(float(row["label"]))

    frame_probs_all, frame_labels_all = [], []
    if frame_csv.exists():
        df_frame = pd.read_csv(frame_csv)
        frame_probs_all = df_frame["prob"].tolist()
        frame_labels_all = df_frame["label"].tolist()

    overall_video = detailed_metrics(video_labels_all, video_probs_all, threshold)
    overall_frame = detailed_metrics(frame_labels_all, frame_probs_all, threshold)

    group_metrics_list: List[Dict] = []
    for gname in group_video_data:
        gd = group_video_data[gname]
        gm_metrics = detailed_metrics(gd["labels"], gd["probs"], threshold)
        gm_metrics["group_model"] = gname
        group_metrics_list.append(gm_metrics)

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

    metrics = {
        "model": model_name,
        "threshold": threshold,
        "overall_video": overall_video,
        "overall_frame": overall_frame,
        "per_group": {gm["group_model"]: gm for gm in group_metrics_list},
        "num_frames": int(len(frame_probs_all)),
        "num_videos": int(len(video_probs_all)),
    }
    metrics_json = output_dir / "test_metrics_summary.json"
    with open(metrics_json, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, ensure_ascii=False, indent=2)

    def _fmt(v: float, decimals: int = 4) -> str:
        if isinstance(v, float) and np.isfinite(v):
            return f"{v:.{decimals}f}"
        return "N/A"

    header = f"{'Group':<28s} {'#Total':>6s} {'#Fake':>6s} {'#Real':>6s} {'AUC':>7s} {'ACC':>7s} {'F1':>7s} {'Prec':>7s} {'FakeRec':>7s} {'RealRec':>7s}"
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(f"  Recalc Results  (threshold={threshold:.4f})")
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
    print(f"\n[Saved] per-group csv   : {group_csv}")
    print(f"[Saved] metrics summary : {metrics_json}")


# ============ 主函数 ============

def main():
    args = parse_args()

    if args.recalc:
        recalc_metrics_from_csv(Path(args.output_dir), args.threshold,
                                f"D3 (ICCV 2025, Training-Free, {args.encoder}, {args.loss})")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    # 加载 D3 模型 (预训练编码器, 无需额外权重)
    print(f"[D3] 加载编码器: {args.encoder}, 距离度量: {args.loss}")
    print(f"[D3] 编码器将从 HuggingFace 自动下载 (首次运行需要网络)")
    model = D3Model(encoder_type=args.encoder, loss_type=args.loss)
    model.eval()
    model.to(device)
    print(f"[D3] 模型加载完成 (Training-Free, 无需任务特定权重)")

    data_root = Path(args.data_root)
    plan = build_eval_plan(data_root, args.real_dir, args.fake_dir)
    n_fake = sum(1 for x in plan if x["label"] == 1)
    n_real = sum(1 for x in plan if x["label"] == 0)
    print(f"[数据] 共 {len(plan)} 个视频 (fake={n_fake}, real={n_real})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = args.threshold

    # ------------------------------------------------------------------
    #  断点续传: 读取已有的 checkpoint 文件
    # ------------------------------------------------------------------
    checkpoint_file = output_dir / "d3_phase1_checkpoint.jsonl"
    # 只在内存保存 plan_idx -> raw_score (不含路径列表，节省内存)
    done_index: Dict[int, float] = {}

    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as cf:
            for line in cf:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done_index[rec["plan_idx"]] = rec["raw_score"]
                except Exception:
                    pass
        print(f"[断点续传] 已加载 {len(done_index)} 个视频的 Phase1 结果")

    # ------------------------------------------------------------------
    #  第一遍: 收集所有视频的 raw score (dis_2nd_std)
    #  D3 的 score 越高 = 越可能是真实视频, 需要归一化反转
    # ------------------------------------------------------------------
    print("\n[Phase 1] 提取所有视频的二阶时序特征...")

    # 只存 (plan_idx, raw_score)，不存路径列表——路径在 Phase 2 按需重扫
    video_raw_scores: List[Tuple[int, float]] = []
    skipped = 0

    # 真实视频推理缓存: key=resolved_path -> raw_score (仅分数，不含路径列表)
    real_video_cache: Dict[str, float] = {}
    cache_hits = 0

    # 推理精度
    infer_dtype = torch.float16 if args.fp16 else torch.float32
    if args.fp16:
        model = model.half()
        print("[FP16] 使用半精度推理以减少显存占用")

    ckpt_f = open(checkpoint_file, "a", encoding="utf-8")

    try:
        with torch.no_grad():
            for i, sample in enumerate(tqdm(plan, desc="D3 推理")):
                video_dir = Path(sample["video_dir"])
                role = sample["role"]

                # 断点续传: 跳过已完成的
                if i in done_index:
                    raw_score = done_index[i]
                    video_raw_scores.append((i, raw_score))
                    if role == "real":
                        cache_key = str(video_dir.resolve()).lower()
                        real_video_cache[cache_key] = raw_score
                    cache_hits += 1
                    continue

                # 检查推理缓存 (真实视频会跨组复用)
                cache_key = str(video_dir.resolve()).lower()
                if cache_key in real_video_cache:
                    raw_score = real_video_cache[cache_key]
                    video_raw_scores.append((i, raw_score))
                    ckpt_f.write(json.dumps({"plan_idx": i, "raw_score": raw_score}) + "\n")
                    ckpt_f.flush()
                    cache_hits += 1
                    continue

                frames_tensor, _ = load_video_frames(video_dir, max_frames=args.num_frames)

                if frames_tensor is None:
                    video_raw_scores.append((i, float("nan")))
                    if role == "real":
                        real_video_cache[cache_key] = float("nan")
                    ckpt_f.write(json.dumps({"plan_idx": i, "raw_score": None}) + "\n")
                    ckpt_f.flush()
                    skipped += 1
                    continue

                frames_tensor = frames_tensor.to(device, dtype=infer_dtype)
                with torch.amp.autocast("cuda", enabled=args.fp16):
                    _, _, dis_2nd_std = model(frames_tensor)
                raw_score = float(dis_2nd_std.float().cpu().item())

                video_raw_scores.append((i, raw_score))

                if role == "real":
                    real_video_cache[cache_key] = raw_score

                ckpt_f.write(json.dumps({"plan_idx": i, "raw_score": raw_score}) + "\n")
                ckpt_f.flush()
    finally:
        ckpt_f.close()

    print(f"\n[缓存] 断点续传/缓存命中 {cache_hits} 次, 推理缓存条目 {len(real_video_cache)} 个")

    if skipped > 0:
        print(f"[警告] {skipped} 个视频帧数不足 8, 已跳过")

    # ------------------------------------------------------------------
    #  归一化: raw_score -> fake_prob
    #  D3 原理: 真实视频 std 高, AI 生成视频 std 低
    #  所以 fake_prob = 1 - (score - min) / (max - min)
    # ------------------------------------------------------------------
    valid_scores = [s for _, s in video_raw_scores if np.isfinite(s)]
    if not valid_scores:
        print("[错误] 没有有效的视频分数, 退出")
        return

    score_min = min(valid_scores)
    score_max = max(valid_scores)
    score_range = score_max - score_min if score_max > score_min else 1e-8
    print(f"\n[Phase 2] 分数归一化: min={score_min:.6f}, max={score_max:.6f}, range={score_range:.6f}")

    def normalize_to_fake_prob(raw_score):
        if not np.isfinite(raw_score):
            return 0.5  # 无效视频给中间值
        normalized = (raw_score - score_min) / score_range
        return float(1.0 - normalized)  # 反转: 低 std -> 高 fake_prob

    # ------------------------------------------------------------------
    #  第二遍: 写入 CSV 和计算指标
    # ------------------------------------------------------------------
    print(f"[Phase 3] 写入结果和计算指标...")

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

        for idx, (plan_idx, raw_score) in enumerate(video_raw_scores):
            sample = plan[plan_idx]
            video_dir = Path(sample["video_dir"])
            label = sample["label"]
            group_model = sample["group_model"]
            role = sample["role"]
            video_id = sample["video_id"]
            source_video = str(video_dir.resolve())

            fake_prob = normalize_to_fake_prob(raw_score)
            video_pred = int(fake_prob >= threshold)

            # 按需重扫目录获取路径（不在内存中长期保存）
            all_paths = [str(p) for p in list_image_files(video_dir)]
            num_frames = len(all_paths)

            if num_frames <= 0:
                continue

            # 帧级: D3 是视频级方法, 所有帧共享同一概率
            for pth in all_paths:
                pred = int(fake_prob >= threshold)
                row = {
                    "path": pth,
                    "video_id": video_id,
                    "source_video": source_video,
                    "group_model": group_model,
                    "role": role,
                    "label": label,
                    "prob": float(fake_prob),
                    "pred": pred,
                    "cache_hit": 0,
                }
                frame_writer.writerow(row)
                frame_probs_all.append(float(fake_prob))
                frame_labels_all.append(float(label))

            # 视频级
            video_row = {
                "video_id": video_id,
                "source_video": source_video,
                "group_model": group_model,
                "role": role,
                "label": label,
                "prob": float(fake_prob),
                "pred": video_pred,
                "num_frames": num_frames,
                "cache_hit": 0,
            }
            video_writer.writerow(video_row)
            video_probs_all.append(float(fake_prob))
            video_labels_all.append(float(label))

            if group_model not in group_video_data:
                group_video_data[group_model] = {"probs": [], "labels": []}
            group_video_data[group_model]["probs"].append(float(fake_prob))
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
        "model": f"D3 (ICCV 2025, Training-Free, {args.encoder}, {args.loss})",
        "threshold": threshold,
        "overall_video": overall_video,
        "overall_frame": overall_frame,
        "per_group": {gm["group_model"]: gm for gm in group_metrics_list},
        "num_frames": int(num_frames_written),
        "num_videos": int(num_videos_written),
        "encoder": args.encoder,
        "loss_type": args.loss,
        "num_frames_per_video": args.num_frames,
        "score_min": float(score_min),
        "score_max": float(score_max),
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
    print(f"  D3 Test Results  (encoder={args.encoder}, loss={args.loss}, threshold={threshold:.4f})")
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
    # 清理断点续传文件 (推理全部完成)
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"[断点续传] 推理完成, 已删除 checkpoint 文件")

    print(f"\n[Saved] frame-level csv : {out_csv}")
    print(f"[Saved] video-level csv : {video_csv}")
    print(f"[Saved] per-group csv   : {group_csv}")
    print(f"[Saved] metrics summary : {metrics_json}")


if __name__ == "__main__":
    main()
