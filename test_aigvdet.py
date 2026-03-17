#!/usr/bin/env python3
"""
AIGVDet (PRCV 2024) 复现脚本 —— AI-Generated Video Detection via Spatial-Temporal Anomaly Learning
双分支: RGB ResNet50 + 光流 ResNet50, 融合权重各 0.5。

用法:
    python test_aigvdet.py --threshold 0.5
    python test_aigvdet.py --threshold 0.5 --raft_iters 12

默认复用 ./data/test 下的测试数据（与 test_thesis.py 同一测试集）。
光流在推理时由 RAFT 实时计算，不保存到磁盘。

输出格式与 test_thesis.py 完全一致:
  - test_frame_predictions.csv      帧级预测
  - test_frame_predictions_video.csv 视频级预测
  - test_metrics_per_group.csv      分生成器指标
  - test_metrics_summary.json       汇总 JSON

权重:
  - RGB 分支:     C:/hejulian/aigvdet_original.pth
  - 光流分支:     C:/hejulian/aigvdet_optical.pth
  - RAFT 光流模型: C:/hejulian/raft-things.pth
"""

import argparse
import csv
import json
import sys
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
AIGVDET_DIR = SCRIPT_DIR / "AIGVDet-main"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIGVDet baseline 测试 (光流实时计算)")
    parser.add_argument("--data_root", type=str, default=str(SCRIPT_DIR / "data" / "test"),
                        help="测试集根目录 (默认: ./data/test)")
    parser.add_argument("--real_dir", type=str, default="0_real", help="真实类目录名")
    parser.add_argument("--fake_dir", type=str, default="1_fake", help="伪造类目录名")
    parser.add_argument("--ckpt_rgb", type=str,
                        default=r"C:\hejulian\aigvdet_original.pth",
                        help="RGB 分支权重路径")
    parser.add_argument("--ckpt_optical", type=str,
                        default=r"C:\hejulian\aigvdet_optical.pth",
                        help="光流分支权重路径")
    parser.add_argument("--ckpt_raft", type=str,
                        default=r"C:\hejulian\raft-things.pth",
                        help="RAFT 光流模型权重路径")
    parser.add_argument("--raft_iters", type=int, default=20,
                        help="RAFT 迭代次数 (默认: 20)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="RGB 分支推理批大小")
    parser.add_argument("--output_dir", type=str, default=r"C:\hejulian\exp\aigvdet_baseline",
                        help="结果输出目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="分类阈值")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="每个视频最多采样帧数 (0=全部帧, 推荐16以加速)")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="使用 FP16 推理加速")
    parser.add_argument("--recalc", action="store_true",
                        help="跳过推理, 从已有 CSV 重算指标 (用于更换阈值)")
    parser.add_argument("--rgb_weight", type=float, default=0.5,
                        help="RGB 分支融合权重 (默认 0.5)")
    parser.add_argument("--of_weight", type=float, default=0.5,
                        help="光流分支融合权重 (默认 0.5)")
    return parser.parse_args()


# ============ 加载 AIGVDet 检测器 (ResNet50, 1-class output) ============

def load_aigvdet_resnet(ckpt_path: str, device: torch.device) -> nn.Module:
    """加载 AIGVDet 的 ResNet50 分支 (RGB 或光流)。"""
    aigvdet_core = str(AIGVDET_DIR)
    if aigvdet_core not in sys.path:
        sys.path.insert(0, aigvdet_core)

    from core.utils1.utils import get_network
    model = get_network("resnet50")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


# ============ 加载 RAFT 光流模型 ============

def load_raft_model(ckpt_path: str, device: torch.device):
    """加载 RAFT 光流估计模型。"""
    aigvdet_core = str(AIGVDET_DIR / "core")
    if aigvdet_core not in sys.path:
        sys.path.insert(0, aigvdet_core)

    # RAFT 用 argparse.Namespace 检查属性
    class RAFTArgs:
        def __init__(self):
            self.small = False
            self.mixed_precision = False
            self.alternate_corr = False
            self.dropout = 0

        def __contains__(self, key):
            return hasattr(self, key)

    from raft import RAFT
    raft_args = RAFTArgs()
    model = nn.DataParallel(RAFT(raft_args))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.module
    model.to(device)
    model.eval()
    return model


# ============ 光流计算 (不存盘) ============

def compute_optical_flow_image(raft_model, img1_tensor, img2_tensor, device, iters=20):
    """
    使用 RAFT 计算两帧间的光流, 并转为 RGB 可视化图像 (不存盘)。

    Args:
        raft_model: RAFT 模型
        img1_tensor: [1, 3, H, W] float tensor (0-255)
        img2_tensor: [1, 3, H, W] float tensor (0-255)

    Returns:
        flow_rgb: [H, W, 3] uint8 numpy array (光流可视化)
    """
    aigvdet_core = str(AIGVDET_DIR / "core")
    if aigvdet_core not in sys.path:
        sys.path.insert(0, aigvdet_core)
    from utils.utils import InputPadder
    from utils import flow_viz

    padder = InputPadder(img1_tensor.shape)
    img1_padded, img2_padded = padder.pad(img1_tensor, img2_tensor)

    with torch.no_grad():
        _, flow_up = raft_model(img1_padded, img2_padded, iters=iters, test_mode=True)

    flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
    flow_rgb = flow_viz.flow_to_image(flow_np)  # [H, W, 3] uint8
    return flow_rgb


def load_image_for_raft(img_path: str, device: torch.device, max_size: int = 448) -> torch.Tensor:
    """加载图像为 RAFT 输入格式: [1, 3, H, W] float (0-255)。
    将长边缩放到 max_size 以避免 RAFT 相关体积 OOM。
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        # 确保宽高为8的倍数（RAFT要求）
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        img = img.resize((new_w, new_h), Image.BILINEAR)
    img_np = np.array(img).astype(np.uint8)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    return img_tensor[None].to(device)


# ============ 数据收集 (与其他 baseline 一致) ============

def list_image_files(folder: Path, max_frames: int = 0) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    if max_frames > 0 and len(files) > max_frames:
        indices = np.linspace(0, len(files) - 1, max_frames, dtype=int)
        files = [files[i] for i in indices]
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


# ============ 视频推理 (RGB + 实时光流) ============

# AIGVDet 预处理: CenterCrop(448) + ImageNet normalize
aigvdet_transform = transforms.Compose([
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
])


def aggregate_topk_mean(probs, ratio=0.3, k_min=3, k_max=32, alpha=0.7):
    """Hybrid Top-k Mean 聚合。"""
    if not probs:
        return float("nan")
    arr = np.asarray(probs, dtype=np.float32)
    all_mean = float(np.mean(arr))
    n = len(arr)
    k = int(np.ceil(n * ratio))
    k = max(k_min, k)
    if k_max > 0:
        k = min(k_max, k)
    k = min(n, max(1, k))
    topk_vals = np.sort(arr)[-k:]
    topk_mean = float(np.mean(topk_vals))
    return float(alpha * topk_mean + (1.0 - alpha) * all_mean)


def infer_one_video(
    model_rgb: nn.Module,
    model_of: nn.Module,
    raft_model: nn.Module,
    device: torch.device,
    video_dir: Path,
    batch_size: int = 4,
    raft_iters: int = 10,
    rgb_weight: float = 0.5,
    of_weight: float = 0.5,
    max_frames: int = 0,
    use_fp16: bool = True,
) -> Tuple[float, List[Tuple[str, float]], int]:
    """
    对一个视频推理:
    1. RGB 分支: 逐帧 CenterCrop(448) + ResNet50 → sigmoid
    2. 光流分支: 相邻帧 RAFT → flow_to_image → CenterCrop(448) + ResNet50 → sigmoid
    3. 融合: frame_prob = rgb_weight * rgb_prob + of_weight * of_prob

    Returns: (video_prob, [(frame_path, prob), ...], num_frames)
    """
    frames = list_image_files(video_dir, max_frames=max_frames)
    if not frames:
        return float("nan"), [], 0

    use_amp = use_fp16 and device.type == "cuda"

    # --- RGB 分支: 批量推理 ---
    rgb_probs = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for start in range(0, len(frames), batch_size):
            batch_paths = frames[start:start + batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(str(p)).convert("RGB")
                img_t = aigvdet_transform(img)
                img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                imgs.append(img_t)
            batch = torch.stack(imgs, dim=0).to(device, non_blocking=True)
            probs = model_rgb(batch).float().sigmoid().squeeze(1).cpu().numpy().tolist()
            rgb_probs.extend(probs)

    # --- 光流分支: RAFT 计算光流 → 批量 ResNet50 推理 ---
    of_flow_tensors = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for i in range(len(frames) - 1):
            img1 = load_image_for_raft(str(frames[i]), device)
            img2 = load_image_for_raft(str(frames[i + 1]), device)
            flow_rgb = compute_optical_flow_image(raft_model, img1, img2, device, iters=raft_iters)
            flow_pil = Image.fromarray(flow_rgb)
            flow_t = aigvdet_transform(flow_pil)
            flow_t = TF.normalize(flow_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            of_flow_tensors.append(flow_t)

    # 批量推理光流分支
    of_probs = []
    if of_flow_tensors:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for start in range(0, len(of_flow_tensors), batch_size):
                batch = torch.stack(of_flow_tensors[start:start + batch_size], dim=0).to(device, non_blocking=True)
                probs = model_of(batch).float().sigmoid().squeeze(1).cpu().numpy().tolist()
                of_probs.extend(probs)

    # --- 融合 ---
    if of_probs:
        of_probs_full = of_probs + [of_probs[-1]]
    else:
        of_probs_full = [0.5] * len(frames)

    frame_results = []
    for i, frame_path in enumerate(frames):
        fused_prob = rgb_weight * rgb_probs[i] + of_weight * of_probs_full[i]
        frame_results.append((str(frame_path), float(fused_prob)))

    frame_probs = [p for _, p in frame_results]
    video_prob = aggregate_topk_mean(frame_probs)
    return video_prob, frame_results, len(frame_results)


# ============ 指标计算 (与 test_thesis.py 一致) ============

def compute_auc_ap(labels, probs):
    """用 numpy 手动计算 AUC-ROC 和 AP，避免 sklearn/scipy 版本冲突。"""
    y = np.asarray(labels, dtype=np.float32)
    p = np.asarray(probs, dtype=np.float32)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    try:
        # AUC-ROC via trapezoidal rule
        desc_idx = np.argsort(-p)
        y_sorted = y[desc_idx]
        npos = y.sum()
        nneg = len(y) - npos
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        tpr = tp / npos
        fpr = fp / nneg
        tpr = np.concatenate([[0.0], tpr])
        fpr = np.concatenate([[0.0], fpr])
        auc = float(np.trapz(tpr, fpr))

        # Average Precision
        prec = tp / (tp + fp)
        rec = tp / npos
        prec = np.concatenate([[1.0], prec])
        rec = np.concatenate([[0.0], rec])
        ap = float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))
    except Exception:
        auc, ap = float("nan"), float("nan")
    return auc, ap


def safe_float(v):
    if v is None:
        return float("nan")
    return float(v)


def detailed_metrics(labels_list, probs_list, threshold):
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
    precision_val = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision_val * recall / max(precision_val + recall, 1e-12) if (precision_val + recall) > 0 else 0.0
    real_recall = tn / max(tn + fp, 1)
    balanced_acc = 0.5 * (recall + real_recall)
    youden_j = recall + real_recall - 1.0

    auc_val, ap_val = compute_auc_ap(labels_list, probs_list)

    return {
        "auc": safe_float(auc_val),
        "ap": safe_float(ap_val),
        "acc": float(acc),
        "precision": float(precision_val),
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
                                "AIGVDet (PRCV 2024, RGB + Optical Flow, ResNet50)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    # 加载三个模型
    print(f"[AIGVDet] 加载 RGB 分支: {args.ckpt_rgb}")
    model_rgb = load_aigvdet_resnet(args.ckpt_rgb, device)
    print(f"[AIGVDet] 加载光流分支: {args.ckpt_optical}")
    model_of = load_aigvdet_resnet(args.ckpt_optical, device)
    print(f"[AIGVDet] 加载 RAFT 光流模型: {args.ckpt_raft}")
    raft_model = load_raft_model(args.ckpt_raft, device)
    print(f"[AIGVDet] 所有模型加载完成 (融合权重: RGB={args.rgb_weight}, OF={args.of_weight})")

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

    # --- 断点续传: 读取已完成的 video_id ---
    done_video_ids = set()
    frame_probs_all: List[float] = []
    frame_labels_all: List[float] = []
    video_probs_all: List[float] = []
    video_labels_all: List[float] = []
    group_video_data: Dict[str, Dict[str, List[float]]] = {}
    num_frames_written = 0
    num_videos_written = 0

    if video_csv.exists() and out_csv.exists():
        import pandas as pd
        try:
            df_done = pd.read_csv(video_csv)
            if len(df_done) > 0:
                done_video_ids = set(df_done["video_id"].tolist())
                for _, row in df_done.iterrows():
                    video_probs_all.append(float(row["prob"]))
                    video_labels_all.append(float(row["label"]))
                    gm = row["group_model"]
                    if gm not in group_video_data:
                        group_video_data[gm] = {"probs": [], "labels": []}
                    group_video_data[gm]["probs"].append(float(row["prob"]))
                    group_video_data[gm]["labels"].append(float(row["label"]))
                    num_videos_written += 1
                df_frame_done = pd.read_csv(out_csv)
                for _, row in df_frame_done.iterrows():
                    frame_probs_all.append(float(row["prob"]))
                    frame_labels_all.append(float(row["label"]))
                    num_frames_written += 1
                print(f"[断点续传] 已完成 {len(done_video_ids)} 个视频, 跳过这些视频继续推理")
        except Exception as e:
            print(f"[断点续传] 读取已有 CSV 失败 ({e}), 从头开始")
            done_video_ids = set()
            frame_probs_all, frame_labels_all = [], []
            video_probs_all, video_labels_all = [], []
            group_video_data = {}
            num_frames_written = 0
            num_videos_written = 0

    if done_video_ids:
        frame_open_mode = "a"
        video_open_mode = "a"
    else:
        frame_open_mode = "w"
        video_open_mode = "w"

    # 真实视频推理缓存: key=resolved_path -> (video_prob, frame_results, num_frames)
    real_video_cache: Dict[str, Tuple[float, List[Tuple[str, float]], int]] = {}
    cache_hits = 0

    with open(out_csv, frame_open_mode, newline="", encoding="utf-8") as frame_f, \
         open(video_csv, video_open_mode, newline="", encoding="utf-8") as video_f:
        frame_writer = csv.DictWriter(frame_f, fieldnames=frame_fieldnames)
        video_writer = csv.DictWriter(video_f, fieldnames=video_fieldnames)
        if not done_video_ids:
            frame_writer.writeheader()
            video_writer.writeheader()

        skipped = 0
        for i, sample in enumerate(tqdm(plan, desc="AIGVDet 推理")):
            video_dir = Path(sample["video_dir"])
            label = sample["label"]
            group_model = sample["group_model"]
            role = sample["role"]
            video_id = sample["video_id"]
            source_video = str(video_dir.resolve())

            # 断点续传: 跳过已完成的视频
            if video_id in done_video_ids:
                skipped += 1
                continue

            # 检查缓存 (真实视频会跨组复用)
            cache_key = source_video.lower()
            is_cache_hit = False
            if cache_key in real_video_cache:
                video_prob, frame_results, num_frames = real_video_cache[cache_key]
                is_cache_hit = True
                cache_hits += 1
            else:
                video_prob, frame_results, num_frames = infer_one_video(
                    model_rgb=model_rgb,
                    model_of=model_of,
                    raft_model=raft_model,
                    device=device,
                    video_dir=video_dir,
                    batch_size=args.batch_size,
                    raft_iters=args.raft_iters,
                    rgb_weight=args.rgb_weight,
                    of_weight=args.of_weight,
                    max_frames=args.max_frames,
                    use_fp16=args.fp16,
                )
                # 缓存真实视频结果
                if role == "real":
                    real_video_cache[cache_key] = (video_prob, frame_results, num_frames)

            if num_frames <= 0:
                continue

            video_pred = int(video_prob >= threshold) if np.isfinite(video_prob) else 0
            hit_flag = 1 if is_cache_hit else 0

            # 帧级
            for pth, prob in frame_results:
                pred = int(prob >= threshold)
                row = {
                    "path": pth,
                    "video_id": video_id,
                    "source_video": source_video,
                    "group_model": group_model,
                    "role": role,
                    "label": label,
                    "prob": float(prob),
                    "pred": pred,
                    "cache_hit": hit_flag,
                }
                frame_writer.writerow(row)
                frame_probs_all.append(float(prob))
                frame_labels_all.append(float(label))

            # 视频级
            video_row = {
                "video_id": video_id,
                "source_video": source_video,
                "group_model": group_model,
                "role": role,
                "label": label,
                "prob": float(video_prob),
                "pred": video_pred,
                "num_frames": num_frames,
                "cache_hit": hit_flag,
            }
            video_writer.writerow(video_row)
            video_probs_all.append(float(video_prob))
            video_labels_all.append(float(label))

            if group_model not in group_video_data:
                group_video_data[group_model] = {"probs": [], "labels": []}
            group_video_data[group_model]["probs"].append(float(video_prob))
            group_video_data[group_model]["labels"].append(float(label))

            num_frames_written += num_frames
            num_videos_written += 1

            frame_f.flush()
            video_f.flush()

    if skipped > 0:
        print(f"\n[断点续传] 跳过已完成视频 {skipped} 个, 本次新推理 {len(plan) - skipped} 个")
    print(f"[缓存] 真实视频缓存命中 {cache_hits} 次, 缓存条目 {len(real_video_cache)} 个")

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
        "model": "AIGVDet (PRCV 2024, RGB + Optical Flow, ResNet50)",
        "threshold": threshold,
        "overall_video": overall_video,
        "overall_frame": overall_frame,
        "per_group": {gm["group_model"]: gm for gm in group_metrics_list},
        "num_frames": int(num_frames_written),
        "num_videos": int(num_videos_written),
        "rgb_weight": args.rgb_weight,
        "of_weight": args.of_weight,
        "raft_iters": args.raft_iters,
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
    print(f"  AIGVDet Test Results  (threshold={threshold:.4f}, RGB={args.rgb_weight}, OF={args.of_weight})")
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
