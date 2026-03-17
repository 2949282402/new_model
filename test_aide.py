#!/usr/bin/env python3
"""
AIDE (ICLR 2025) 复现脚本 —— 在本项目测试集上评估。

用法:
    python test_aide.py

默认复用 ./data/test 下的测试数据（与 test_thesis.py 同一测试集）。

权重下载:
    从 https://drive.google.com/drive/folders/1qx76UFvDpgCxaPLBCmsA2WY-SSzeJrd4
    下载 checkpoint-best.pth, 放到 ./AIDE/pretrained_weights/checkpoint-best.pth

依赖:
    pip install open_clip_torch kornia

输出格式与 test_thesis.py 完全一致:
  - test_frame_predictions.csv      帧级预测
  - test_frame_predictions_video.csv 视频级预测
  - test_metrics_per_group.csv      分生成器指标
  - test_metrics_summary.json       汇总 JSON
"""

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from scipy.special import softmax


class RealVideoFileCache:
    """基于 SQLite 的真实视频推理缓存，结果写入磁盘，不占用内存。"""

    def __init__(self, cache_path: Path):
        self.path = cache_path
        self._conn = sqlite3.connect(str(cache_path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                cache_key  TEXT PRIMARY KEY,
                video_prob REAL,
                frame_json TEXT,
                num_frames INTEGER
            )
        """)
        self._conn.commit()

    def __contains__(self, key: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM cache WHERE cache_key=?", (key,))
        return cur.fetchone() is not None

    def get(self, key: str) -> Optional[Tuple]:
        cur = self._conn.execute(
            "SELECT video_prob, frame_json, num_frames FROM cache WHERE cache_key=?", (key,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        video_prob, frame_json, num_frames = row
        frame_results = [(p, f) for p, f in json.loads(frame_json)]
        return video_prob, frame_results, num_frames

    def set(self, key: str, video_prob: float, frame_results: list, num_frames: int):
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (cache_key, video_prob, frame_json, num_frames) "
            "VALUES (?,?,?,?)",
            (key, video_prob, json.dumps(frame_results), num_frames)
        )
        self._conn.commit()

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

    def close(self):
        self._conn.close()

SCRIPT_DIR = Path(__file__).resolve().parent
AIDE_DIR = SCRIPT_DIR / "AIDE"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIDE baseline 测试")
    parser.add_argument("--data_root", type=str, default=str(SCRIPT_DIR / "data" / "test"),
                        help="测试集根目录 (默认: ./data/test)")
    parser.add_argument("--real_dir", type=str, default="0_real", help="真实类目录名")
    parser.add_argument("--fake_dir", type=str, default="1_fake", help="伪造类目录名")
    parser.add_argument("--ckpt", type=str,
                        default=r"C:\hejulian\progan_train.pth",
                        help="AIDE 权重路径")
    parser.add_argument("--batch_size", type=int, default=32, help="推理批大小")
    parser.add_argument("--output_dir", type=str, default=r"C:\hejulian\exp\aide_baseline",
                        help="结果输出目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="分类阈值")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="每个视频最多采样帧数 (0=全部帧, 推荐16以加速)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader 预处理并行数")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="使用 FP16 推理加速")
    parser.add_argument("--recalc", action="store_true",
                        help="跳过推理, 从已有 CSV 重算指标 (用于更换阈值)")
    return parser.parse_args()


# ============ 模型加载 ============

def load_aide_model(ckpt_path: str, device: torch.device) -> nn.Module:
    """加载 AIDE 模型。"""
    sys.path.insert(0, str(AIDE_DIR))
    import models.AIDE as AIDE_module
    model = AIDE_module.AIDE(resnet_path=None, convnext_path=None)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    # 去除 DDP 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model.to(device)
    sys.path.remove(str(AIDE_DIR))
    print(f"[AIDE] 模型加载完成: {ckpt_path}")
    return model


# ============ DCT 预处理 ============

def load_dct_module():
    """加载 AIDE 的 DCT 模块。"""
    sys.path.insert(0, str(AIDE_DIR))
    from data.dct import DCT_base_Rec_Module
    dct = DCT_base_Rec_Module()
    sys.path.remove(str(AIDE_DIR))
    return dct


transform_normalize = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(img_tensor: torch.Tensor, dct_module) -> torch.Tensor:
    """
    AIDE 预处理: ToTensor 后的图像 -> DCT patch 选择 -> 5通道堆叠
    输入: img_tensor [3, H, W] (已 ToTensor)
    输出: [5, 3, 256, 256]
    """
    x_minmin, x_maxmax, x_minmin1, x_maxmax1 = dct_module(img_tensor)
    x_0 = transform_normalize(img_tensor)
    x_minmin = transform_normalize(x_minmin)
    x_maxmax = transform_normalize(x_maxmax)
    x_minmin1 = transform_normalize(x_minmin1)
    x_maxmax1 = transform_normalize(x_maxmax1)
    return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0)


# ============ 数据收集 ============

def list_image_files(folder: Path, max_frames: int = 0) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    if max_frames > 0 and len(files) > max_frames:
        # 均匀采样
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


# ============ 推理 ============

def aggregate_topk_mean(probs, ratio=0.3, k_min=3, k_max=32, alpha=0.7):
    """Hybrid Top-k Mean 聚合, 与本项目保持一致。"""
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
    model: nn.Module,
    device: torch.device,
    video_dir: Path,
    dct_module,
    batch_size: int = 32,
    max_frames: int = 0,
    num_workers: int = 0,
    use_fp16: bool = True,
) -> Tuple[float, List[Tuple[str, float]], int]:
    """对一个视频的所有帧推理, 返回 (video_prob, [(frame_path, prob), ...], num_frames)。"""
    frames = list_image_files(video_dir, max_frames=max_frames)
    if not frames:
        return float("nan"), [], 0

    to_tensor = transforms.ToTensor()
    frame_results: List[Tuple[str, float]] = []

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_fp16 and device.type == "cuda"):
        for start in range(0, len(frames), batch_size):
            batch_paths = frames[start:start + batch_size]
            batch_tensors = []
            valid_paths = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    img_tensor = to_tensor(img)
                    preprocessed = preprocess_image(img_tensor, dct_module)  # [5, 3, 256, 256]
                    batch_tensors.append(preprocessed)
                    valid_paths.append(p)
                except Exception as e:
                    continue

            if not batch_tensors:
                continue

            batch = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
            logits = model(batch)
            probs = softmax(logits.float().cpu().numpy(), axis=1)[:, 1]

            for p_path, prob in zip(valid_paths, probs):
                frame_results.append((str(p_path), float(prob)))

    frame_probs = [p for _, p in frame_results]
    video_prob = aggregate_topk_mean(frame_probs)
    return video_prob, frame_results, len(frame_results)


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
        recalc_metrics_from_csv(Path(args.output_dir), args.threshold, "AIDE (ICLR 2025)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    model = load_aide_model(args.ckpt, device)
    dct_module = load_dct_module()

    data_root = Path(args.data_root)
    plan = build_eval_plan(data_root, args.real_dir, args.fake_dir)
    n_fake = sum(1 for x in plan if x["label"] == 1)
    n_real = sum(1 for x in plan if x["label"] == 0)
    print(f"[数据] 共 {len(plan)} 个视频 (fake={n_fake}, real={n_real})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = args.threshold

    # --- CSV 文件名与 test_thesis.py 一致 ---
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

    if video_csv.exists() and out_csv.exists():
        import pandas as pd
        try:
            df_done = pd.read_csv(video_csv)
            if len(df_done) > 0:
                done_video_ids = set(df_done["video_id"].tolist())
                # 恢复已有数据到内存
                for _, row in df_done.iterrows():
                    video_probs_all.append(float(row["prob"]))
                    video_labels_all.append(float(row["label"]))
                    gm = row["group_model"]
                    if gm not in group_video_data:
                        group_video_data[gm] = {"probs": [], "labels": []}
                    group_video_data[gm]["probs"].append(float(row["prob"]))
                    group_video_data[gm]["labels"].append(float(row["label"]))
                df_frame_done = pd.read_csv(out_csv)
                for _, row in df_frame_done.iterrows():
                    frame_probs_all.append(float(row["prob"]))
                    frame_labels_all.append(float(row["label"]))
                print(f"[断点续传] 已完成 {len(done_video_ids)} 个视频, 跳过这些视频继续推理")
        except Exception as e:
            print(f"[断点续传] 读取已有 CSV 失败 ({e}), 从头开始")
            done_video_ids = set()
            frame_probs_all, frame_labels_all = [], []
            video_probs_all, video_labels_all = [], []
            group_video_data = {}

    # 决定文件打开模式
    if done_video_ids:
        frame_open_mode = "a"
        video_open_mode = "a"
    else:
        frame_open_mode = "w"
        video_open_mode = "w"

    # 真实视频推理缓存: 写入磁盘，不占内存
    real_video_cache = RealVideoFileCache(output_dir / "real_video_cache.db")
    print(f"[文件缓存] 已加载 SQLite 缓存: {output_dir / 'real_video_cache.db'} "
          f"(已有 {len(real_video_cache)} 条)")
    cache_hits = 0

    with open(out_csv, frame_open_mode, newline="", encoding="utf-8") as frame_f, \
         open(video_csv, video_open_mode, newline="", encoding="utf-8") as video_f:
        frame_writer = csv.DictWriter(frame_f, fieldnames=frame_fieldnames)
        video_writer = csv.DictWriter(video_f, fieldnames=video_fieldnames)
        if not done_video_ids:
            frame_writer.writeheader()
            video_writer.writeheader()

        skipped = 0
        for i, sample in enumerate(tqdm(plan, desc="推理中")):
            video_dir = sample["video_dir"]
            label = sample["label"]
            group_model = sample["group_model"]
            role = sample["role"]
            video_id = sample["video_id"]
            source_video = sample["source_video"]

            # 断点续传: 跳过已完成的视频
            if video_id in done_video_ids:
                skipped += 1
                continue

            # 检查文件缓存 (真实视频会跨组复用)
            cache_key = str(Path(video_dir).resolve()).lower()
            is_cache_hit = False
            cached = real_video_cache.get(cache_key) if role == "real" else None
            if cached is not None:
                video_prob, frame_results, num_frames = cached
                is_cache_hit = True
                cache_hits += 1
            else:
                video_prob, frame_results, num_frames = infer_one_video(
                    model, device, video_dir, dct_module, args.batch_size,
                    max_frames=args.max_frames, num_workers=args.num_workers,
                    use_fp16=args.fp16,
                )
                if role == "real":
                    real_video_cache.set(cache_key, video_prob, frame_results, num_frames)

            if num_frames == 0:
                print(f"[{i:04d}/{len(plan):04d}] {group_model} {role} skip (empty video): {source_video}")
                continue

            video_pred = 1 if video_prob >= threshold else 0
            hit_flag = 1 if is_cache_hit else 0

            # 写帧级 CSV
            for fpath, fprob in frame_results:
                fpred = 1 if fprob >= threshold else 0
                frame_writer.writerow({
                    "path": fpath,
                    "video_id": video_id,
                    "source_video": source_video,
                    "group_model": group_model,
                    "role": role,
                    "label": label,
                    "prob": float(fprob),
                    "pred": int(fpred),
                    "cache_hit": hit_flag,
                })
                frame_probs_all.append(float(fprob))
                frame_labels_all.append(float(label))

            # 写视频级 CSV
            video_writer.writerow({
                "video_id": video_id,
                "source_video": source_video,
                "group_model": group_model,
                "role": role,
                "label": label,
                "prob": float(video_prob),
                "pred": int(video_pred),
                "num_frames": int(num_frames),
                "cache_hit": hit_flag,
            })
            frame_f.flush()
            video_f.flush()

            video_probs_all.append(float(video_prob))
            video_labels_all.append(float(label))
            if group_model not in group_video_data:
                group_video_data[group_model] = {"probs": [], "labels": []}
            group_video_data[group_model]["probs"].append(float(video_prob))
            group_video_data[group_model]["labels"].append(float(label))

    cache_total = len(real_video_cache)
    real_video_cache.close()

    if skipped:
        print(f"\n[断点续传] 跳过已完成视频 {skipped} 个, 本次新推理 {len(plan) - skipped} 个")

    print(f"\n[文件缓存] 真实视频缓存命中 {cache_hits} 次, 缓存条目 {cache_total} 个")

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
    #  Save per-group CSV (与 test_thesis.py 一致)
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
    #  Save summary JSON (与 test_thesis.py 一致)
    # ------------------------------------------------------------------
    metrics = {
        "model": "AIDE (ICLR 2025)",
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

    # ------------------------------------------------------------------
    #  Console output (与 test_thesis.py 一致的表格格式)
    # ------------------------------------------------------------------
    def _fmt(v: float, decimals: int = 4) -> str:
        if isinstance(v, float) and np.isfinite(v):
            return f"{v:.{decimals}f}"
        return "N/A"

    header = f"{'Group':<28s} {'#Total':>6s} {'#Fake':>6s} {'#Real':>6s} {'AUC':>7s} {'ACC':>7s} {'F1':>7s} {'Prec':>7s} {'FakeRec':>7s} {'RealRec':>7s}"
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(f"  Test Results  (threshold={threshold:.4f})")
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
