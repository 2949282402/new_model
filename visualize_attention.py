#!/usr/bin/env python3
"""
注意力热力图可视化脚本 —— 利用 ST_CrossAttention 的注意力权重
生成模型关注伪造痕迹区域的热力图。

用法:
    # 可视化单个视频目录
    python visualize_attention.py --video_dir data/test/1_fake/CogVideoX-5B/video_001

    # 可视化多个视频（自动从测试集采样）
    python visualize_attention.py --num_samples 10

    # 指定输出目录
    python visualize_attention.py --video_dir path/to/video --output_dir C:/hejulian/exp/attention_maps

输出:
    每个视频生成:
    - frame_XXX_attn.png   原图 + 注意力热力图叠加
    - frame_XXX_attn_only.png  纯注意力热力图
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.cm as cm

# 中文字体
for fname in ["SimSun", "SimHei", "Microsoft YaHei", "STSong"]:
    if any(fname in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fname
        break
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="注意力热力图可视化")
    parser.add_argument("--video_dir", type=str, default="",
                        help="单个视频帧目录路径")
    parser.add_argument("--data_root", type=str, default=str(SCRIPT_DIR / "data" / "test"),
                        help="测试集根目录 (用于自动采样)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="自动采样的视频数 (仅当 video_dir 为空时)")
    parser.add_argument("--max_frames", type=int, default=8,
                        help="每个视频最多可视化的帧数")
    parser.add_argument("--checkpoint", type=str,
                        default=r"C:\hejulian\exp\exp8\best.pth",
                        help="模型权重路径")
    parser.add_argument("--image_size", type=int, default=224,
                        help="输入图像尺寸")
    parser.add_argument("--motion_stride", type=int, default=1,
                        help="运动帧偏移量")
    parser.add_argument("--output_dir", type=str,
                        default=r"C:\hejulian\exp\attention_maps",
                        help="输出目录")
    parser.add_argument("--top_k", type=int, default=5,
                        help="只绘制概率最高的 top_k 个连续帧 (默认: 5)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="热力图叠加透明度 (0=纯原图, 1=纯热力图)")
    return parser.parse_args()


# ============ 工具函数 ============

def list_image_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    return files


def collect_video_dirs(root: Path) -> List[Path]:
    out = []
    if not root.exists():
        return out
    for d in sorted(root.rglob("*")):
        if d.is_dir():
            imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            if imgs:
                out.append(d)
    return out


def sample_videos(data_root: Path, num_samples: int) -> List[Tuple[Path, str, int]]:
    """从测试集中采样 fake 和 real 视频。返回 (video_dir, label_str, label_int)"""
    results = []
    fake_root = data_root / "1_fake"
    real_root = data_root / "0_real"

    # 采样 fake 视频
    if fake_root.exists():
        for gen_dir in sorted(fake_root.iterdir()):
            if gen_dir.is_dir():
                vids = collect_video_dirs(gen_dir)
                if vids:
                    results.append((vids[0], f"fake/{gen_dir.name}", 1))
                    if len(results) >= num_samples - 1:
                        break

    # 采样 1 个 real 视频
    if real_root.exists():
        vids = collect_video_dirs(real_root)
        if vids:
            results.append((vids[0], "real", 0))

    return results[:num_samples]


# ============ 模型加载 ============

def load_model(ckpt_path: str, device: torch.device):
    from thesis_model import Enhanced_STF_Detector

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    # 从 checkpoint config 推断参数
    model_cfg = {
        "feature_dim": int(ckpt_cfg.get("feature_dim", 512)),
        "fusion_mode": ckpt_cfg.get("fusion_mode", "cross_attention"),
        "hfri_mode": ckpt_cfg.get("hfri_mode", "fft"),
        "use_srm": bool(ckpt_cfg.get("use_srm", False)),
        "use_hfri": bool(ckpt_cfg.get("use_hfri", True)),
        "use_temporal": bool(ckpt_cfg.get("use_temporal", True)),
        "head_dropout": float(ckpt_cfg.get("head_dropout", 0.3)),
        "flowformer_repo": ckpt_cfg.get("flowformer_repo", "./FlowFormerPlusPlus-main"),
        "flowformer_ckpt": ckpt_cfg.get("flowformer_ckpt", "./checkpoints/things.pth"),
    }

    if model_cfg["fusion_mode"] != "cross_attention":
        print(f"[警告] 模型 fusion_mode={model_cfg['fusion_mode']}，非 cross_attention 模式，"
              "没有注意力权重。将强制切换为 cross_attention 模式。")
        model_cfg["fusion_mode"] = "cross_attention"

    model = Enhanced_STF_Detector(
        feature_dim=model_cfg["feature_dim"],
        fusion_mode=model_cfg["fusion_mode"],
        use_resnet_imagenet=False,
        hfri_mode=model_cfg["hfri_mode"],
        require_flowformer=model_cfg["use_temporal"],
        flowformer_repo=model_cfg["flowformer_repo"],
        flowformer_ckpt=model_cfg["flowformer_ckpt"],
        head_dropout=model_cfg["head_dropout"],
        freeze_backbone_stages=0,
        use_srm=model_cfg["use_srm"],
        use_hfri=model_cfg["use_hfri"],
        use_temporal=model_cfg["use_temporal"],
    ).to(device)

    # 加载权重 (允许 strict=False，因为 fusion_mode 可能改变了)
    msg = model.load_state_dict(state_dict, strict=False)
    if msg.missing_keys:
        print(f"[权重] missing keys: {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        print(f"[权重] unexpected keys: {len(msg.unexpected_keys)}")

    model.eval()
    print(f"[模型] 加载完成, fusion_mode=cross_attention, feature_dim={model_cfg['feature_dim']}")
    return model


# ============ 注意力提取与可视化 ============

def extract_attention(
    model,
    img_spatial: torch.Tensor,
    img_motion_1: torch.Tensor,
    img_motion_2: torch.Tensor,
) -> Tuple[np.ndarray, float]:
    """
    提取注意力权重。
    Returns:
        attn_map: [H, W] 归一化到 [0, 1] 的注意力图
        prob: 检测概率
    """
    with torch.no_grad():
        outputs = model(img_spatial, img_motion_1, img_motion_2)

    logit = outputs["logits"]
    prob = torch.sigmoid(logit).item()
    attn_weights = outputs["attn_weights"]  # [1, N] where N = H'*W'

    # 推断空间分辨率 (224 / 32 = 7)
    N = attn_weights.shape[1]
    h = w = int(N ** 0.5)
    if h * w != N:
        # 非正方形，尝试常见比例
        for hh in range(1, N + 1):
            if N % hh == 0:
                ww = N // hh
                if abs(hh - ww) < abs(h - w):
                    h, w = hh, ww
    attn_map = attn_weights[0].reshape(h, w)  # [H', W']

    # 上采样到原图尺寸
    attn_map = attn_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H', W']
    attn_map = F.interpolate(attn_map, size=(img_spatial.shape[2], img_spatial.shape[3]),
                             mode="bilinear", align_corners=False)
    attn_map = attn_map[0, 0]  # [H, W]

    # 归一化到 [0, 1]
    a_min, a_max = attn_map.min(), attn_map.max()
    if a_max - a_min > 1e-8:
        attn_map = (attn_map - a_min) / (a_max - a_min)
    else:
        attn_map = torch.zeros_like(attn_map)

    return attn_map.cpu().numpy(), prob


def create_heatmap_overlay(
    original_img: np.ndarray,
    attn_map: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成热力图叠加图和纯热力图。
    Args:
        original_img: [H, W, 3] uint8
        attn_map: [H, W] float [0, 1]
        alpha: 叠加透明度
    Returns:
        overlay: [H, W, 3] uint8 — 原图 + 热力图叠加
        heatmap_only: [H, W, 3] uint8 — 纯热力图
    """
    colormap = cm.jet(attn_map)[:, :, :3]  # [H, W, 3] float [0, 1]
    heatmap_only = (colormap * 255).astype(np.uint8)

    original_float = original_img.astype(np.float32) / 255.0
    overlay = (1 - alpha) * original_float + alpha * colormap
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    return overlay, heatmap_only


def _load_frame_pair(
    frame_path: Path,
    motion_frame_path: Path,
    image_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """加载一帧及其运动帧对，返回 (original_np, img_spatial, img_motion_2)。"""
    img_pil = Image.open(str(frame_path)).convert("RGB")
    original_np = np.array(img_pil.resize((image_size, image_size), Image.BILINEAR))
    img_tensor = TF.to_tensor(TF.resize(img_pil, [image_size, image_size]))
    img_spatial = img_tensor.unsqueeze(0).to(device)

    img_motion_pil = Image.open(str(motion_frame_path)).convert("RGB")
    img_motion_tensor = TF.to_tensor(TF.resize(img_motion_pil, [image_size, image_size]))
    img_motion_2 = img_motion_tensor.unsqueeze(0).to(device)
    return original_np, img_spatial, img_motion_2


def _find_top_consecutive(probs: List[float], top_k: int) -> List[int]:
    """
    找到概率最高的帧，然后向两侧扩展为连续窗口，共 top_k 帧。
    """
    n = len(probs)
    if n <= top_k:
        return list(range(n))

    # 找概率最高的帧作为中心
    peak_idx = int(np.argmax(probs))

    # 以 peak 为中心扩展窗口
    half = top_k // 2
    start = peak_idx - half
    start = max(0, start)
    end = start + top_k
    if end > n:
        end = n
        start = max(0, end - top_k)

    return list(range(start, end))


def visualize_video(
    model,
    video_dir: Path,
    output_dir: Path,
    device: torch.device,
    image_size: int = 224,
    motion_stride: int = 1,
    max_frames: int = 8,
    top_k: int = 4,
    alpha: float = 0.5,
    label_str: str = "",
):
    """对一个视频生成注意力热力图，只画概率最高的 top_k 个连续帧。"""
    all_frames = list_image_files(video_dir)
    if not all_frames:
        print(f"  [跳过] 无帧: {video_dir}")
        return

    # 均匀采样用于扫描
    if len(all_frames) > max_frames:
        scan_indices = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int).tolist()
    else:
        scan_indices = list(range(len(all_frames)))
    scan_frames = [all_frames[i] for i in scan_indices]

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 第 1 阶段: 快速扫描全部采样帧的概率 ----
    print(f"  [扫描] {len(scan_frames)} 帧, 寻找概率最高区域...")
    scan_probs = []
    for idx, frame_path in enumerate(scan_frames):
        orig_idx = scan_indices[idx]
        motion_idx = min(len(all_frames) - 1, orig_idx + motion_stride)
        _, img_spatial, img_motion_2 = _load_frame_pair(
            frame_path, all_frames[motion_idx], image_size, device)
        with torch.no_grad():
            outputs = model(img_spatial, img_spatial, img_motion_2)
        prob = torch.sigmoid(outputs["logits"]).item()
        scan_probs.append(prob)

    # ---- 第 2 阶段: 选择 top_k 个连续帧 ----
    top_indices = _find_top_consecutive(scan_probs, top_k)
    peak_prob = max(scan_probs)
    peak_scan_idx = int(np.argmax(scan_probs))
    print(f"  [选帧] 峰值帧 #{scan_indices[peak_scan_idx]} (P={peak_prob:.4f}), "
          f"绘制连续 {len(top_indices)} 帧")

    # ---- 第 3 阶段: 对选中帧提取注意力 ----
    originals = []   # List[np.ndarray]  原图
    attn_maps = []   # List[np.ndarray]  注意力图
    overlays = []    # List[np.ndarray]  叠加图
    drawn_probs = []
    orig_indices = []

    for ti in top_indices:
        frame_path = scan_frames[ti]
        orig_idx = scan_indices[ti]
        motion_idx = min(len(all_frames) - 1, orig_idx + motion_stride)

        original_np, img_spatial, img_motion_2 = _load_frame_pair(
            frame_path, all_frames[motion_idx], image_size, device)

        attn_map, prob = extract_attention(model, img_spatial, img_spatial, img_motion_2)
        overlay, _ = create_heatmap_overlay(original_np, attn_map, alpha=alpha)

        originals.append(original_np)
        attn_maps.append(attn_map)
        overlays.append(overlay)
        drawn_probs.append(prob)
        orig_indices.append(orig_idx)

    k = len(top_indices)

    # ---- 第 4 阶段: 三组图 ----

    # 第 1 组: 连续 RGB 原图拼接
    fig1, axes1 = plt.subplots(1, k, figsize=(4 * k, 4))
    if k == 1:
        axes1 = [axes1]
    for i in range(k):
        axes1[i].imshow(originals[i])
        axes1[i].set_title(f"帧 #{orig_indices[i]}", fontsize=11)
        axes1[i].axis("off")
    fig1.tight_layout(pad=0.5)
    for ext in ("svg", "pdf", "png"):
        kw = {"dpi": 200, "bbox_inches": "tight"} if ext == "png" else {"bbox_inches": "tight"}
        fig1.savefig(str(output_dir / f"row1_rgb.{ext}"), **kw)
    plt.close(fig1)

    # 第 2 组: 连续注意力热力图拼接
    fig2, axes2 = plt.subplots(1, k, figsize=(4 * k, 4))
    if k == 1:
        axes2 = [axes2]
    for i in range(k):
        im = axes2[i].imshow(attn_maps[i], cmap="jet", vmin=0, vmax=1)
        axes2[i].set_title(f"帧 #{orig_indices[i]}", fontsize=11)
        axes2[i].axis("off")
    # 在最右侧加 colorbar
    fig2.subplots_adjust(right=0.92)
    cbar_ax = fig2.add_axes([0.93, 0.15, 0.015, 0.7])
    fig2.colorbar(im, cax=cbar_ax)
    for ext in ("svg", "pdf", "png"):
        kw = {"dpi": 200, "bbox_inches": "tight"} if ext == "png" else {"bbox_inches": "tight"}
        fig2.savefig(str(output_dir / f"row2_heatmap.{ext}"), **kw)
    plt.close(fig2)

    # 第 3 组: RGB + 热力图叠加拼接
    fig3, axes3 = plt.subplots(1, k, figsize=(4 * k, 4))
    if k == 1:
        axes3 = [axes3]
    for i in range(k):
        axes3[i].imshow(overlays[i])
        pred_str = "伪造" if drawn_probs[i] >= 0.5 else "真实"
        axes3[i].set_title(f"#{orig_indices[i]} P={drawn_probs[i]:.3f}", fontsize=11)
        axes3[i].axis("off")
    fig3.tight_layout(pad=0.5)
    for ext in ("svg", "pdf", "png"):
        kw = {"dpi": 200, "bbox_inches": "tight"} if ext == "png" else {"bbox_inches": "tight"}
        fig3.savefig(str(output_dir / f"row3_overlay.{ext}"), **kw)
    plt.close(fig3)

    # ---- 额外: 三行合并大图 (论文用) ----
    fig_all, axes_all = plt.subplots(3, k, figsize=(4 * k, 12))
    if k == 1:
        axes_all = axes_all.reshape(3, 1)
    for i in range(k):
        axes_all[0, i].imshow(originals[i])
        axes_all[0, i].set_title(f"帧 #{orig_indices[i]}", fontsize=11)
        axes_all[0, i].axis("off")

        im = axes_all[1, i].imshow(attn_maps[i], cmap="jet", vmin=0, vmax=1)
        axes_all[1, i].axis("off")

        axes_all[2, i].imshow(overlays[i])
        axes_all[2, i].set_title(f"P={drawn_probs[i]:.3f}", fontsize=11)
        axes_all[2, i].axis("off")

    # 行标签
    axes_all[0, 0].set_ylabel("原始帧", fontsize=13, rotation=90, labelpad=10)
    axes_all[1, 0].set_ylabel("注意力图", fontsize=13, rotation=90, labelpad=10)
    axes_all[2, 0].set_ylabel("叠加图", fontsize=13, rotation=90, labelpad=10)
    for r in range(3):
        axes_all[r, 0].yaxis.set_visible(True)
        axes_all[r, 0].tick_params(left=False, labelleft=False)

    fig_all.tight_layout(pad=1.0)
    for ext in ("svg", "pdf", "png"):
        kw = {"dpi": 200, "bbox_inches": "tight"} if ext == "png" else {"bbox_inches": "tight"}
        fig_all.savefig(str(output_dir / f"combined_3rows.{ext}"), **kw)
    plt.close(fig_all)

    # 汇总
    avg_prob = np.mean(scan_probs)
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"视频: {video_dir}\n")
        f.write(f"标签: {label_str}\n")
        f.write(f"扫描帧数: {len(scan_frames)}\n")
        f.write(f"绘制帧数: {k} (概率最高连续帧)\n")
        f.write(f"平均概率: {avg_prob:.4f}\n")
        f.write(f"峰值概率: {peak_prob:.4f} (帧 #{scan_indices[peak_scan_idx]})\n\n")
        f.write(f"绘制帧详情:\n")
        for i in range(k):
            f.write(f"  #{orig_indices[i]:03d}: P={drawn_probs[i]:.4f}\n")
        f.write(f"\n全部扫描帧概率:\n")
        for i, (si, p) in enumerate(zip(scan_indices, scan_probs)):
            marker = " <<<" if i in top_indices else ""
            f.write(f"  #{si:03d}: {all_frames[si].name}  P={p:.4f}{marker}\n")

    print(f"  已生成 3 组图 + 合并大图 → {output_dir}")
    print(f"  绘制帧: {orig_indices}, 概率: {[f'{p:.4f}' for p in drawn_probs]}")


# ============ 主函数 ============

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    # 加载模型
    print(f"[加载] 模型权重: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    output_root = Path(args.output_dir)

    if args.video_dir:
        # 单个视频
        video_dir = Path(args.video_dir)
        out_dir = output_root / video_dir.name
        print(f"\n[可视化] {video_dir}")
        visualize_video(
            model, video_dir, out_dir, device,
            image_size=args.image_size,
            motion_stride=args.motion_stride,
            max_frames=args.max_frames,
            top_k=args.top_k,
            alpha=args.alpha,
            label_str="",
        )
    else:
        # 自动采样
        data_root = Path(args.data_root)
        videos = sample_videos(data_root, args.num_samples)
        print(f"\n[采样] 从 {data_root} 采样 {len(videos)} 个视频")

        for i, (video_dir, label_str, label_int) in enumerate(videos):
            safe_name = label_str.replace("/", "_").replace("\\", "_")
            out_dir = output_root / f"{i:02d}_{safe_name}_{video_dir.name}"
            print(f"\n[{i+1}/{len(videos)}] {label_str} — {video_dir.name}")
            visualize_video(
                model, video_dir, out_dir, device,
                image_size=args.image_size,
                motion_stride=args.motion_stride,
                max_frames=args.max_frames,
                top_k=args.top_k,
                alpha=args.alpha,
                label_str=label_str,
            )

    print(f"\n[完成] 所有热力图已保存到: {output_root}")


if __name__ == "__main__":
    main()
