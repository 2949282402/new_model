#!/usr/bin/env python3
"""
单视频 AI 生成检测推理脚本。

用法:
    python infer.py <视频文件或帧目录路径>
    python infer.py video.mp4
    python infer.py ./frames/video_001/
    python infer.py video.mp4 --checkpoint best.pth --threshold 0.5

默认模型权重 (best.pth) 与本脚本在同一目录下。
"""

import argparse
import io
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单视频 AI 生成检测")
    parser.add_argument("video_path", type=str, help="视频文件路径或已抽帧的目录路径")
    parser.add_argument("--checkpoint", type=str, default=str(SCRIPT_DIR / "best.pth"),
                        help="模型权重路径 (默认: 脚本同目录下 best.pth)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="判定阈值, >= threshold 判为 AI 生成 (默认: 0.5)")
    parser.add_argument("--image_size", type=int, default=224, help="输入图像尺寸 (默认: 224)")
    parser.add_argument("--batch_size", type=int, default=8, help="推理批大小 (默认: 8)")
    parser.add_argument("--every_n", type=int, default=1, help="每隔 N 帧抽取一帧 (默认: 1)")
    parser.add_argument("--max_frames", type=int, default=0, help="最多使用帧数, 0=不限制 (默认: 0)")
    parser.add_argument("--device", type=str, default="", help="推理设备 (默认: 自动选择)")
    return parser.parse_args()


def _torch_load_compat(path: str, map_location: str = "cpu", weights_only: bool = False):
    kwargs = {"map_location": map_location, "weights_only": weights_only}
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(path, **kwargs)


# --------------- 视频抽帧 ---------------

def extract_frames_from_video(video_path: Path, every_n: int = 1) -> Path:
    """将视频文件抽帧到临时目录, 返回帧目录路径。"""
    try:
        import cv2
    except ImportError:
        print("[错误] 输入为视频文件, 需要安装 opencv-python: pip install opencv-python")
        sys.exit(1)

    tmp_dir = Path(tempfile.mkdtemp(prefix="infer_frames_"))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[错误] 无法打开视频文件: {video_path}")
        sys.exit(1)

    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % max(1, every_n) == 0:
            out_path = tmp_dir / f"{saved:05d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        idx += 1
    cap.release()

    if saved == 0:
        print(f"[错误] 视频中未提取到任何帧: {video_path}")
        sys.exit(1)

    print(f"[抽帧] 从视频中提取了 {saved} 帧 (共 {idx} 帧, 间隔 {every_n})")
    return tmp_dir


# --------------- 帧加载与预处理 ---------------

def list_image_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    return files


def preprocess_pair(img1: Image.Image, img2: Image.Image, image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    img1 = TF.resize(img1, [image_size, image_size], interpolation=InterpolationMode.BILINEAR)
    img2 = TF.resize(img2, [image_size, image_size], interpolation=InterpolationMode.BILINEAR)
    return TF.to_tensor(img1), TF.to_tensor(img2)


# --------------- 聚合 ---------------

def aggregate_hybrid_topk_mean(
    probs: List[float],
    topk_ratio: float = 0.3,
    k_min: int = 3,
    k_max: int = 32,
    alpha: float = 0.7,
) -> float:
    if not probs:
        return float("nan")
    arr = np.asarray(probs, dtype=np.float32)
    all_mean = float(np.mean(arr))
    n = len(arr)
    k = int(np.ceil(n * topk_ratio))
    k = max(k_min, k)
    if k_max > 0:
        k = min(k_max, k)
    k = min(n, max(1, k))
    topk_vals = np.sort(arr)[-k:]
    topk_mean = float(np.mean(topk_vals))
    return float(alpha * topk_mean + (1.0 - alpha) * all_mean)


# --------------- 模型加载 ---------------

def load_model(checkpoint_path: str, device: torch.device):
    """加载模型和权重, 返回 (model, model_args)。"""
    sys.path.insert(0, str(SCRIPT_DIR))
    from thesis_model import Enhanced_STF_Detector

    ckpt = _torch_load_compat(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    # 从 checkpoint 配置推断模型参数
    defaults = {
        "fusion_mode": "cross_attention",
        "feature_dim": 512,
        "hfri_mode": "fft",
        "head_dropout": 0.3,
        "use_srm": False,
        "use_hfri": True,
        "use_temporal": True,
        "flowformer_repo": str(SCRIPT_DIR / "FlowFormerPlusPlus-main"),
        "flowformer_ckpt": str(SCRIPT_DIR / "checkpoints" / "things.pth"),
        "require_flowformer": True,
    }
    model_args = {}
    for key, default_val in defaults.items():
        val = ckpt_cfg.get(key, default_val)
        if isinstance(default_val, bool):
            val = bool(val)
        elif isinstance(default_val, int):
            val = int(val)
        elif isinstance(default_val, float):
            val = float(val)
        else:
            val = str(val)
        model_args[key] = val

    # flowformer 路径: 如果 ckpt 中是相对路径, 转为基于脚本目录的绝对路径
    for path_key in ("flowformer_repo", "flowformer_ckpt"):
        p = Path(model_args[path_key])
        if not p.is_absolute():
            model_args[path_key] = str((SCRIPT_DIR / p).resolve())

    model = Enhanced_STF_Detector(
        feature_dim=model_args["feature_dim"],
        fusion_mode=model_args["fusion_mode"],
        use_resnet_imagenet=False,
        hfri_mode=model_args["hfri_mode"],
        require_flowformer=(model_args["require_flowformer"] and model_args["use_temporal"]),
        flowformer_repo=model_args["flowformer_repo"],
        flowformer_ckpt=model_args["flowformer_ckpt"],
        head_dropout=model_args["head_dropout"],
        freeze_backbone_stages=0,
        use_srm=model_args["use_srm"],
        use_hfri=model_args["use_hfri"],
        use_temporal=model_args["use_temporal"],
    ).to(device)

    load_msg = model.load_state_dict(state_dict, strict=True)
    if load_msg.missing_keys or load_msg.unexpected_keys:
        print(f"[警告] 权重加载: missing={len(load_msg.missing_keys)} unexpected={len(load_msg.unexpected_keys)}")

    model.eval()
    print(f"[模型] 已加载: fusion={model_args['fusion_mode']} dim={model_args['feature_dim']} "
          f"temporal={model_args['use_temporal']} hfri={model_args['use_hfri']} srm={model_args['use_srm']}")
    return model, model_args


# --------------- 推理 ---------------

def infer_video(
    model: torch.nn.Module,
    model_args: Dict,
    device: torch.device,
    frame_dir: Path,
    image_size: int = 224,
    batch_size: int = 8,
    max_frames: int = 0,
) -> Tuple[float, List[float]]:
    """对一个视频(帧目录)进行推理, 返回 (视频概率, 帧概率列表)。"""
    frames = list_image_files(frame_dir)
    if max_frames > 0:
        frames = frames[:max_frames]
    if not frames:
        print("[错误] 帧目录中未找到图像文件")
        sys.exit(1)

    n = len(frames)
    use_temporal = model_args.get("use_temporal", True)
    motion_offsets = [1, 2, 4]
    frame_probs: List[float] = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx_list = list(range(start, min(start + batch_size, n)))

            # 加载空间帧
            x1_list = []
            for i in idx_list:
                with Image.open(frames[i]) as img:
                    img = img.convert("RGB")
                x1, _ = preprocess_pair(img, img, image_size)
                x1_list.append(x1)
            x1_batch = torch.stack(x1_list, dim=0).to(device, non_blocking=True)

            if not use_temporal:
                outputs = model(x1_batch)
            else:
                # 多偏移时序集成
                token_sum = None
                for off in motion_offsets:
                    x2_list = []
                    for i in idx_list:
                        j = 0 if n == 1 else min(n - 1, i + off)
                        with Image.open(frames[j]) as img2:
                            img2 = img2.convert("RGB")
                        _, x2 = preprocess_pair(img2, img2, image_size)
                        x2_list.append(x2)
                    x2_batch = torch.stack(x2_list, dim=0).to(device, non_blocking=True)
                    tokens_k = model.motion_encoder.extract_motion_tokens(x1_batch, x2_batch)
                    token_sum = tokens_k if token_sum is None else (token_sum + tokens_k)
                motion_tokens = token_sum / float(len(motion_offsets))
                outputs = model(x1_batch, motion_tokens=motion_tokens)

            probs = torch.sigmoid(outputs["logits"]).squeeze(1).detach().cpu().numpy().tolist()
            frame_probs.extend([float(p) for p in probs])

            # 进度
            done = min(start + batch_size, n)
            print(f"\r[推理] {done}/{n} 帧", end="", flush=True)

    print()  # 换行

    video_prob = aggregate_hybrid_topk_mean(frame_probs)
    return video_prob, frame_probs


# --------------- 主函数 ---------------

def main():
    args = parse_args()
    video_path = Path(args.video_path)

    if not video_path.exists():
        print(f"[错误] 路径不存在: {video_path}")
        sys.exit(1)

    # 选择设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    # 判断输入是视频文件还是帧目录
    tmp_dir = None
    if video_path.is_file():
        if video_path.suffix.lower() in VIDEO_EXTS:
            frame_dir = extract_frames_from_video(video_path, every_n=args.every_n)
            tmp_dir = frame_dir
        elif video_path.suffix.lower() in IMAGE_EXTS:
            # 单张图片 → 用其所在目录
            frame_dir = video_path.parent
            print(f"[提示] 输入为单张图片, 使用所在目录: {frame_dir}")
        else:
            print(f"[错误] 不支持的文件格式: {video_path.suffix}")
            sys.exit(1)
    elif video_path.is_dir():
        frame_dir = video_path
    else:
        print(f"[错误] 无法识别的路径类型: {video_path}")
        sys.exit(1)

    # 加载模型
    if not Path(args.checkpoint).exists():
        print(f"[错误] 模型权重不存在: {args.checkpoint}")
        sys.exit(1)
    model, model_args = load_model(args.checkpoint, device)

    # 推理
    video_prob, frame_probs = infer_video(
        model=model,
        model_args=model_args,
        device=device,
        frame_dir=frame_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
    )

    # 清理临时目录
    if tmp_dir is not None:
        import shutil
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    # 输出结果
    label = "AI 生成" if video_prob >= args.threshold else "真实"
    print()
    print("=" * 50)
    print(f"  视频路径:   {video_path}")
    print(f"  分析帧数:   {len(frame_probs)}")
    print(f"  AI生成概率: {video_prob:.4f} ({video_prob * 100:.2f}%)")
    print(f"  判定阈值:   {args.threshold}")
    print(f"  检测结果:   {label}")
    print("=" * 50)


if __name__ == "__main__":
    main()
