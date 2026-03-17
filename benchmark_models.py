#!/usr/bin/env python3
"""
统一基准测试：计算所有模型的参数量和推理速度。
"""

import sys
import os
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_ITERS = 20
BENCH_ITERS = 100
IMG_SIZE = 224
BASE_DIR = Path(__file__).resolve().parent


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fmt(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


@torch.no_grad()
def measure_fps(model, dummy_input, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    model.eval()
    for _ in range(warmup):
        if isinstance(dummy_input, (list, tuple)):
            model(*dummy_input)
        else:
            model(dummy_input)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        if isinstance(dummy_input, (list, tuple)):
            model(*dummy_input)
        else:
            model(dummy_input)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / iters) * 1000
    fps = iters / elapsed
    return avg_ms, fps


# ═══════ 1. Enhanced-STF Detector (Ours) ═══════
def bench_ours():
    print("\n" + "=" * 60)
    print("1. Enhanced-STF Detector (Ours)")
    print("=" * 60)

    sys.path.insert(0, str(BASE_DIR))
    from thesis_model import Enhanced_STF_Detector

    model = Enhanced_STF_Detector(
        feature_dim=512,
        fusion_mode="cross_attention",
        use_resnet_imagenet=True,
        use_hfri=True,
        use_temporal=True,
        use_srm=False,
        head_dropout=0.3,
        freeze_backbone_stages=2,
        flowformer_repo=str(BASE_DIR / "FlowFormerPlusPlus-main"),
        flowformer_ckpt=str(BASE_DIR / "checkpoints" / "things.pth"),
    ).to(DEVICE).eval()

    total, trainable = count_params(model)
    print(f"  Total params:     {fmt(total)} ({total:,})")
    print(f"  Trainable params: {fmt(trainable)} ({trainable:,})")

    # 分支统计
    sp = sum(p.numel() for p in model.spatial_encoder.parameters())
    mp = sum(p.numel() for p in model.motion_encoder.parameters())
    fp = total - sp - mp
    print(f"  Spatial branch:   {fmt(sp)}")
    print(f"  Motion branch:    {fmt(mp)}")
    print(f"  Fusion+Head:      {fmt(fp)}")

    # 含缓存 token 推理
    dummy_img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    token_dim = model.motion_encoder.flowformer_token_dim or 128
    dummy_tok = torch.randn(1, token_dim, device=DEVICE)
    avg_ms, fps = measure_fps(model, (dummy_img, None, None, dummy_tok))
    print(f"  Inference (cached token): {avg_ms:.2f} ms/frame, {fps:.1f} FPS")

    # 完整推理含 FlowFormer
    dummy2 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    avg_ms2, fps2 = measure_fps(model, (dummy_img, dummy_img, dummy2, None),
                                warmup=5, iters=20)
    print(f"  Inference (full FlowFormer): {avg_ms2:.2f} ms/frame, {fps2:.1f} FPS")

    return total, trainable


# ═══════ 2. UnivFD (CVPR 2023) ═══════
def bench_univfd():
    print("\n" + "=" * 60)
    print("2. UnivFD (CVPR 2023)")
    print("=" * 60)

    p = str(BASE_DIR / "UniversalFakeDetect")
    sys.path.insert(0, p)
    from models import get_model
    model = get_model("CLIP:ViT-L/14")
    ckpt = Path(r"C:\hejulian\fc_weights.pth")
    if ckpt.exists():
        model.fc.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model = model.to(DEVICE).eval()
    if p in sys.path:
        sys.path.remove(p)

    total, trainable = count_params(model)
    print(f"  Total params:     {fmt(total)} ({total:,})")
    print(f"  Trainable params: {fmt(trainable)} ({trainable:,})")

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    avg_ms, fps = measure_fps(model, dummy)
    print(f"  Inference: {avg_ms:.2f} ms/frame, {fps:.1f} FPS")

    return total, trainable


# ═══════ 3. FreqNet (AAAI 2024) ═══════
def bench_freqnet():
    print("\n" + "=" * 60)
    print("3. FreqNet (AAAI 2024)")
    print("=" * 60)

    p = str(BASE_DIR / "FreqNet-DeepfakeDetection-main")
    sys.path.insert(0, p)
    from networks.freqnet import freqnet
    model = freqnet()
    ckpt = Path(r"C:\hejulian\4-classes-freqnet-v2.pth")
    if ckpt.exists():
        sd = torch.load(str(ckpt), map_location="cpu")
        model.load_state_dict(sd.get("model", sd))
    model = model.to(DEVICE).eval()
    if p in sys.path:
        sys.path.remove(p)

    total, trainable = count_params(model)
    print(f"  Total params:     {fmt(total)} ({total:,})")
    print(f"  Trainable params: {fmt(trainable)} ({trainable:,})")

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    avg_ms, fps = measure_fps(model, dummy)
    print(f"  Inference: {avg_ms:.2f} ms/frame, {fps:.1f} FPS")

    return total, trainable


# ═══════ 4. AIDE (ICLR 2025) ═══════
def bench_aide():
    print("\n" + "=" * 60)
    print("4. AIDE (ICLR 2025)")
    print("=" * 60)

    aide_dir = str(BASE_DIR / "AIDE")
    sys.path.insert(0, aide_dir)
    import importlib, types

    # Register the 'models' package so relative imports work
    models_dir = os.path.join(aide_dir, "models")
    models_pkg = types.ModuleType("models")
    models_pkg.__path__ = [models_dir]
    models_pkg.__package__ = "models"
    sys.modules["models"] = models_pkg

    # Load srm_filter_kernel first (dependency of AIDE.py)
    spec_srm = importlib.util.spec_from_file_location(
        "models.srm_filter_kernel",
        os.path.join(models_dir, "srm_filter_kernel.py"),
        submodule_search_locations=[])
    srm_mod = importlib.util.module_from_spec(spec_srm)
    srm_mod.__package__ = "models"
    sys.modules["models.srm_filter_kernel"] = srm_mod
    spec_srm.loader.exec_module(srm_mod)

    # Now load AIDE
    spec = importlib.util.spec_from_file_location("models.AIDE",
        os.path.join(models_dir, "AIDE.py"),
        submodule_search_locations=[])
    aide_mod = importlib.util.module_from_spec(spec)
    aide_mod.__package__ = "models"
    sys.modules["models.AIDE"] = aide_mod
    spec.loader.exec_module(aide_mod)
    AIDEModel = aide_mod.AIDE

    model = AIDEModel(resnet_path=None, convnext_path=None)
    ckpt = Path(r"C:\hejulian\progan_train.pth")
    if ckpt.exists():
        checkpoint = torch.load(str(ckpt), map_location="cpu")
        sd = checkpoint.get("model", checkpoint)
        new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
    model = model.to(DEVICE).eval()
    if aide_dir in sys.path:
        sys.path.remove(aide_dir)

    total, trainable = count_params(model)
    print(f"  Total params:     {fmt(total)} ({total:,})")
    print(f"  Trainable params: {fmt(trainable)} ({trainable:,})")

    # AIDE: forward expects (batch, 5, 3, 256, 256) — 5 DCT patches per image
    dummy = torch.randn(1, 5, 3, 256, 256, device=DEVICE)
    avg_ms, fps = measure_fps(model, dummy, warmup=5, iters=20)
    print(f"  Inference: {avg_ms:.2f} ms/image (5 DCT patches), {fps:.1f} img/s")

    return total, trainable


# ═══════ 5. D3 (ICCV 2025) ═══════
def bench_d3():
    print("\n" + "=" * 60)
    print("5. D3 (ICCV 2025) - Training-Free")
    print("=" * 60)

    # D3 uses XCLIP-base-patch16 encoder; XCLIP forward needs batch_size >= 1
    # and has msg_token issue with batch=1. Use batch=8 for XCLIP speed test.
    from transformers import XCLIPVisionModel
    encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16")
    encoder_name = "XCLIP-base-patch16"
    encoder = encoder.to(DEVICE).eval()

    total, _ = count_params(encoder)
    print(f"  Encoder: {encoder_name}")
    print(f"  Encoder params:   {fmt(total)} ({total:,})")
    print(f"  Trainable params: 0 (training-free)")

    class D3Wrapper(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
        def forward(self, x):
            out = self.enc(pixel_values=x)
            return out.last_hidden_state.mean(dim=1)

    # XCLIP needs batch > 0 for msg_token reshape; use batch=8
    dummy = torch.randn(8, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    wrapper = D3Wrapper(encoder).to(DEVICE).eval()
    avg_ms, fps = measure_fps(wrapper, dummy)
    per_frame_ms = avg_ms / 8
    print(f"  Encoder inference: {per_frame_ms:.2f} ms/frame (batch=8), {8*fps:.1f} FPS")

    return total, 0


# ═══════ 6. AIGVDet (PRCV 2024) ═══════
def bench_aigvdet():
    print("\n" + "=" * 60)
    print("6. AIGVDet (PRCV 2024)")
    print("=" * 60)

    import importlib.util as ilu

    aigvdet_dir = str(BASE_DIR / "AIGVDet-main")

    # Clear cached 'networks' module from FreqNet to avoid clash
    for k in list(sys.modules.keys()):
        if k == "networks" or k.startswith("networks."):
            del sys.modules[k]

    # Load get_network from AIGVDet via importlib to avoid 'core' name clash
    spec_net = ilu.spec_from_file_location(
        "aigvdet_utils",
        os.path.join(aigvdet_dir, "core", "utils1", "utils.py"))
    # Need networks.resnet on sys.path
    sys.path.insert(0, aigvdet_dir)
    net_mod = ilu.module_from_spec(spec_net)
    spec_net.loader.exec_module(net_mod)
    get_network = net_mod.get_network

    # RGB branch
    rgb_model = get_network("resnet50").to(DEVICE).eval()
    rgb_total, rgb_train = count_params(rgb_model)

    # OF branch (same architecture)
    of_model = get_network("resnet50").to(DEVICE).eval()
    of_total, of_train = count_params(of_model)

    # RAFT - load via importlib too
    aigvdet_core = os.path.join(aigvdet_dir, "core")
    sys.path.insert(0, aigvdet_core)
    spec_raft = ilu.spec_from_file_location(
        "aigvdet_raft",
        os.path.join(aigvdet_core, "raft.py"))
    raft_mod = ilu.module_from_spec(spec_raft)
    spec_raft.loader.exec_module(raft_mod)
    RAFT = raft_mod.RAFT

    class RAFTArgs:
        small = False
        mixed_precision = False
        alternate_corr = False
        dropout = 0
        def __contains__(self, key):
            return hasattr(self, key)

    raft_model = nn.DataParallel(RAFT(RAFTArgs()))
    raft_ckpt = Path(r"C:\hejulian\raft-things.pth")
    if raft_ckpt.exists():
        raft_model.load_state_dict(torch.load(str(raft_ckpt), map_location="cpu"))
    raft_model = raft_model.module.to(DEVICE).eval()
    raft_total, _ = count_params(raft_model)

    if aigvdet_dir in sys.path:
        sys.path.remove(aigvdet_dir)
    if aigvdet_core in sys.path:
        sys.path.remove(aigvdet_core)

    total = rgb_total + of_total + raft_total
    trainable = rgb_train + of_train
    print(f"  RGB ResNet50:     {fmt(rgb_total)}")
    print(f"  OF ResNet50:      {fmt(of_total)}")
    print(f"  RAFT:             {fmt(raft_total)}")
    print(f"  Total params:     {fmt(total)} ({total:,})")
    print(f"  Trainable params: {fmt(trainable)} ({trainable:,})")

    # RGB speed (448x448)
    dummy448 = torch.randn(1, 3, 448, 448, device=DEVICE)
    avg_ms_rgb, _ = measure_fps(rgb_model, dummy448)
    print(f"  RGB ResNet50:     {avg_ms_rgb:.2f} ms/frame")

    # RAFT speed
    d1 = torch.randn(1, 3, 448, 448, device=DEVICE)
    d2 = torch.randn(1, 3, 448, 448, device=DEVICE)

    class RAFTWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x1, x2):
            return self.m(x1, x2, iters=20, test_mode=True)

    rw = RAFTWrapper(raft_model).to(DEVICE).eval()
    raft_ms, _ = measure_fps(rw, (d1, d2), warmup=5, iters=20)
    print(f"  RAFT flow:        {raft_ms:.2f} ms/pair")

    # OF ResNet50 speed
    avg_ms_of, _ = measure_fps(of_model, dummy448)
    total_ms = avg_ms_rgb + raft_ms + avg_ms_of
    print(f"  OF ResNet50:      {avg_ms_of:.2f} ms/frame")
    print(f"  Total per frame:  ~{total_ms:.2f} ms, ~{1000/total_ms:.1f} FPS")

    return total, trainable


# ═══════ Main ═══════
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    results = {}
    speed_results = {}

    benchmarks = [
        ("Ours (Enhanced-STF)", bench_ours),
        ("UnivFD", bench_univfd),
        ("FreqNet", bench_freqnet),
        ("AIDE", bench_aide),
        ("D3", bench_d3),
        ("AIGVDet", bench_aigvdet),
    ]

    for name, fn in benchmarks:
        try:
            total, trainable = fn()
            results[name] = (total, trainable)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results[name] = (None, None)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Method':<30s} {'Total Params':>14s} {'Trainable':>14s}")
    print("-" * 60)
    for name, (total, trainable) in results.items():
        if total is not None:
            print(f"{name:<30s} {fmt(total):>14s} {fmt(trainable):>14s}")
        else:
            print(f"{name:<30s} {'ERROR':>14s} {'ERROR':>14s}")
