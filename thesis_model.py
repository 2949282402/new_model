import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import ResNet50_Weights, resnet50
except Exception:  # pragma: no cover - fallback for old torchvision
    ResNet50_Weights = None
    from torchvision.models import resnet50


def _torch_load_compat(path: str, map_location: str = "cpu", weights_only: Optional[bool] = None):
    kwargs = {"map_location": map_location}
    if weights_only is not None:
        kwargs["weights_only"] = weights_only
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        # Older torch versions do not support weights_only.
        kwargs.pop("weights_only", None)
        return torch.load(path, **kwargs)


class SRMConv(nn.Module):
    """Fixed SRM (Steganalysis Rich Model) high-pass filters.

    Extracts noise residuals from RGB images using 3 classical SRM kernels.
    All weights are frozen (not learnable) — the filters only expose noise-level
    manipulation traces that are generator-agnostic.
    """

    def __init__(self) -> None:
        super().__init__()
        # 3 classical SRM kernels (5x5), applied identically to each RGB channel.
        q = [
            # Kernel 1: 2nd-order horizontal finite difference
            np.array([
                [0, 0,  0, 0, 0],
                [0, 0,  0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0,  0, 0, 0],
                [0, 0,  0, 0, 0],
            ], dtype=np.float32),
            # Kernel 2: SQUARE 5x5
            np.array([
                [-1,  2, -2,  2, -1],
                [ 2, -6,  8, -6,  2],
                [-2,  8,-12,  8, -2],
                [ 2, -6,  8, -6,  2],
                [-1,  2, -2,  2, -1],
            ], dtype=np.float32) / 12.0,
            # Kernel 3: EDGE 3x3 centred in 5x5
            np.array([
                [0,  0,  0,  0, 0],
                [0, -1,  2, -1, 0],
                [0,  2, -4,  2, 0],
                [0, -1,  2, -1, 0],
                [0,  0,  0,  0, 0],
            ], dtype=np.float32) / 4.0,
        ]
        # Build weight tensor: [out_channels=3, in_channels=3, 5, 5]
        # Each kernel is shared across R/G/B (weight = kernel/3 per channel).
        weight = torch.zeros(len(q), 3, 5, 5)
        for i, kernel in enumerate(q):
            k = torch.from_numpy(kernel)
            for c in range(3):
                weight[i, c] = k / 3.0
        self.conv = nn.Conv2d(3, len(q), kernel_size=5, padding=2, bias=False)
        self.conv.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] -> [B, 3, H, W] noise residual
        return self.conv(x)


class NoiseEncoder(nn.Module):
    """Lightweight CNN that maps SRM noise residuals to a feature vector."""

    def __init__(self, in_channels: int = 3, out_dim: int = 512) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W] -> [B, out_dim]
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class HFRI(nn.Module):
    """
    High-Frequency Residual Injection.
    Supports FFT or DCT domain high-pass filtering.
    """

    def __init__(self, remove_ratio: float = 0.25, mode: str = "fft", use_relu: bool = True):
        super().__init__()
        if mode not in {"fft", "dct"}:
            raise ValueError(f"Unsupported HFRI mode: {mode}. Choose from ['fft', 'dct'].")
        if not (0.0 <= remove_ratio < 1.0):
            raise ValueError("remove_ratio must be in [0, 1).")

        self.remove_ratio = remove_ratio
        self.mode = mode
        self.use_relu = use_relu
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self._dct_cache: Dict[Tuple[int, str, str], torch.Tensor] = {}

    def _get_dct_basis(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (n, str(device), str(dtype))
        if key in self._dct_cache:
            return self._dct_cache[key]

        k = torch.arange(n, device=device, dtype=dtype).view(-1, 1)  # [n, 1]
        i = torch.arange(n, device=device, dtype=dtype).view(1, -1)  # [1, n]
        basis = torch.cos(torch.pi / float(n) * (i + 0.5) * k)  # [n, n]
        basis[0, :] *= (1.0 / torch.sqrt(torch.tensor(float(n), device=device, dtype=dtype)))
        if n > 1:
            basis[1:, :] *= torch.sqrt(torch.tensor(2.0 / float(n), device=device, dtype=dtype))
        self._dct_cache[key] = basis
        return basis

    def _dct2(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        x_bc = x.reshape(b * c, h, w)  # [B*C, H, W]
        c_h = self._get_dct_basis(h, x.device, x.dtype)  # [H, H]
        c_w = self._get_dct_basis(w, x.device, x.dtype)  # [W, W]
        x_dct = torch.matmul(c_h, x_bc)  # [B*C, H, W]
        x_dct = torch.matmul(x_dct, c_w.transpose(0, 1))  # [B*C, H, W]
        return x_dct.reshape(b, c, h, w)

    def _idct2(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        x_bc = x.reshape(b * c, h, w)  # [B*C, H, W]
        c_h = self._get_dct_basis(h, x.device, x.dtype)  # [H, H]
        c_w = self._get_dct_basis(w, x.device, x.dtype)  # [W, W]
        x_idct = torch.matmul(c_h.transpose(0, 1), x_bc)  # [B*C, H, W]
        x_idct = torch.matmul(x_idct, c_w)  # [B*C, H, W]
        return x_idct.reshape(b, c, h, w)

    def _build_fft_highpass_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Mask shape: [1, 1, H, W], 1=keep, 0=remove low frequency center
        _, _, h, w = x.shape
        mask = torch.ones((1, 1, h, w), device=x.device, dtype=x.dtype)
        if self.remove_ratio <= 0:
            return mask

        half_h = max(1, int(h * self.remove_ratio * 0.5))
        half_w = max(1, int(w * self.remove_ratio * 0.5))
        cy, cx = h // 2, w // 2
        mask[:, :, cy - half_h : cy + half_h, cx - half_w : cx + half_w] = 0.0
        return mask

    def _build_dct_highpass_mask(self, x: torch.Tensor) -> torch.Tensor:
        # DCT low frequencies are near top-left corner.
        _, _, h, w = x.shape
        mask = torch.ones((1, 1, h, w), device=x.device, dtype=x.dtype)
        if self.remove_ratio <= 0:
            return mask

        low_h = max(1, int(h * self.remove_ratio))
        low_w = max(1, int(w * self.remove_ratio))
        mask[:, :, :low_h, :low_w] = 0.0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        if self.mode == "fft":
            x_fft = torch.fft.fft2(x, norm="ortho")
            x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))
            hp_mask = self._build_fft_highpass_mask(x)
            x_fft = x_fft * hp_mask
            x_fft = torch.fft.ifftshift(x_fft, dim=(-2, -1))
            out = torch.fft.ifft2(x_fft, norm="ortho").real
        else:
            x_dct = self._dct2(x)
            hp_mask = self._build_dct_highpass_mask(x)
            x_dct = x_dct * hp_mask
            out = self._idct2(x_dct)

        if self.use_relu:
            out = F.relu(out, inplace=True)
        return x + self.alpha * out


class FCL(nn.Module):
    """
    Frequency Convolution Layer:
    FFT -> complex 1x1 conv (real/imag) -> IFFT.
    """

    def __init__(self, channels: int, use_residual: bool = True):
        super().__init__()
        self.real_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.imag_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x_fft = torch.fft.fft2(x, norm="ortho")
        real = self.real_conv(x_fft.real)
        imag = self.imag_conv(x_fft.imag)
        x_ifft = torch.fft.ifft2(torch.complex(real, imag), norm="ortho").real
        out = x + x_ifft if self.use_residual else x_ifft
        out = self.bn(out)
        out = self.act(out)
        return out


class ST_CrossAttention(nn.Module):
    """
    Cross-attention: motion feature queries spatial feature map.
    Q = motion feature [B, C] -> unsqueeze to [B, 1, C]
    K/V = spatial feature map [B, N, C]  (N = H'*W')
    Output: fused feature [B, C], attention weights [B, N]
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q_feat: torch.Tensor,
        kv_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q_feat: [B, C] (motion feature)
        # kv_feat: [B, N, C] (spatial feature map tokens)
        q = self.norm_q(q_feat).unsqueeze(1)  # [B, 1, C]
        kv = self.norm_kv(kv_feat)             # [B, N, C]

        attn_out, attn_weights = self.attn(q, kv, kv, need_weights=True)  # [B, 1, C], [B, 1, N]
        fused = q_feat.unsqueeze(1) + attn_out  # [B, 1, C]
        fused = fused + self.ffn(self.norm_ffn(fused))  # [B, 1, C]
        return fused.squeeze(1), attn_weights.squeeze(1)  # [B, C], [B, N]


class SpatialFreqEncoder(nn.Module):
    """
    Spatial branch based on ResNet50 + HFRI + FCL(Layer2/Layer3).
    Optionally fuses an SRM noise-residual branch for cross-generator robustness.
    """

    def __init__(
        self,
        out_dim: int = 512,
        use_resnet_imagenet: bool = False,
        hfri_mode: str = "fft",
        hfri_remove_ratio: float = 0.25,
        use_srm: bool = False,
        use_hfri: bool = True,
    ):
        super().__init__()
        self.use_srm = use_srm
        self.use_hfri = use_hfri

        if self.use_hfri:
            self.hfri = HFRI(remove_ratio=hfri_remove_ratio, mode=hfri_mode, use_relu=True)

        weights = None
        if use_resnet_imagenet and ResNet50_Weights is not None:
            try:
                weights = ResNet50_Weights.IMAGENET1K_V2
            except Exception:
                weights = ResNet50_Weights.IMAGENET1K_V1
        backbone = resnet50(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2  # out channels: 512
        self.layer3 = backbone.layer3  # out channels: 1024
        self.layer4 = backbone.layer4  # out channels: 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.use_hfri:
            self.fcl2 = FCL(channels=512)
            self.fcl3 = FCL(channels=1024)
        self.proj = nn.Linear(2048, out_dim)
        self.spatial_token_proj = nn.Linear(2048, out_dim)

        # SRM noise-residual branch (optional)
        if self.use_srm:
            self.srm_conv = SRMConv()  # fixed filters, not learnable
            self.noise_encoder = NoiseEncoder(in_channels=3, out_dim=out_dim)
            self.spatial_noise_fuse = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.ReLU(inplace=True),
            )

    def _fuse_noise(self, x_rgb: torch.Tensor, f_rgb: torch.Tensor) -> torch.Tensor:
        """Fuse RGB spatial feature with SRM noise feature."""
        noise_map = self.srm_conv(x_rgb)          # [B, 3, H, W]
        f_noise = self.noise_encoder(noise_map)    # [B, C]
        return self.spatial_noise_fuse(
            torch.cat([f_rgb, f_noise], dim=-1),   # [B, 2C]
        )                                          # [B, C]

    def forward(
        self, x: torch.Tensor, return_map: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [B, 3, H, W]
        x_rgb = x  # keep original for SRM branch
        if self.use_hfri:
            x = self.hfri(x)  # [B, 3, H, W]

        x = self.conv1(x)  # [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [B, 64, H/4, W/4]

        x = self.layer1(x)  # [B, 256, H/4, W/4]
        x = self.layer2(x)  # [B, 512, H/8, W/8]
        if self.use_hfri:
            x = self.fcl2(x)    # [B, 512, H/8, W/8]

        x = self.layer3(x)  # [B, 1024, H/16, W/16]
        if self.use_hfri:
            x = self.fcl3(x)    # [B, 1024, H/16, W/16]

        x = self.layer4(x)  # [B, 2048, H/32, W/32]

        if return_map:
            # Return spatial feature map tokens for cross-attention
            feat_map = x  # [B, 2048, H', W']
            spatial_tokens = feat_map.flatten(2).permute(0, 2, 1)  # [B, H'*W', 2048]
            spatial_tokens = self.spatial_token_proj(spatial_tokens)  # [B, H'*W', C]
            f_spatial = self.avgpool(feat_map).flatten(1)  # [B, 2048]
            f_spatial = self.proj(f_spatial)  # [B, C]
            if self.use_srm:
                f_spatial = self._fuse_noise(x_rgb, f_spatial)
            return spatial_tokens, f_spatial

        x = self.avgpool(x).flatten(1)  # [B, 2048]
        f_spatial = self.proj(x)  # [B, C]
        if self.use_srm:
            f_spatial = self._fuse_noise(x_rgb, f_spatial)
        return f_spatial


class MotionEncoder(nn.Module):
    """
    FlowFormer++ wrapper for motion feature extraction.
    Priority:
    1) Extract cost memory (latent tokens) from FlowFormer memory encoder.
    2) If FlowFormer dependency loading fails, fallback to lightweight conv encoder.
    """

    def __init__(
        self,
        out_dim: int = 512,
        flowformer_repo: str = "./FlowFormerPlusPlus-main",
        flowformer_ckpt: str = "./checkpoints/things.pth",
        require_flowformer: bool = True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.flowformer_repo = flowformer_repo
        self.flowformer_ckpt = flowformer_ckpt  # fixed path required by thesis setting
        self.require_flowformer = require_flowformer

        self.flowformer = None
        self.flowformer_token_dim = None
        self.use_flowformer = False

        self._init_flowformer()
        if self.use_flowformer:
            self.align = nn.Linear(self.flowformer_token_dim, out_dim)
        else:
            if self.require_flowformer:
                raise RuntimeError(
                    "FlowFormer is required but failed to initialize. "
                    "Please check flowformer_repo and flowformer_ckpt."
                )
            self.fallback = nn.Sequential(
                nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.align = nn.Linear(256, out_dim)

    def _init_flowformer(self) -> None:
        repo_path = Path(self.flowformer_repo)
        if not repo_path.is_absolute():
            repo_path = (Path(__file__).resolve().parent / repo_path).resolve()

        ckpt_path = Path(self.flowformer_ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = (Path(__file__).resolve().parent / ckpt_path).resolve()

        if not repo_path.exists():
            msg = f"[MotionEncoder] FlowFormer repo not found: {repo_path}."
            if self.require_flowformer:
                raise FileNotFoundError(msg)
            warnings.warn(msg + " Use fallback encoder.")
            return

        sys.path.insert(0, str(repo_path))
        try:
            from configs.things import get_cfg  # type: ignore
            from core.FlowFormer import build_flowformer  # type: ignore

            cfg = get_cfg()
            # Avoid downloading extra pretrained backbone weights.
            cfg[cfg.transformer].pretrain = False
            model = build_flowformer(cfg)
            token_dim = int(cfg[cfg.transformer].cost_latent_dim)

            if ckpt_path.exists():
                raw_state = _torch_load_compat(str(ckpt_path), map_location="cpu", weights_only=True)
                if isinstance(raw_state, dict) and "state_dict" in raw_state:
                    raw_state = raw_state["state_dict"]
                if isinstance(raw_state, dict) and "model" in raw_state:
                    raw_state = raw_state["model"]

                if isinstance(raw_state, dict):
                    cleaned_state = {}
                    for k, v in raw_state.items():
                        new_key = k[7:] if k.startswith("module.") else k
                        cleaned_state[new_key] = v
                    load_msg = model.load_state_dict(cleaned_state, strict=False)
                    missing_cnt = len(load_msg.missing_keys)
                    unexpected_cnt = len(load_msg.unexpected_keys)
                    if missing_cnt > 0 or unexpected_cnt > 0:
                        msg = (
                            "[MotionEncoder] FlowFormer checkpoint state mismatch: "
                            f"missing_keys={missing_cnt}, unexpected_keys={unexpected_cnt}."
                        )
                        if self.require_flowformer:
                            raise RuntimeError(msg)
                        warnings.warn(msg + " Continue with partial loading.")
                else:
                    msg = (
                        "[MotionEncoder] Unsupported checkpoint format for FlowFormer. "
                        "Will still run with randomly initialized FlowFormer weights."
                    )
                    if self.require_flowformer:
                        raise ValueError(msg)
                    warnings.warn(msg)
            else:
                msg = (
                    f"[MotionEncoder] FlowFormer checkpoint not found at {ckpt_path}. "
                    "Will run with randomly initialized FlowFormer weights."
                )
                if self.require_flowformer:
                    raise FileNotFoundError(msg)
                warnings.warn(msg)

            for p in model.parameters():
                p.requires_grad = False
            model.eval()

            self.flowformer = model
            self.flowformer_token_dim = token_dim
            self.use_flowformer = True
        except Exception as e:
            if self.require_flowformer:
                raise RuntimeError(f"[MotionEncoder] Failed to load FlowFormer: {e}") from e
            warnings.warn(f"[MotionEncoder] Failed to load FlowFormer ({e}). Use fallback encoder.")
            self.flowformer = None
            self.flowformer_token_dim = None
            self.use_flowformer = False
        finally:
            try:
                sys.path.remove(str(repo_path))
            except ValueError:
                pass

    @staticmethod
    def _to_flowformer_range(x: torch.Tensor) -> torch.Tensor:
        # Convert common float ranges into [0, 255] float.
        x = x.float()
        x_min = float(x.detach().amin().cpu())
        x_max = float(x.detach().amax().cpu())

        if x_max <= 1.5 and x_min >= -1.5:
            # [0,1] or [-1,1] -> [0,255]
            if x_min < 0:
                x = (x + 1.0) * 0.5
            x = x.clamp(0.0, 1.0) * 255.0
        else:
            x = x.clamp(0.0, 255.0)
        return x

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep FlowFormer frozen/eval even when outer model switches to train mode.
        if self.flowformer is not None:
            self.flowformer.eval()
        return self

    def _extract_fallback_tokens(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # Fallback token shape: [B, 256]
        x = torch.cat([img1, img2], dim=1)  # [B, 6, H, W]
        x = self.fallback(x)  # [B, 256, H/8, W/8]
        return x.mean(dim=(2, 3))  # [B, 256]

    def extract_motion_tokens(
        self,
        img1: torch.Tensor,
        img2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract frozen motion tokens before `self.align`.
        Output shape:
            FlowFormer mode -> [B, D]
            fallback mode   -> [B, 256]
        """
        if img2 is None:
            img2 = img1

        if self.use_flowformer and self.flowformer is not None:
            with torch.no_grad():
                self.flowformer.eval()

                ff_img1 = self._to_flowformer_range(img1)  # [B, 3, H, W] in [0,255]
                ff_img2 = self._to_flowformer_range(img2)  # [B, 3, H, W] in [0,255]

                ff_img1 = 2.0 * (ff_img1 / 255.0) - 1.0  # [B, 3, H, W]
                ff_img2 = 2.0 * (ff_img2 / 255.0) - 1.0  # [B, 3, H, W]

                context, _ = self.flowformer.context_encoder(ff_img1)  # [B, Cc, H/8, W/8]
                data = {}
                cost_memory, _, _, _ = self.flowformer.memory_encoder(
                    ff_img1, ff_img2, data, context
                )
                # cost_memory: [B*H1*W1, T, D]
                b = ff_img1.size(0)
                _, t, d = cost_memory.shape
                cost_memory = cost_memory.reshape(b, -1, t, d)  # [B, H1*W1, T, D]
                motion_tokens = cost_memory.mean(dim=(1, 2))  # [B, D]
            return motion_tokens

        return self._extract_fallback_tokens(img1, img2)

    def forward(
        self,
        img1: Optional[torch.Tensor] = None,
        img2: Optional[torch.Tensor] = None,
        motion_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # img1/img2: [B, 3, H, W], motion_tokens: [B, D]
        if motion_tokens is None:
            if img1 is None:
                raise ValueError("img1 is required when motion_tokens is None.")
            motion_tokens = self.extract_motion_tokens(img1, img2)

        motion_tokens = motion_tokens.to(dtype=self.align.weight.dtype)
        f_motion = self.align(motion_tokens)  # [B, C]
        return f_motion


class Enhanced_STF_Detector(nn.Module):
    """
    Thesis model:
    - Spatial branch: Freq-aware ResNet50 (HFRI + FCL)
    - Temporal branch: Frozen FlowFormer++ MotionEncoder
    - Fusion mode:
        1) cross_attention: ST Cross-Attention + single head
        2) independent (default): dual independent heads + 0.5/0.5 decision fusion
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 1,
        fusion_mode: str = "independent",
        use_resnet_imagenet: bool = False,
        hfri_mode: str = "fft",
        flowformer_repo: str = "./FlowFormerPlusPlus-main",
        flowformer_ckpt: str = "./checkpoints/things.pth",
        require_flowformer: bool = True,
        head_dropout: float = 0.3,
        freeze_backbone_stages: int = 0,
        use_srm: bool = False,
        use_hfri: bool = True,
        use_temporal: bool = True,
    ):
        super().__init__()
        if fusion_mode not in {"cross_attention", "independent"}:
            raise ValueError(
                f"fusion_mode must be one of ['cross_attention', 'independent'], got {fusion_mode}"
            )

        self.fusion_mode = fusion_mode
        self.feature_dim = feature_dim
        self.use_temporal = use_temporal

        self.spatial_encoder = SpatialFreqEncoder(
            out_dim=feature_dim,
            use_resnet_imagenet=use_resnet_imagenet,
            hfri_mode=hfri_mode,
            hfri_remove_ratio=0.25,
            use_srm=use_srm,
            use_hfri=use_hfri,
        )

        # Freeze early backbone stages to prevent overfitting
        if freeze_backbone_stages > 0:
            self._freeze_backbone(freeze_backbone_stages)

        if self.use_temporal:
            self.motion_encoder = MotionEncoder(
                out_dim=feature_dim,
                flowformer_repo=flowformer_repo,
                flowformer_ckpt=flowformer_ckpt,
                require_flowformer=require_flowformer,
            )

            if self.fusion_mode == "cross_attention":
                self.st_attention = ST_CrossAttention(dim=feature_dim, num_heads=8, dropout=0.1)
                self.fusion_gate = nn.Sequential(
                    nn.Linear(feature_dim * 2, feature_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(head_dropout),
                )
                self.head_fusion = nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Dropout(head_dropout),
                    nn.Linear(feature_dim, num_classes),
                )
            else:
                # AIGVDet-style independent dual heads
                self.head_spatial = nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Dropout(head_dropout),
                    nn.Linear(feature_dim, num_classes),
                )
                self.head_temporal = nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Dropout(head_dropout),
                    nn.Linear(feature_dim, num_classes),
                )
        else:
            # Spatial-only mode: single classification head
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(head_dropout),
                nn.Linear(feature_dim, num_classes),
            )

    def _freeze_backbone(self, stages: int) -> None:
        """Freeze early ResNet stages in spatial_encoder to reduce overfitting.
        stages=1: freeze conv1+bn1, stages=2: +layer1, stages=3: +layer2.
        """
        enc = self.spatial_encoder
        if stages >= 1:
            for p in enc.conv1.parameters():
                p.requires_grad = False
            for p in enc.bn1.parameters():
                p.requires_grad = False
        if stages >= 2:
            for p in enc.layer1.parameters():
                p.requires_grad = False
        if stages >= 3:
            for p in enc.layer2.parameters():
                p.requires_grad = False

    def forward(
        self,
        img_spatial: torch.Tensor,
        img_motion_1: Optional[torch.Tensor] = None,
        img_motion_2: Optional[torch.Tensor] = None,
        motion_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            img_spatial: [B, 3, H, W]
            img_motion_1: [B, 3, H, W], optional
            img_motion_2: [B, 3, H, W], optional
            motion_tokens: [B, D], optional precomputed motion tokens

        Returns:
            cross_attention:
                {'logits': [B,1], 'fusion_loss': scalar, ...}
            independent:
                {'logits': [B,1], 'aux_logits': (logit_s, logit_t)}
        """
        # Spatial-only mode: no temporal branch
        if not self.use_temporal:
            f_spatial = self.spatial_encoder(img_spatial)  # [B, C]
            logit = self.head(f_spatial)  # [B, 1]
            return {"logits": logit}

        if motion_tokens is None:
            if img_motion_1 is None:
                img_motion_1 = img_spatial
            if img_motion_2 is None:
                img_motion_2 = img_motion_1

        # 1) Motion feature extraction (shared by both modes)
        # f_motion: [B, C]
        f_motion = self.motion_encoder(
            img1=img_motion_1,
            img2=img_motion_2,
            motion_tokens=motion_tokens,
        )

        # 2) Fusion mode branch
        if self.fusion_mode == "cross_attention":
            # Spatial encoder with feature map for cross-attention
            # spatial_tokens: [B, H'*W', C], f_spatial: [B, C]
            spatial_tokens, f_spatial = self.spatial_encoder(
                img_spatial, return_map=True,
            )

            # Cross-attention: motion queries spatial feature map
            # f_attended: [B, C], attn_weights: [B, H'*W']
            f_attended, attn_weights = self.st_attention(f_motion, spatial_tokens)

            # Fuse motion-attended feature with pooled spatial feature
            f_final = self.fusion_gate(
                torch.cat([f_attended, f_spatial], dim=-1),
            )
            logit = self.head_fusion(f_final)  # [B, 1]

            # Orthogonality regularization: encourage complementary features
            cos_sim = (
                F.normalize(f_spatial, dim=-1) * F.normalize(f_motion, dim=-1)
            ).sum(dim=-1)
            fusion_loss = cos_sim.pow(2).mean()

            return {
                "logits": logit,
                "fusion_loss": fusion_loss,
                "attn_weights": attn_weights,
            }

        # independent mode (baseline / AIGVDet-style)
        f_spatial = self.spatial_encoder(img_spatial)  # [B, C]
        logit_s = self.head_spatial(f_spatial)   # [B, 1]
        logit_t = self.head_temporal(f_motion)   # [B, 1]
        final_logit = 0.5 * logit_s + 0.5 * logit_t  # [B, 1]

        return {
            "logits": final_logit,
            "aux_logits": (logit_s, logit_t),
        }
