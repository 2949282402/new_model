import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from config import get_default_config
from thesis_model import Enhanced_STF_Detector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def infer_class_dirs(root: Path) -> List[Tuple[Path, int]]:
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError(f"No class directories found in {root}")

    mapping: Dict[str, int] = {
        "0_real": 0,
        "real": 0,
        "1_fake": 1,
        "fake": 1,
    }
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
        raise RuntimeError(
            "Unable to infer labels. Use class dirs named 0_real/1_fake (or real/fake), "
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


class DeepfakePairDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 224,
        split: str = "train",
        motion_stride: int = 2,
        max_frames_per_video: int = 0,
    ):
        self.root = Path(root)
        self.split = split
        self.motion_stride = max(1, int(motion_stride))
        self.max_frames_per_video = max(0, int(max_frames_per_video))
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
                    if n == 1:
                        j = 0
                    elif self.split == "train":
                        max_j = min(n - 1, i + self.motion_stride)
                        if max_j <= i:
                            j = n - 1
                        else:
                            j = random.randint(i + 1, max_j)
                    else:
                        j = min(n - 1, i + self.motion_stride)

                    samples.append(
                        {
                            "img1": frame_i,
                            "img2": frames[j],
                            "label": float(label),
                            "video_id": video_id,
                        }
                    )

        if not samples:
            raise RuntimeError(f"No valid image samples found under {self.root}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        img1 = Image.open(sample["img1"]).convert("RGB")
        img2 = Image.open(sample["img2"]).convert("RGB")
        x1, x2 = self.transform(img1, img2)

        return {
            "img_spatial": x1,  # [3,H,W]
            "img_motion_1": x1,  # [3,H,W]
            "img_motion_2": x2,  # [3,H,W]
            "label": torch.tensor(sample["label"], dtype=torch.float32),  # []
            "video_id": sample["video_id"],
            "path": str(sample["img1"]),
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

    fusion_reg = outputs.get("fusion_loss", torch.tensor(0.0, device=logits.device))
    total_loss = main_loss + fusion_loss_weight * fusion_reg
    return total_loss, {
        "main_loss": float(main_loss.detach().cpu()),
        "aux_loss": 0.0,
        "fusion_reg": float(fusion_reg.detach().cpu()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    fusion_mode: str,
    aux_loss_weight: float,
    fusion_loss_weight: float,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    probs_all: List[float] = []
    labels_all: List[float] = []
    video_pool: Dict[str, List[Tuple[float, float]]] = {}

    for batch in loader:
        img_spatial = batch["img_spatial"].to(device, non_blocking=True)
        img_motion_1 = batch["img_motion_1"].to(device, non_blocking=True)
        img_motion_2 = batch["img_motion_2"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)  # [B,1]

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

        for vid, p, y in zip(batch["video_id"], probs_np, labels_np):
            if vid not in video_pool:
                video_pool[vid] = []
            video_pool[vid].append((float(p), float(y)))

    frame_preds = [1.0 if p >= 0.5 else 0.0 for p in probs_all]
    frame_acc = float(np.mean([int(p == y) for p, y in zip(frame_preds, labels_all)])) if labels_all else 0.0

    video_probs: List[float] = []
    video_labels: List[float] = []
    for _, items in video_pool.items():
        p = float(np.mean([it[0] for it in items]))
        y = float(round(np.mean([it[1] for it in items])))
        video_probs.append(p)
        video_labels.append(y)
    video_preds = [1.0 if p >= 0.5 else 0.0 for p in video_probs]
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

    return metrics


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
    parser.add_argument("--save_every", type=int, default=train_cfg["save_every"])
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

    parser.add_argument("--flowformer_repo", type=str, default=model_cfg["flowformer_repo"])
    parser.add_argument("--flowformer_ckpt", type=str, default=model_cfg["flowformer_ckpt"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    train_dataset = DeepfakePairDataset(
        root=args.train_root,
        image_size=args.image_size,
        split="train",
        motion_stride=args.motion_stride,
        max_frames_per_video=args.max_frames_per_video,
    )
    val_dataset: Optional[DeepfakePairDataset] = None
    if args.val_root:
        val_dataset = DeepfakePairDataset(
            root=args.val_root,
            image_size=args.image_size,
            split="val",
            motion_stride=args.motion_stride,
            max_frames_per_video=args.max_frames_per_video,
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
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_metric: Optional[float] = None
    best_metric_name = ""
    best_metric_higher_is_better = True
    early_stop_wait = 0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
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
            img_motion_1 = batch["img_motion_1"].to(device, non_blocking=True)
            img_motion_2 = batch["img_motion_2"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)  # [B,1]

            optimizer.zero_grad(set_to_none=True)
            amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
            with amp_ctx():
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
                    f"loss={avg_loss:.4f} main={avg_main:.4f} aux={avg_aux:.4f} reg={avg_reg:.4f}"
                )

        scheduler.step()

        val_metrics = None
        monitor_value: Optional[float] = None
        monitor_name = "neg_train_loss"
        monitor_higher_is_better = True

        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                criterion=criterion,
                fusion_mode=args.fusion_mode,
                aux_loss_weight=args.aux_loss_weight,
                fusion_loss_weight=args.fusion_loss_weight,
            )
            metric_for_best = val_metrics["video_acc"]
            print(
                f"[Val][Epoch {epoch:03d}] "
                f"loss={val_metrics['loss']:.4f} frame_acc={val_metrics['frame_acc']:.4f} "
                f"video_acc={val_metrics['video_acc']:.4f} frame_auc={val_metrics['frame_auc']:.4f} "
                f"video_auc={val_metrics['video_auc']:.4f}"
            )
            monitor_value, monitor_name, monitor_higher_is_better = select_monitor_value(
                val_metrics, args.early_stop_metric
            )
        else:
            train_loss = running_loss / max(1, len(train_loader))
            monitor_value = -train_loss
            monitor_name = "neg_train_loss"
            monitor_higher_is_better = True

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
