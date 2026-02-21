import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import get_default_config


def parse_args() -> argparse.Namespace:
    cfg = get_default_config()
    t_cfg = cfg["threshold_search"]

    parser = argparse.ArgumentParser("Search threshold range from validation predictions")
    parser.add_argument("--val_cache_dir", type=str, default=t_cfg["val_cache_dir"])
    parser.add_argument("--input_csv", type=str, default=t_cfg["input_csv"])
    parser.add_argument("--level", type=str, default=t_cfg["level"], choices=["video", "frame"])
    parser.add_argument("--epoch", type=int, default=t_cfg["epoch"])

    parser.add_argument("--threshold_min", type=float, default=t_cfg["threshold_min"])
    parser.add_argument("--threshold_max", type=float, default=t_cfg["threshold_max"])
    parser.add_argument("--threshold_step", type=float, default=t_cfg["threshold_step"])

    parser.add_argument(
        "--opt_metric",
        type=str,
        default=t_cfg["opt_metric"],
        choices=["f1", "acc", "balanced_acc", "youden_j", "precision", "recall"],
    )
    parser.add_argument("--near_best_delta", type=float, default=t_cfg["near_best_delta"])
    parser.add_argument("--min_precision", type=float, default=t_cfg["min_precision"])
    parser.add_argument("--min_recall", type=float, default=t_cfg["min_recall"])
    parser.add_argument("--topk", type=int, default=t_cfg["topk"])

    parser.add_argument("--curve_output_csv", type=str, default=t_cfg["curve_output_csv"])
    parser.add_argument("--summary_output_json", type=str, default=t_cfg["summary_output_json"])
    return parser.parse_args()


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (probs >= threshold).astype(np.int32)
    y = labels.astype(np.int32)

    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    tnr = _safe_div(tn, tn + fp)
    balanced_acc = 0.5 * (recall + tnr)
    youden_j = recall + tnr - 1.0

    return {
        "threshold": float(threshold),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_acc": float(balanced_acc),
        "youden_j": float(youden_j),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _find_csv_by_epoch(val_cache_dir: Path, level: str, epoch: int) -> Path:
    suffix = f"_{level}.csv"

    if epoch > 0:
        target = val_cache_dir / f"epoch_{epoch:03d}{suffix}"
        if not target.exists():
            raise FileNotFoundError(f"Specified epoch csv not found: {target}")
        return target

    pattern = re.compile(rf"^epoch_(\d+){re.escape(suffix)}$")
    candidates: List[Tuple[int, Path]] = []
    for p in val_cache_dir.glob(f"epoch_*{suffix}"):
        m = pattern.match(p.name)
        if not m:
            continue
        candidates.append((int(m.group(1)), p))

    if not candidates:
        raise FileNotFoundError(f"No csv files found in {val_cache_dir} for level={level}")

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _load_labels_probs(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    labels: List[float] = []
    probs: List[float] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"label", "prob"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(f"{csv_path} must include columns: {sorted(required)}")
        for row in reader:
            try:
                y = float(row["label"])
                p = float(row["prob"])
            except Exception:
                continue
            if not math.isfinite(y) or not math.isfinite(p):
                continue
            labels.append(y)
            probs.append(p)

    if not labels:
        raise RuntimeError(f"No valid rows loaded from {csv_path}")

    y_np = np.asarray(labels, dtype=np.float32)
    p_np = np.asarray(probs, dtype=np.float32)
    return y_np, p_np


def _build_threshold_grid(t_min: float, t_max: float, t_step: float) -> np.ndarray:
    if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0 and t_min <= t_max):
        raise ValueError("threshold_min/max must satisfy: 0<=min<=max<=1")
    if t_step <= 0:
        raise ValueError("threshold_step must be > 0")
    n = int(np.floor((t_max - t_min) / t_step)) + 1
    grid = t_min + np.arange(n, dtype=np.float64) * t_step
    if grid.size == 0 or grid[-1] < t_max - 1e-12:
        grid = np.append(grid, t_max)
    return np.clip(grid, t_min, t_max)


def main() -> None:
    args = parse_args()

    if args.input_csv:
        input_csv = Path(args.input_csv)
        if not input_csv.exists():
            raise FileNotFoundError(f"input_csv not found: {input_csv}")
    else:
        input_csv = _find_csv_by_epoch(Path(args.val_cache_dir), args.level, int(args.epoch))

    labels, probs = _load_labels_probs(input_csv)
    grid = _build_threshold_grid(float(args.threshold_min), float(args.threshold_max), float(args.threshold_step))

    rows: List[Dict[str, float]] = []
    for t in grid.tolist():
        m = _compute_metrics(labels, probs, t)
        if m["precision"] + 1e-12 < float(args.min_precision):
            continue
        if m["recall"] + 1e-12 < float(args.min_recall):
            continue
        rows.append(m)

    if not rows:
        raise RuntimeError("No threshold satisfies the constraints. Relax min_precision/min_recall.")

    metric_name = str(args.opt_metric)
    rows_sorted = sorted(rows, key=lambda x: (x[metric_name], -abs(x["threshold"] - 0.5)), reverse=True)
    best = rows_sorted[0]

    best_score = float(best[metric_name])
    delta = max(0.0, float(args.near_best_delta))
    near_rows = [r for r in rows if float(r[metric_name]) >= best_score - delta]
    near_rows_sorted = sorted(near_rows, key=lambda x: x["threshold"])
    range_low = float(near_rows_sorted[0]["threshold"])
    range_high = float(near_rows_sorted[-1]["threshold"])

    curve_out = Path(args.curve_output_csv)
    curve_out.parent.mkdir(parents=True, exist_ok=True)
    with open(curve_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threshold", "acc", "precision", "recall", "f1", "balanced_acc", "youden_j", "tp", "tn", "fp", "fn"],
        )
        writer.writeheader()
        for r in sorted(rows, key=lambda x: x["threshold"]):
            writer.writerow(r)

    # Compute threshold-independent AUC metric.
    auc_value = float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels.tolist())) >= 2:
            auc_value = float(roc_auc_score(labels, probs))
    except Exception:
        pass

    # Best threshold full metrics breakdown.
    best_all_metrics = {
        k: float(best[k])
        for k in ["threshold", "acc", "precision", "recall", "f1", "balanced_acc", "youden_j"]
    }
    best_all_metrics["tp"] = int(best["tp"])
    best_all_metrics["tn"] = int(best["tn"])
    best_all_metrics["fp"] = int(best["fp"])
    best_all_metrics["fn"] = int(best["fn"])

    summary = {
        "input_csv": str(input_csv),
        "num_samples": int(labels.shape[0]),
        "positive_ratio": float(np.mean(labels > 0.5)),
        "auc": auc_value,
        "opt_metric": metric_name,
        "best_threshold": float(best["threshold"]),
        "best_metric_value": best_score,
        "best_all_metrics": best_all_metrics,
        "recommended_range": [range_low, range_high],
        "near_best_delta": delta,
        "constraints": {
            "min_precision": float(args.min_precision),
            "min_recall": float(args.min_recall),
        },
        "topk": rows_sorted[: max(1, int(args.topk))],
    }

    summary_out = Path(args.summary_output_json)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"[Input] {input_csv}")
    print(f"[Samples] N={labels.shape[0]}  positive(fake)={int(np.sum(labels > 0.5))}  negative(real)={int(np.sum(labels <= 0.5))}  pos_ratio={summary['positive_ratio']:.4f}")
    print(f"[AUC] {auc_value:.6f}  (threshold-independent)")
    print(f"{'='*60}")
    print(f"[Optimal Metric] {metric_name}")
    print(f"[Best Threshold] {best['threshold']:.4f}  ({metric_name}={best_score:.6f})")
    print(f"  acc={best['acc']:.4f}  precision={best['precision']:.4f}  recall={best['recall']:.4f}  f1={best['f1']:.4f}")
    print(f"  balanced_acc={best['balanced_acc']:.4f}  youden_j={best['youden_j']:.4f}")
    print(f"  TP={int(best['tp'])}  TN={int(best['tn'])}  FP={int(best['fp'])}  FN={int(best['fn'])}")
    print(f"{'='*60}")
    print(f"[Recommended Range] [{range_low:.4f}, {range_high:.4f}]  ({metric_name} >= {best_score:.6f} - {delta:.6f})")
    print(f"[Saved] curve csv  : {curve_out}")
    print(f"[Saved] summary json: {summary_out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
