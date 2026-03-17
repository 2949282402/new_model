#!/usr/bin/env python3
"""
绘制压缩鲁棒性折线图: AUC 和 ACC 随 JPEG 压缩质量的变化。
两张独立图片，无标题，中文轴标注。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

# ============ 中文字体 ============
for fname in ["SimSun", "SimHei", "Microsoft YaHei", "STSong"]:
    if any(fname in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fname
        break
plt.rcParams["axes.unicode_minus"] = False

# ============ 数据（硬编码） ============
qualities = [100, 90, 70, 50, 20, 10]
aucs      = [0.9547, 0.9315, 0.9136, 0.9091, 0.8938, 0.8519]
accs      = [0.8799, 0.8591, 0.8455, 0.8354, 0.8082, 0.7692]

out_dir = Path(r"C:\hejulian\exp\compress_robustness")
out_dir.mkdir(parents=True, exist_ok=True)

common_rc = {
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
}
plt.rcParams.update(common_rc)

# ============ AUC 图 ============
fig1, ax1 = plt.subplots(figsize=(6, 5))
ax1.plot(qualities, aucs, "o-", color="#2563EB", linewidth=2, markersize=8)
ax1.set_xlabel("压缩质量", fontsize=13)
ax1.set_ylabel("AUC", fontsize=13)
ax1.set_xticks(qualities)
ax1.set_xlim(5, 105)
ax1.invert_xaxis()
for q, v in zip(qualities, aucs):
    ax1.annotate(f"{v:.4f}", (q, v), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=9)
fig1.tight_layout()
fig1.savefig(str(out_dir / "compress_robustness_auc.png"), dpi=300, bbox_inches="tight")
fig1.savefig(str(out_dir / "compress_robustness_auc.pdf"), bbox_inches="tight")
print(f"[Saved] compress_robustness_auc.png / .pdf")

# ============ ACC 图 ============
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(qualities, accs, "s-", color="#DC2626", linewidth=2, markersize=8)
ax2.set_xlabel("压缩质量", fontsize=13)
ax2.set_ylabel("ACC", fontsize=13)
ax2.set_xticks(qualities)
ax2.set_xlim(5, 105)
ax2.invert_xaxis()
for q, v in zip(qualities, accs):
    ax2.annotate(f"{v*100:.2f}%", (q, v), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=9)
fig2.tight_layout()
fig2.savefig(str(out_dir / "compress_robustness_acc.png"), dpi=300, bbox_inches="tight")
fig2.savefig(str(out_dir / "compress_robustness_acc.pdf"), bbox_inches="tight")
print(f"[Saved] compress_robustness_acc.png / .pdf")
