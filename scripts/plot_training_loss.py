
import json
import csv
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib

# Use a non-interactive backend so PNG export does not depend on Qt or an IDE window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. 项目路径与绘图配置
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "reports" / "figures" / "training_loss"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "Qwen2.5-3B-Instruct": (
        ROOT / "outputs/qwen25_3b_instruct_formal_20261105/metrics.jsonl",
        "qwen25_3b",
    ),
    "Qwen3.5-4B": (
        ROOT / "outputs/qwen35_4b_formal/bf16_medical_20261104/logs/metrics.jsonl",
        "qwen35_4b",
    ),
}

WINDOW = 100
BLOCK = 500

# 完整曲线图尺寸：压缩横向宽度
COMPACT_FIGSIZE = (7.2, 5.4)

# 500-Step 趋势图尺寸
BLOCK_FIGSIZE = (7.2, 5.4)

summary = []


# ============================================================
# 2. 读取训练日志
# ============================================================

for model, (source, prefix) in SOURCES.items():

    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not records:
        raise ValueError(f"{model}: empty training log")

    steps = [int(r["step"]) for r in records]
    losses = [float(r["loss"]) for r in records]

    if not np.isfinite(losses).all():
        raise ValueError(f"{model}: invalid loss value")

    print(f"\n===== {model} =====")
    print("Steps:", len(losses))
    print("Initial Loss:", losses[0])
    print("Final Loss:", losses[-1])
    print("Last 1000 Steps Mean:", mean(losses[-1000:]))


    # ========================================================
    # 3. 第一种图：压缩横向距离的完整 Loss 曲线
    #    保留全部 Raw Loss + 100-Step Moving Average
    # ========================================================

    # 计算 100-Step 移动平均
    rolling = [
        mean(losses[max(0, i - WINDOW + 1):i + 1])
        for i in range(len(losses))
    ]

    # 导出完整数据 CSV
    compact_csv = OUT / f"{prefix}_loss_compact.csv"

    with compact_csv.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "step",
            "raw_loss",
            "moving_average_100",
        ])

        writer.writerows(
            zip(steps, losses, rolling)
        )

    # 创建完整曲线图
    fig, ax = plt.subplots(figsize=COMPACT_FIGSIZE)

    # 原始 Loss：保留所有训练 Step
    ax.plot(
        steps,
        losses,
        linewidth=0.55,
        alpha=0.22,
        label="Raw Training Loss",
    )

    # 100-Step 移动平均
    ax.plot(
        steps,
        rolling,
        linewidth=2.0,
        label="100-Step Moving Average",
    )

    # 自动缩小纵轴范围
    # 使用 0.5% 和 99.5% 分位数，避免少数极端值
    # 过度拉伸纵轴范围
    y_low = min(losses)
    y_high = max(losses)

    padding = (y_high - y_low) * 0.04

    y_min = max(0, y_low - padding)
    y_max = y_high + padding

    ax.set_ylim(y_min, y_max)

    ax.set_title(f"{model} - Training Loss")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Training Loss")

    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.margins(x=0)

    fig.tight_layout()

    compact_png = OUT / f"{prefix}_loss_compact.png"

    fig.savefig(
        compact_png,
        dpi=240,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Compact PNG:", compact_png)
    print("Compact CSV:", compact_csv)
    print(f"Compact Y-axis: {y_min:.3f} - {y_max:.3f}")






    # ========================================================
    # 4. 第二种图：仅展示 500-Step 非重叠区间平均趋势
    # ========================================================

    block_steps = []
    block_averages = []
    block_rows = []

    for block_index, start in enumerate(range(0, len(losses), BLOCK), start=1):
        end = min(start + BLOCK, len(losses))
        block_step_values = steps[start:end]
        block_loss_values = losses[start:end]

        # Plot each block at its starting step (0, 500, 1000, ...),
        # matching the requested sparse 500-step trend presentation.
        block_step_position = block_step_values[0] - 1
        block_average = mean(block_loss_values)

        block_steps.append(block_step_position)
        block_averages.append(block_average)
        block_rows.append([
            block_index,
            block_step_values[0],
            block_step_values[-1],
            block_step_position,
            len(block_loss_values),
            block_average,
        ])

    # 导出非重叠区间平均值；全部 Raw Loss 仍完整保存在 compact CSV 中
    block_csv = OUT / f"{prefix}_loss_500step.csv"

    with block_csv.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)
        writer.writerow([
            "block_index",
            "step_start",
            "step_end",
            "block_step_position",
            "raw_loss_count",
            "average_loss_per_500_steps",
        ])
        writer.writerows(block_rows)

    # 只绘制每个非重叠 500-Step 区间的平均值，样式与用户参考图一致
    fig, ax = plt.subplots(figsize=BLOCK_FIGSIZE)

    ax.plot(
        block_steps,
        block_averages,
        linewidth=2.5,
        marker="o",
        markersize=5,
        label="Average Loss per 500 Steps",
    )

    # 纵轴围绕区间平均值自动缩放，突出整体下降与收敛趋势
    avg_min = min(block_averages)
    avg_max = max(block_averages)
    avg_padding = max((avg_max - avg_min) * 0.08, 0.01)
    ax.set_ylim(avg_min - avg_padding, avg_max + avg_padding)

    ax.set_title(f"{model} - 500-Step Loss Trend")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Average Training Loss")

    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.margins(x=0)

    fig.tight_layout()

    block_png = OUT / f"{prefix}_loss_500step.png"

    fig.savefig(
        block_png,
        dpi=240,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("500-Step PNG:", block_png)
    print("500-Step CSV:", block_csv)
    print("Raw Log Records:", len(losses))
    print("500-Step Average Points:", len(block_averages))





    # ========================================================
    # 5. 汇总统计
    # ========================================================

    summary.append({
        "model": model,
        "steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "last_1000_mean": mean(losses[-1000:]),
        "last_2000_mean": mean(losses[-2000:]),
        "compact_y_min": y_min,
        "compact_y_max": y_max,
    })


# ============================================================
# 6. 导出训练 Loss 汇总表
# ============================================================

summary_csv = OUT / "training_loss_summary.csv"

with summary_csv.open(
    "w", newline="", encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary[0].keys()
    )

    writer.writeheader()
    writer.writerows(summary)


# ============================================================
# 7. 完成
# ============================================================

print("\n" + "=" * 60)
print("ALL TRAINING LOSS FIGURES GENERATED")
print("=" * 60)

print("Output directory:", OUT)
print("Total figures:", len(SOURCES) * 2)
print("Total CSV files:", len(SOURCES) * 2 + 1)
print("Summary CSV:", summary_csv)
