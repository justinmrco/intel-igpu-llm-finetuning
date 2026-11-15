"""
Plot convergence curves from existing train_loss.csv files.

Produces two panels:
  Left:  ZO, 3 seeds, raw (faint) + moving average (solid)
  Right: QLoRA (solid) and BF16-FO (dashed), 3 seeds each,
         raw (faint) + moving average (full alpha)

No new compute — only plots existing data.
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

THESIS_DIR = Path("results/thesis_relevant")
OUT_PATH = Path("results/figures/convergence_curves.png")
MOVING_AVG_WINDOW = 30

RUNS = {
    "ZO seed 42": "zo_C1_seed42/train_loss.csv",
    "ZO seed 0": "zo_C1_seed0/train_loss.csv",
    "ZO seed 1": "zo_C1_seed1/train_loss.csv",
    "QLoRA seed 42": "qlora_sst2_nf4_seed42/train_loss.csv",
    "QLoRA seed 0": "qlora_sst2_nf4_seed0/train_loss.csv",
    "QLoRA seed 1": "qlora_sst2_nf4_seed1/train_loss.csv",
    "BF16-FO seed 42": "fo_bf16_classification_qloraconfig_seed42/train_loss.csv",
    "BF16-FO seed 0": "fo_bf16_classification_qloraconfig_seed0/train_loss.csv",
    "BF16-FO seed 1": "fo_bf16_classification_qloraconfig_seed1/train_loss.csv",
}


def read_loss(path):
    """Read step and train_loss columns from a train_loss.csv file."""
    steps, losses = [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["train_loss"]))
    return steps, losses


def moving_average(values, window=30):
    """Causal moving average. The first `window-1` points are dropped."""
    if len(values) < window:
        return list(values)
    out = []
    running = sum(values[:window])
    out.append(running / window)
    for i in range(window, len(values)):
        running += values[i] - values[i - window]
        out.append(running / window)
    return out


def plot_run(ax, label, path, linestyle="-", raw_alpha=0.15, smooth_alpha=0.9):
    """Plot one run: raw (faint) + smoothed (solid)."""
    if not path.exists():
        print(f"[SKIP] {label}: {path} not found")
        return
    steps, losses = read_loss(path)
    smooth = moving_average(losses, MOVING_AVG_WINDOW)
    smooth_steps = steps[MOVING_AVG_WINDOW - 1:]

    # Raw per-step loss, faint
    ax.plot(steps, losses, linestyle=linestyle, alpha=raw_alpha, color=None)
    # Smoothed trend, solid
    ax.plot(
        smooth_steps,
        smooth,
        linestyle=linestyle,
        alpha=smooth_alpha,
        label=label,
    )


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # ---------- Left panel: ZO, 3 seeds ----------
    for label in ["ZO seed 42", "ZO seed 0", "ZO seed 1"]:
        plot_run(axes[0], label, THESIS_DIR / RUNS[label], linestyle="-")

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title(
        f"ZO (3 seeds) — raw + {MOVING_AVG_WINDOW}-step moving average"
    )
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, 2000)

    # ---------- Right panel: QLoRA (solid) + BF16-FO (dashed) ----------
    for label in ["QLoRA seed 42", "QLoRA seed 0", "QLoRA seed 1"]:
        plot_run(axes[1], label, THESIS_DIR / RUNS[label], linestyle="-")
    for label in ["BF16-FO seed 42", "BF16-FO seed 0", "BF16-FO seed 1"]:
        plot_run(axes[1], label, THESIS_DIR / RUNS[label], linestyle="--")

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Training Loss")
    axes[1].set_title(
        f"QLoRA (solid) and BF16-FO (dashed), 3 seeds each "
        f"— raw + {MOVING_AVG_WINDOW}-step moving average"
    )
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, 2000)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()