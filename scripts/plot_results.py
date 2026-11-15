"""Plot verified saved records on CPU; writes only figures/generated/."""
import csv
import json
import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_results import ROOT, analyze


def main():
    rows, _ = analyze()
    output = ROOT / "figures/generated"
    output.mkdir(parents=True, exist_ok=True)
    colors = {0: "#0072B2", 1: "#D55E00", 42: "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, group in zip(axes, ("FO", "QLoRA", "ZO")):
        for row in (item for item in rows if item["group"] == group):
            label = int(row["run"].rsplit("seed", 1)[1])
            with (ROOT / "results" / row["run"] / "train_loss.csv").open(newline="", encoding="utf-8") as stream:
                losses = list(csv.DictReader(stream))
            steps = [int(item["step"]) for item in losses]
            values = [float(item["train_loss"]) for item in losses]
            window = 30
            running = sum(values[:window])
            smooth = [running / window]
            for index in range(window, len(values)):
                running += values[index] - values[index - window]
                smooth.append(running / window)
            ax.plot(steps, values, alpha=0.10, color=colors[label], linewidth=0.6)
            ax.plot(steps[window - 1:], smooth, color=colors[label], label=f"Run {label}", linewidth=1.3)
        ax.set(title=group, xlabel="Update", ylabel="Training loss")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Recorded training loss: raw and 30-update moving average")
    fig.text(0.5, 0.015, "Different objectives/settings; panel scales differ. Run labels do not establish controlled seeds.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output / "training_loss.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for row in (item for item in rows if item["group"] == "ZO"):
        label = int(row["run"].rsplit("seed", 1)[1])
        log = json.loads((ROOT / "results" / row["run"] / "result.json").read_text())["validation_log"]
        if not log or not all(math.isfinite(item["val_acc"]) and 0 <= item["val_acc"] <= 1 for item in log):
            raise ValueError(f"Invalid validation history: {row['run']}")
        ax.plot([item["step"] for item in log], [item["val_acc"] * 100 for item in log],
                label=f"Run {label}", color=colors[label])
    ax.axhline(50, linestyle=":", color="gray", alpha=0.6, label="50% reference")
    ax.set(title="ZO validation accuracy: 872 SST-2 examples", xlabel="Update", ylabel="Accuracy (%)", ylim=(40, 85))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "zo_validation_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"Saved 2 figures to {output}; no training or network requests.")


if __name__ == "__main__":
    main()
