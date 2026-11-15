"""Plot ZO validation accuracy from result.json validation_log."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

THESIS_DIR = Path("results/thesis_relevant")
OUT_PATH = Path("results/figures/validation_accuracy_zo.png")

RUNS = {
    "ZO seed 42": "zo_C1_seed42/result.json",
    "ZO seed 0": "zo_C1_seed0/result.json",
    "ZO seed 1": "zo_C1_seed1/result.json",
}


def read_val_log(path):
    with open(path, "r") as f:
        data = json.load(f)
    log = data.get("validation_log", [])
    steps = [entry["step"] for entry in log]
    accs = [entry["val_acc"] * 100 for entry in log]
    return steps, accs


def main():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for label, rel in RUNS.items():
        path = THESIS_DIR / rel
        if not path.exists():
            print(f"[SKIP] {label}: {path} not found")
            continue
        steps, accs = read_val_log(path)
        if not steps:
            print(f"[SKIP] {label}: empty validation_log")
            continue
        ax.plot(steps, accs, "-", label=label, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("ZO validation accuracy (3 seeds)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.axhline(50, linestyle=":", color="gray", alpha=0.5, label="chance")
    ax.set_ylim(40, 100)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()