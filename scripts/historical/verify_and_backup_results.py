"""
Verify and back up thesis-relevant results.

1. Check that each thesis-relevant directory exists and contains the expected files.
2. Extract the reported metrics from each result.json.
3. Compare against the thesis's reported numbers.
4. Copy the thesis-relevant directories to results/thesis_relevant/.
5. Copy all result directories to a backup location.
"""
import json
import shutil
from pathlib import Path

ROOT = Path("results")
THESIS_DIR = ROOT / "thesis_relevant"
BACKUP_DIR = Path("results_backup_2026_10_27")

# Thesis-relevant directories and their expected reported values
THESIS_RUNS = {
    "zo_hyperfix_lr1e5/zo_C1_seed42": {"acc": 0.6961, "label": "ZO seed 42"},
    "zo_hyperfix_lr1e5_seed0/zo_C1_seed0": {"acc": 0.6422, "label": "ZO seed 0"},
    "zo_hyperfix_lr1e5_seed1/zo_C1_seed1": {"acc": 0.7489, "label": "ZO seed 1"},
    "qlora_sst2_nf4_seed42": {"acc": 0.93, "label": "QLoRA seed 42"},
    "qlora_sst2_nf4_seed0": {"acc": 0.95, "label": "QLoRA seed 0"},
    "qlora_sst2_nf4_seed1": {"acc": 0.94, "label": "QLoRA seed 1"},
    "fo_bf16_classification_qloraconfig_seed42": {"acc": 0.96, "label": "BF16-FO seed 42"},
    "fo_bf16_classification_qloraconfig_seed0": {"acc": 0.97, "label": "BF16-FO seed 0"},
    "fo_bf16_classification_qloraconfig_seed1": {"acc": 0.95, "label": "BF16-FO seed 1"},
    "qb4_3b_boundary_probe_seed42": {"acc": None, "label": "3B boundary probe"},
    "qb4_7b_boundary_probe_seed42": {"acc": None, "label": "7B boundary probe (OOM)"},
}

def find_result_json(run_dir):
    """Find result.json in a run directory or one level down."""
    candidates = list(run_dir.glob("result.json")) + list(run_dir.glob("*/result.json"))
    return candidates[0] if candidates else None

def main():
    THESIS_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("THESIS-RELEVANT RESULTS VERIFICATION")
    print("=" * 70)

    missing = []
    mismatched = []

    for rel_path, expected in THESIS_RUNS.items():
        run_dir = ROOT / rel_path
        label = expected["label"]

        if not run_dir.exists():
            print(f"[MISSING] {label}: {run_dir} does not exist")
            missing.append(rel_path)
            continue

        result_json = find_result_json(run_dir)
        if result_json is None:
            print(f"[NO JSON] {label}: no result.json found in {run_dir}")
            missing.append(rel_path)
            continue

        with open(result_json, "r") as f:
            data = json.load(f)

        acc = data.get("final_validation_accuracy") or data.get("final_validation_metric")
        peak = data.get("peak_gpu_memory_gb")

        if expected["acc"] is not None:
            if acc is None:
                print(f"[NO ACC] {label}: result.json has no final_validation_accuracy")
                mismatched.append(rel_path)
            elif abs(acc - expected["acc"]) > 0.005:
                print(f"[MISMATCH] {label}: expected {expected['acc']:.4f}, got {acc:.4f}")
                mismatched.append(rel_path)
            else:
                print(f"[OK] {label}: acc={acc:.4f}, peak_mem={peak:.3f} GB" if peak else f"[OK] {label}: acc={acc:.4f}")
        else:
            print(f"[OK] {label}: no accuracy check (boundary probe)")

        # Copy to thesis_relevant/
        dest = THESIS_DIR / Path(rel_path).name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(run_dir, dest)
        print(f"  -> Copied to {dest}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total thesis runs: {len(THESIS_RUNS)}")
    print(f"Missing: {len(missing)}")
    print(f"Mismatched: {len(mismatched)}")

    if missing:
        print("\nMissing runs:")
        for m in missing:
            print(f"  - {m}")
    if mismatched:
        print("\nMismatched runs:")
        for m in mismatched:
            print(f"  - {m}")

    # Back up all results
    print("\n" + "=" * 70)
    print("BACKING UP ALL RESULTS")
    print("=" * 70)
    for src in ROOT.iterdir():
        if src.is_dir() and src.name not in ("thesis_relevant", "figures"):
            dest = BACKUP_DIR / src.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            print(f"  -> {src.name} -> {dest}")

    print(f"\nBackup complete: {BACKUP_DIR}")

if __name__ == "__main__":
    main()