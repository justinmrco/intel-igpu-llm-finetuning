"""Verify saved evidence and summarize outcomes; stdlib only, no training or writes."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]


def analyze(root=ROOT):
    root = root.resolve()
    manifest = json.loads((root / "docs/provenance.json").read_text(encoding="utf-8"))
    seen = set()
    for record in manifest["files"]:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or record["path"] in seen:
            raise ValueError(f"Invalid or duplicate manifest path: {record['path']}")
        seen.add(record["path"])
        content = path.read_bytes()
        if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"Evidence checksum mismatch: {record['path']}")

    results = sorted((root / "results").glob("*/result.json"))
    expected = {name for name in seen if name.startswith("results/") and name.endswith("/result.json")}
    if not expected or {p.relative_to(root).as_posix() for p in results} != expected:
        raise ValueError("Result inventory differs from the evidence manifest")
    rows = []
    for path in results:
        data = json.loads(path.read_text(encoding="utf-8"))
        steps = data.get("steps_completed", data.get("num_steps"))
        accuracy = data.get("final_validation_metric", data.get("final_validation_accuracy"))
        if data.get("status") != "completed" or type(steps) is not int or steps < 1:
            raise ValueError(f"Invalid completion record: {path.parent.name}")
        if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"Invalid accuracy: {path.parent.name}")
        csv_path = path.with_name("train_loss.csv")
        if csv_path.relative_to(root).as_posix() not in seen:
            raise ValueError(f"Untracked training CSV: {csv_path}")
        with csv_path.open(encoding="utf-8", newline="") as stream:
            losses = list(csv.DictReader(stream))
        if len(losses) != steps:
            raise ValueError(f"Step count mismatch: {path.parent.name}")
        for index, loss in enumerate(losses, 1):
            if int(loss["step"]) != index or not math.isfinite(float(loss["train_loss"])):
                raise ValueError(f"Invalid training row: {path.parent.name}, row {index}")
        if data["method"] == "zo":
            diagnostic = path.with_name("prediction_diagnostics.json")
            if diagnostic.relative_to(root).as_posix() not in seen:
                raise ValueError(f"Untracked ZO diagnostics: {diagnostic}")
            final = json.loads(diagnostic.read_text(encoding="utf-8"))["final_validation"]
            samples = final["examples"]
            if not math.isclose(final["accuracy"], accuracy, abs_tol=1e-12):
                raise ValueError(f"Diagnostic accuracy mismatch: {path.parent.name}")
            group = "ZO"
        else:
            samples = data["validation_samples"]
            group = "3B probe" if data.get("model", "").endswith("-3B") else (
                "FO" if data["method"] == "first_order" else "QLoRA"
            )
        if type(samples) is not int or samples < 1:
            raise ValueError(f"Invalid validation population: {path.parent.name}")
        rows.append({"run": path.parent.name, "group": group, "steps": steps,
                     "accuracy": accuracy, "samples": samples,
                     "peak_gb": data["peak_gpu_memory_gb"],
                     "seconds_per_update": data["total_elapsed_time_sec"] / steps})
        if not all(math.isfinite(rows[-1][key]) and rows[-1][key] > 0
                   for key in ("peak_gb", "seconds_per_update")):
            raise ValueError(f"Invalid resource metric: {path.parent.name}")
    return rows, len(seen)


def main():
    try:
        rows, count = analyze()
        print(f"PASS: {count} evidence checksums; {len(rows)} completed run records and CSVs.")
        print("Workflow | Runs | Updates/run | Validation n | Accuracy mean +/- sample SD (pp) | Peak GB range | Elapsed s/update mean")
        for group in ("FO", "QLoRA", "ZO", "3B probe"):
            runs = [row for row in rows if row["group"] == group]
            if not runs:
                continue
            accuracies = [row["accuracy"] * 100 for row in runs]
            spread = f"{statistics.stdev(accuracies):.2f}" if len(runs) > 1 else "n/a"
            updates = ",".join(map(str, sorted({row["steps"] for row in runs})))
            samples = ",".join(map(str, sorted({row["samples"] for row in runs})))
            print(f"{group} | {len(runs)} | {updates} | {samples} | "
                  f"{statistics.mean(accuracies):.2f} +/- {spread} | "
                  f"{min(row['peak_gb'] for row in runs):.6f}-{max(row['peak_gb'] for row in runs):.6f} | "
                  f"{statistics.mean(row['seconds_per_update'] for row in runs):.4f}")
        print("Descriptive workflows with different settings/populations; run labels are not proof of controlled seeds.")
        print("Elapsed time includes workflow overhead. Allocator peaks are not total system RAM. 3B is one update only.")
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
