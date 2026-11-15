"""Runnable integrity checks; no ML libraries, GPU, network or original-file writes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis", ROOT / "scripts/analyze_results.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

rows, count = analysis.analyze()
assert len(rows) == 10 and count >= 54

with tempfile.TemporaryDirectory() as directory:
    copy = Path(directory)
    shutil.copytree(ROOT / "results", copy / "results")
    (copy / "docs").mkdir()
    manifest = json.loads((ROOT / "docs/provenance.json").read_text(encoding="utf-8"))
    manifest["files"] = [record for record in manifest["files"] if record["path"].startswith("results/")]
    manifest_path = copy / "docs/provenance.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    analysis.analyze(copy)
    target = copy / "results/qlora_sst2_nf4_seed42/train_loss.csv"
    original = target.read_bytes()
    target.write_bytes(original + b"\n")
    try:
        analysis.analyze(copy)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("Modified evidence was accepted")
    target.write_bytes(b"\n".join(original.splitlines()[:-1]) + b"\n")
    for record in manifest["files"]:
        if record["path"] == target.relative_to(copy).as_posix():
            record["bytes"] = target.stat().st_size
            record["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        analysis.analyze(copy)
    except ValueError as error:
        assert "Step count" in str(error)
    else:
        raise AssertionError("Truncated training record was accepted")

print("PASS: saved evidence, tamper detection and truncated-CSV rejection.")
