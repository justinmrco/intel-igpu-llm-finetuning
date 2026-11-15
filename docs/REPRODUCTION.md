# Reproduction and provenance

## Source mapping

Copies preserve the supplied file bytes. Paths below are relative to the original `thesis_bundle_2026_10_27/`.

| Candidate repository | Original location |
| --- | --- |
| `src/` | `04_source_code/src/` |
| `configs/` | `04_source_code/configs/` |
| `scripts/historical/` | `05_test_scripts/` |
| `results/<run>/` | Top-level files in `02_results_verified/<run>/`, excluding adapter folders |
| `figures/` | `03_figures/` |
| `environment/` | `06_environment/` |
| `diagnostics/` | `07_diagnostics/` |

`provenance.json` records the original workspace path and SHA-256 of each copied evidence file. The complete original collection is separately snapshotted under the local preservation folder. Newly written documentation is not historical evidence.

## Important interpretation constraints

- The ZO loader requests FP32 even though its YAML says BF16; historical frozen-base runtime dtype was not directly logged.
- The YAML seed list and selected run labels differ. Preserve both rather than relabeling past runs.
- FO/QLoRA scripts accept seed labels without initializing random generators.
- Methods differ in training data size, validation population, head/objective, sequence length and adapter settings. Precision alone cannot explain the observed differences.
- Keep decimal GB and binary GiB distinct. A cumulative allocation test and a model transfer failure do not prove a universal maximum trainable model size.
- The old bundle inventory lists some changed or absent historical files. The preservation manifest describes the actual collection supplied on 15 November 2026.

## Execution readiness

The new `scripts/analyze_results.py` and `scripts/plot_results.py` are the supported analysis entry points. Use the commands in the root README. Verification uses Python's standard library; optional charts require the separate `requirements-analysis.txt`. Preparation tested these commands with Python 3.11.16 and Matplotlib 3.11.2. No XPU library or training dependency is needed for them.

`tests/check_analysis.py` runs against temporary copies when simulating damaged evidence. The originals are not changed. `.gitattributes` disables line-ending normalization so fresh checkouts retain the exact bytes described by the manifest. Checksums demonstrate consistency with the supplied manifest; they are not a signature or independent proof of experimental authenticity.

Historical scripts are preserved, not adapted to this new layout. Plotting expects paths such as `results/thesis_relevant`; selected outputs now live directly under `results/`. Experiment scripts can overwrite their configured result directories. Some verification programs create backup folders rather than perform read-only checks.

Before any future run, document the intended entry point, working directory, environment, input paths and a fresh output directory. Pin compatible dependencies and model/dataset revisions where possible. Keep repairs in a separate development change, preserving this historical baseline.

The archived ZO entry point is `python -m src.train`; its required options are `--config_id`, `--method` and `--seed`. It reads `configs/phase2_config.yaml` from the working directory and supports `--out_dir`. FO/QLoRA standalone scripts use positional run labels and hardcoded output paths. These are source-level interface descriptions, not recovered historical commands or instructions to rerun now.

The environment text files are historical exports, not portable requirements files. Some include build-local `file://` references and the Conda list is Windows-specific. Do not install the entire export blindly. Training used different environments; a single verified training lockfile is not available. Downloading models/datasets also requires independent review of upstream revisions, terms and any `trust_remote_code=True` paths in historical scripts.

Start with analysis of saved files rather than new training. New experiments are outside this arrangement task.
