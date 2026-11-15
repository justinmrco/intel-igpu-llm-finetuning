# Experiment settings and evidence map

Generated from the supplied result records; checked against the raw testing workspace on 15 November 2026. Relative raw paths below refer to the private preserved workspace, not files required in a public clone.

## Selected run records

| Run folder | Updates | Validation n | Final accuracy | Matching raw run directory |
| --- | ---: | ---: | ---: | --- |
| [fo_bf16_classification_qloraconfig_seed0](../results/fo_bf16_classification_qloraconfig_seed0/result.json) | 2000 | 100 | 97.00% | `results/fo_bf16_classification_qloraconfig_seed0` |
| [fo_bf16_classification_qloraconfig_seed1](../results/fo_bf16_classification_qloraconfig_seed1/result.json) | 2000 | 100 | 95.00% | `results/fo_bf16_classification_qloraconfig_seed1` |
| [fo_bf16_classification_qloraconfig_seed42](../results/fo_bf16_classification_qloraconfig_seed42/result.json) | 2000 | 100 | 96.00% | `results/fo_bf16_classification_qloraconfig_seed42` |
| [qb4_3b_boundary_probe_seed42](../results/qb4_3b_boundary_probe_seed42/result.json) | 1 | 100 | 62.00% | `results/qb4_3b_boundary_probe_seed42` |
| [qlora_sst2_nf4_seed0](../results/qlora_sst2_nf4_seed0/result.json) | 2000 | 100 | 95.00% | `results/qlora_sst2_nf4_seed0` |
| [qlora_sst2_nf4_seed1](../results/qlora_sst2_nf4_seed1/result.json) | 2000 | 100 | 94.00% | `results/qlora_sst2_nf4_seed1` |
| [qlora_sst2_nf4_seed42](../results/qlora_sst2_nf4_seed42/result.json) | 2000 | 100 | 93.00% | `results/qlora_sst2_nf4_seed42` |
| [zo_C1_seed0](../results/zo_C1_seed0/result.json) | 2000 | 872 | 64.22% | `results/zo_hyperfix_lr1e5_seed0/zo_C1_seed0` |
| [zo_C1_seed1](../results/zo_C1_seed1/result.json) | 2000 | 872 | 74.89% | `results/zo_hyperfix_lr1e5_seed1/zo_C1_seed1` |
| [zo_C1_seed42](../results/zo_C1_seed42/result.json) | 2000 | 872 | 69.61% | `results/zo_hyperfix_lr1e5/zo_C1_seed42` |

Both the JSON and training CSV of every row match the primary raw run directory byte-for-byte. Backup duplicates are preserved privately, not counted as additional runs. JSON `run_id` values sometimes differ from folder names; original identifiers are retained.

## Workflow settings

| Setting | BF16 FO | NF4 QLoRA | ZO |
| --- | --- | --- | --- |
| Main model | Qwen2.5-1.5B | Qwen2.5-1.5B | Qwen2.5-1.5B |
| Objective | Classification head | Generative label completion | Classification head |
| Base/compute path | BF16 requested | NF4 base, BF16 compute requested | FP32 loader; YAML says BF16 |
| Training subset | First 2,000 SST-2 examples | First 2,000 SST-2 examples | Full SST-2 training split |
| Accuracy population | First 100 validation examples | First 100 validation examples | 872 validation examples |
| Batch / max length | 4 / 128 | 4 / 128 | 16 / 256 |
| LoRA rank / alpha | 8 / 32 | 8 / 32 | 4 / 16 |
| Target modules | q/k/v_proj | q/k/v_proj | q_proj |
| Dropout | 0.05 | 0.05 | 0 |
| Learning rate | 2e-5 | 2e-5 | 1e-5 |
| Recorded labels | 0, 1, 42 | 0, 1, 42 | 0, 1, 42 |

The matrix combines saved result records and archived source/configuration. The historical ZO base runtime dtype and exact package environment per run were not logged. Source association below identifies candidate archived implementations, not cryptographic proof of the exact version executed.

## Source and configuration associations

- FO: [standalone control script](../scripts/historical/test_fo_bf16_classification_qloraconfig.py). Major settings are constants in the script; no per-run immutable configuration snapshot is provided. Its seed argument labels outputs without initializing random generators.
- QLoRA: [standalone NF4 script](../scripts/historical/test_qlora_2000steps.py). Settings are embedded; seed labels do not establish controlled replication. The script disables double quantization and uses AdamW.
- ZO: [training entry point](../src/train.py), [model loader](../src/model_setup.py), [optimizer](../src/zo_optimizer.py) and [annotated configuration](../configs/phase2_config.yaml). A [historical run-42 configuration snapshot](../results/zo_C1_seed42/phase2_config_snapshot.yaml) also exists. Config seed lists differ from selected run labels; preserve this discrepancy.
- 3B: [probe script](../scripts/historical/test_3b_qlora_boundary.py) and [one-update result](../results/qb4_3b_boundary_probe_seed42/result.json). Do not present its 62% as converged performance.
- 7B: [probe script](../scripts/historical/test_7b_qlora_boundary.py) and [condensed failure record](../evidence/7b/OOM_traceback.txt). It records the launch command, settings and error, but omits full stack frames and does not establish a universal model-size limit.
- Phase 1: [smoke-test source](../evidence/phase1/phase1_feasibility.py) and [matrix CSV](../evidence/phase1/feasibility_matrix_results.csv). This is synthetic-text feasibility evidence; the QLoRA label in those rows denotes BF16 first-order LoRA in source, and model reuse complicates independence.
- Epsilon check: [saved diagnostic](../diagnostics/eps_stability.txt) and [historical program](../scripts/historical/verify_multi_eps.py). Local fixed-batch behavior does not establish global convergence.
- Allocation check: [transcript](../diagnostics/xpu_ceiling_measurement.txt) and [historical program](../scripts/historical/measure_xpu_ceiling.py). Cumulative allocation success is not a largest-single-block measurement.

## Claims supported by this package

Recorded workflow outcomes, trajectory plots and configuration-specific feasibility observations can be inspected from the files. Isolated causal method comparisons, exact rerun determinism, a general hardware ceiling, inference performance and a state-of-the-art ranking are not established. Scalar JSON/CSV records and their consistent copies are evidence, not independent experimental replication.
