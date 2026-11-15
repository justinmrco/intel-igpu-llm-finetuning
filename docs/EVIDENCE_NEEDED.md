# Useful raw evidence

The current curated collection is sufficient to preserve and explain the selected results. The original testing folder would help establish their surrounding history; it should be archived separately rather than merged blindly into selected results.

## Received on 15 November 2026

The original testing workspace has now been supplied as `originaltestingfolder/`: 894 files totaling 1,037,507,705 bytes. It is retained in place and snapshotted separately. It includes earlier ZO variants, diagnostics, result backups, Phase 1 source and CSV, and a condensed 7B OOM record. Small copies of the latter evidence are included under `evidence/`, with checksums in `provenance.json`.

The list below remains a guide to evidence quality, not a request to provide the same folder again. Linking every historical run to an exact source/environment remains a separate review task. No extra files or reruns are needed to finish preservation.

## Highest value

1. The original 7B OOM console log or condensed failure record, including the command, model/configuration, failure stage and allocation message.
2. Exact scripts and configurations used for each selected run, if different from the copies here; launch commands and dated logs help link source to outputs.
3. Earlier failed attempts cited in the thesis, including error traces and settings. A failure without its configuration is hard to interpret.
4. Original environment exports, package versions and hardware/runtime details captured near the runs.
5. Phase 1 smoke-test source and CSV records if those results will be discussed publicly.

Full base-model weights, downloaded dataset caches, virtual environments and repeated checkpoints are not needed in the public source repository. They can remain in a private raw archive if needed for preservation.

## Handling rules

- Preserve raw files and timestamps; do not edit errors or relabel successful and failed runs.
- Inspect for access tokens, credentials, personal paths and dataset content before publication. Keep originals private and publish redacted copies when needed.
- Record source paths, checksums and the relationship to each claim. Missing evidence should be marked missing, not regenerated or invented.
- Failure logs provide context; they do not automatically establish universal framework incompatibility, a hardware ceiling or a general model-size boundary.

No new training or probes are required to supply these files. Provide the original folder path when available; archival review can proceed from existing records.
