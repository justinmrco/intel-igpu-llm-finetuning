# Publication review

Reviewed on 15 November 2026. This document describes local preparation, not a new training campaign or live GitHub CI result.

## Completed

- Curated and raw workspaces snapshotted separately; ZIP entries and source files checked by SHA-256. All 1,053 original files remained unchanged during preparation.
- All ten selected result JSONs found byte-identical in the supplied raw testing workspace. Source mapping and limits are recorded in `EVIDENCE_MAP.md`; a source file's presence is not proof that exact version produced a historical run.
- All 54 copied evidence files verified against `provenance.json`.
- Ten completion records checked against contiguous training CSVs: nine 2,000-update runs and one single-update probe.
- Accuracy summaries recomputed; ZO final accuracy checked against its saved prediction diagnostics.
- Integrity regression checks exercised both modified-file rejection and a rehashed, truncated CSV.
- Two charts regenerated on CPU and visually inspected; original figures retained unchanged.
- All 17 Python files parsed for Python 3.11 syntax without importing training code.
- Text scan found no obvious private keys, provider tokens, email addresses, personal Windows user paths or the known academic identifier. This is a targeted review, not a guarantee that no sensitive content exists.
- Public package omits original PDFs, handoffs, backups, adapters, tokenizers and raw testing directories. No model weights, dataset downloads or third-party reference PDFs are included.

## Limits

Training, inference, XPU capacity, dependency vulnerability status, upstream license terms and historical cloud/account state were not newly tested. No accuracy or efficiency improvements are claimed. CI is configured for the standard-library verifier and regression check; remote execution must be observed after publication.

No broad open-source license was selected on the author's behalf. Third-party tools and models retain their own terms. The public repository is an evidence-backed portfolio research archive with reproducible saved-result analysis, not a claim of exact end-to-end training reproducibility.
