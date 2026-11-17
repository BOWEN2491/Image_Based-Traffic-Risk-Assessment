# Independent Holdout Protocol

## Purpose and current evidence status

This protocol defines how to qualify a future natural-distribution holdout for a frozen traffic-risk candidate. The current BDD100K-derived local manifests are marked `local-unverified`; their upstream source archive and annotation history have not been independently authenticated. Earlier validation and holdout roles, performance samples, error analysis, and manual image exposure must be inventoried before any new sample is treated as independent.

The automatic audit can prove exact overlaps among the manifests supplied to it and verify that evidence files match their recorded SHA256. It cannot prove that a manifest inventory is historically complete, that a source document is truthful, that annotations are correct, or that adjacent frames are independent. Human reviewers must supply and sign those conclusions with supporting evidence. Missing evidence keeps the corresponding gate false.

The current S4 manifests do not include `source_namespace`. They cannot pass the identity completeness or history coverage gates as-is. Add a canonical namespace to each row only after documenting its provider/archive identity; the tool will not invent one from IDs, filenames, or local folder names.

## Human work required before a formal holdout

1. Inventory all known training, development, prior holdout, error-analysis, manual-review, and performance-sample manifests. Record each file name, SHA256, scope, collection time, and known gaps. A manifest discovered later invalidates a prior `history_coverage=complete` claim until the audit is rerun.
2. Verify the provider, archive identity, original archive checksum, acquisition date, acquisition/conversion steps, and source URL against original records. Keep source evidence files under the provenance file's directory and record their SHA256. A URL or a `source-verified` text field by itself is not evidence.
3. Verify how source groups are formed from ride/video/source metadata. Do not treat `image-id-fallback` as group independence. Record reviewer, date, scope, and references supporting the mapping.
4. Freeze the sampling protocol and candidate/configuration hashes before viewing candidate predictions. Select up to 500 naturally sampled images by reliable source group. Keep any targeted stress images separate and never add difficult images to the natural sample merely to fill class counts.
5. Review source and label integrity, category coverage, duplicate/corrupt images, annotation completeness, and EXIF/coordinate semantics. Record who reviewed which scope and when. Do not represent automatic checks as human annotation review.
6. Search likely adjacent frames or perceptual near-duplicates and create a human review queue when the source contains video sequences. Exact file and pixel hashes do not identify visually similar neighboring frames.
7. Only after every gate is satisfied, reserve the candidate set as a final holdout. A person selecting/tuning a model must not inspect its predictions on this set. Any later tuning exposes the set and requires a new protocol and sample.

Minimum operational sample support is 500 natural images from at least 100 reliable source groups, with at least 100 `person`, `car`, and `traffic light` instances per class spread across at least 25 groups each. Falling short means `sample_size_eligible=false`; do not backfill from old exposed data. These are minimum process gates, not proof of adequate statistical power.

## Manifest identity contract

Candidate and historical manifests are UTF-8 JSONL objects. Each row should contain:

```json
{
  "source_namespace": "provider/archive-or-dataset-version",
  "sample_id": "source-relative-stable-id",
  "group_id": "source-namespaced-ride-or-video-id",
  "grouping": "rideID",
  "split": "development",
  "image_path": "relative/path/image.jpg",
  "image_sha256": "64 lowercase hex characters",
  "pixel_sha256": "64 lowercase hex characters",
  "objects": []
}
```

The audit compares `sample_id` and `group_id` inside `source_namespace`; it compares `pixel_sha256` and `image_sha256` globally. Any equality with any supplied historical role, including an old holdout row with the same role name, excludes that candidate. Candidates with missing identities or fallback groups are excluded. Candidate image paths must stay under `--dataset-root`, exist, and match their manifest `image_sha256`. The tool also decodes each candidate as RGB using the same `prepare_evaluation.pixel_hash` semantics and rejects a stored `pixel_sha256` that does not match. Historical image files are not reopened; their identity rows are treated as records from the supplied, SHA-pinned manifests.

## Provenance record for the audit

The input is a JSON object following the existing `prepare_evaluation` provenance fields, plus separate human-reviewed records:

```json
{
  "schema_version": 1,
  "source_status": "local-unverified",
  "provider": null,
  "archive_id": null,
  "source_url": null,
  "acquired_date": null,
  "acquisition_and_conversion": [],
  "source_archive_sha256": null,
  "verified_by": null,
  "verified_date": null,
  "source_evidence": [],
  "annotation_review": {
    "status": "awaiting-human-review",
    "reviewer": null,
    "reviewed_date": null,
    "scope": null
  },
  "annotation_evidence": [],
  "grouping_review": {
    "status": "awaiting-human-review",
    "reviewer": null,
    "reviewed_date": null,
    "scope": null
  },
  "grouping_evidence": [],
  "history_coverage": {
    "status": "incomplete",
    "reviewer": null,
    "reviewed_date": null,
    "scope": [],
    "manifest_inventory": []
  }
}
```

Evidence entries use `{"path": "relative-evidence-file", "sha256": "..."}` and resolve relative to this JSON file. A recorded evidence hash must match the file bytes. For a complete history attestation, the scope must include `training`, `development`, `prior-holdouts`, `error-analysis`, `manual-review`, and `performance-samples`; `manifest_inventory` must exactly list every `--history-manifest` by file name and SHA256. This verifies the inventory was recorded consistently, not that the reviewer actually found every historical use. Source, annotation, and grouping blocks also require reviewer identity, ISO date, and scope; their evidence content still requires human judgement.

## Running the automatic audit

Run from the repository root in the locked Python environment:

```powershell
.\.venv\python.exe tools/audit_independent_holdout.py `
  --candidate-manifest local/new-candidate/images.jsonl `
  --history-manifest results/next-iteration/s4-evaluation/images.jsonl `
  --history-manifest local/other-exposure/images.jsonl `
  --provenance local/new-candidate/provenance.json `
  --dataset-root 'D:\path\to\verified-dataset-root' `
  --output results/independent-holdout/audit-001
```

The output must be a new directory outside the dataset root and all input files. The CLI writes `eligible.jsonl`, `excluded.jsonl`, and `audit.json`; it exits 0 only when all audit gates, including minimum sample support, pass. Exit 3 means the audit completed but the holdout is not eligible. Exit 2 means invalid inputs or unsafe paths. Outputs do not contain absolute paths.

JSONL scanning is capped at 1,000,000 rows per manifest by default, with a maximum wall budget of 30 minutes. Use `--max-rows-per-manifest` and `--max-seconds` to set lower limits. Candidate paths must be relative to `--dataset-root`; their bytes are checked against `image_sha256` before eligibility is considered.

The required separate gates are `no_known_overlap`, `history_coverage`, `grouping_verified`, `source_verified`, `review_complete`, `sample_size_eligible`, and `holdout_eligible`. The command's `eligible.jsonl` is only the non-overlapping candidate subset; it must not be interpreted as a qualified holdout unless `holdout_eligible` is true. Promotion remains disabled by this audit tool.

## Candidate-set internal duplicate gate

Before computing support or eligibility, the audit checks identities and content duplicates inside the candidate manifest itself. Sample IDs are compared within their source namespace; file SHA256 and decoded-pixel SHA256 are compared globally across namespaces. Duplicate relations form connected groups, and every candidate in a duplicate group is excluded from support counts. A candidate manifest with any unresolved internal duplicate is ineligible, even when it otherwise reaches image and group count thresholds.

`group_id` remains a source-group identity, not a uniqueness key: distinct images in the same reliable group are retained and contribute at most one support group. The audit reports `candidate_content_unique`, duplicate group/row counts, stable group IDs, and blocker details separately from history overlap. Human review cannot override a machine-detected duplicate or restore excluded rows.
