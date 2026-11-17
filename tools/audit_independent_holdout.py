"""Strictly exclude known data exposure from a candidate holdout manifest.

All paths are supplied explicitly. The tool validates candidate file hashes,
compares exact identities with every supplied historical manifest, and keeps
human-attestation gates separate from machine checks. It never assigns or
repairs source labels, group IDs, or annotation review status.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import sys
import time
from typing import Any

from PIL import Image

from traffic_risk.prepare_evaluation import pixel_hash, validate_provenance_record


IDENTITY_FIELDS = ("sample_id", "group_id", "pixel_sha256", "image_sha256")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
REQUIRED_HISTORY_SCOPE = {
    "training", "development", "prior-holdouts", "error-analysis",
    "manual-review", "performance-samples",
}
REQUIRED_SUPPORT = {"person": (100, 25), "car": (100, 25), "traffic light": (100, 25)}
MAX_IMAGE_PIXELS = 80_000_000


def _read_jsonl(path: Path, *, max_rows: int, deadline: float) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("audit time budget reached while reading manifests")
            if not line.strip():
                continue
            if len(rows) >= max_rows:
                raise ValueError(f"manifest row limit exceeded: {path.name}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path.name}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest row must be an object at {path.name}:{line_number}")
            rows.append(row)
    return rows


def _sha256_file(path: Path, deadline: float) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if time.monotonic() >= deadline:
                raise TimeoutError("audit time budget reached while hashing files")
            digest.update(chunk)
    return digest.hexdigest()


def _read_provenance(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("provenance must be a JSON object")
    # Reuse the established schema validator; its source-verified status is an
    # attestation field and is not by itself accepted as substantive evidence.
    base = {key: value.get(key) for key in (
        "schema_version", "source_status", "provider", "archive_id", "source_url",
        "acquired_date", "acquisition_and_conversion", "source_archive_sha256",
        "verified_by", "verified_date", "annotation_review",
    )}
    validate_provenance_record(base)
    return value


def _validate_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _human_review_block(value: Any, *, status: str = "reviewed") -> bool:
    scope = value.get("scope") if isinstance(value, dict) else None
    scope_valid = isinstance(scope, str) and bool(scope.strip()) or isinstance(scope, list) and bool(scope)
    return (
        isinstance(value, dict)
        and value.get("status") == status
        and isinstance(value.get("reviewer"), str) and bool(value["reviewer"].strip())
        and _validate_date(value.get("reviewed_date"))
        and scope_valid
    )


def _verified_evidence(value: Any, provenance_path: Path, deadline: float) -> tuple[bool, list[dict[str, str]]]:
    if not isinstance(value, list) or not value:
        return False, []
    result = []
    for item in value:
        if not isinstance(item, dict):
            return False, []
        rel = item.get("path")
        digest = item.get("sha256")
        if not isinstance(rel, str) or not rel or not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            return False, []
        evidence_path = (provenance_path.parent / rel).resolve()
        if not evidence_path.is_relative_to(provenance_path.parent.resolve()) or not evidence_path.is_file():
            return False, []
        actual = _sha256_file(evidence_path, deadline)
        if actual.lower() != digest.lower():
            return False, []
        result.append({"file": Path(rel).as_posix(), "sha256": actual})
    return True, result


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _absolute_path(value: str) -> bool:
    return Path(value).is_absolute() or bool(Path(value).drive) or PureWindowsPath(value).is_absolute()


def _safe_output_row(row: dict[str, Any]) -> dict[str, Any]:
    """Remove absolute filesystem path metadata before writing local audit output."""
    safe = dict(row)
    for key, value in list(safe.items()):
        if isinstance(value, str) and (key.lower().endswith("path") or key.lower().endswith("file")):
            if _absolute_path(value):
                safe[key] = None
    return safe


def _identity(row: dict[str, Any], field: str) -> tuple[str, str] | str | None:
    value = row.get(field)
    if value is None or value == "":
        return None
    value = str(value)
    if field in {"sample_id", "group_id"}:
        namespace = row.get("source_namespace")
        if not isinstance(namespace, str) or not namespace.strip():
            return None
        return (namespace.strip(), value)
    return value.lower()


def _row_missing(row: dict[str, Any]) -> list[str]:
    missing = []
    for field in IDENTITY_FIELDS:
        value = row.get(field)
        if value is None or value == "":
            missing.append(field)
        elif field.endswith("sha256") and not _is_sha256(value):
            missing.append(field)
    if not isinstance(row.get("source_namespace"), str) or not row["source_namespace"].strip():
        missing.append("source_namespace")
    if not isinstance(row.get("split"), str) or not row["split"].strip():
        missing.append("split")
    return missing


def _grouping_unreliable(row: dict[str, Any]) -> bool:
    grouping = str(row.get("grouping", "")).strip().lower().replace("_", "-")
    return not grouping or grouping in {"image-id-fallback", "image-id", "unknown", "none"}


def _sample_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    duplicate_info = _validate_unique_candidates(rows)
    duplicate_rows = set(duplicate_info["row_fields"])
    usable_rows = [row for index, row in enumerate(rows) if index not in duplicate_rows]
    groups = {_identity(row, "group_id") for row in usable_rows}
    class_support: dict[str, dict[str, Any]] = {}
    for category, (min_instances, min_groups) in REQUIRED_SUPPORT.items():
        instance_count = 0
        supporting_groups: set[Any] = set()
        for row in usable_rows:
            objects = row.get("objects") or []
            if not isinstance(objects, list):
                continue
            matching = [obj for obj in objects if isinstance(obj, dict)
                        and str(obj.get("category", "")).lower().replace("_", " ") == category]
            if matching:
                instance_count += len(matching)
                supporting_groups.add(_identity(row, "group_id"))
        class_support[category] = {
            "instances": instance_count,
            "groups": len(supporting_groups),
            "required_instances": min_instances,
            "required_groups": min_groups,
            "eligible": instance_count >= min_instances and len(supporting_groups) >= min_groups,
        }
    return {
        "images": len(usable_rows), "groups": len(groups),
        "duplicate_rows": len(duplicate_rows), "content_unique": not duplicate_rows,
        "minimum_images": 500, "minimum_groups": 100,
        "class_support": class_support,
        "eligible": len(usable_rows) >= 500 and len(groups) >= 100
        and all(value["eligible"] for value in class_support.values()),
    }


def _verify_candidate_files(rows: list[dict[str, Any]], dataset_root: Path, deadline: float) -> list[str]:
    problems = []
    root = dataset_root.resolve()
    for index, row in enumerate(rows):
        if time.monotonic() >= deadline:
            raise TimeoutError("audit time budget reached while checking candidate files")
        rel = row.get("image_path")
        if not isinstance(rel, str) or not rel:
            problems.append(f"candidate[{index}]:missing_image_path")
            continue
        if _absolute_path(rel) or Path(rel).root:
            problems.append(f"candidate[{index}]:absolute_image_path")
            continue
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            problems.append(f"candidate[{index}]:image_path_outside_dataset_root")
            continue
        if not path.is_file():
            problems.append(f"candidate[{index}]:image_file_missing")
            continue
        if _sha256_file(path, deadline).lower() != str(row.get("image_sha256", "")).lower():
            problems.append(f"candidate[{index}]:image_sha256_mismatch")
        try:
            with Image.open(path) as source:
                width, height = source.size
                if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                    problems.append(f"candidate[{index}]:image_dimensions_out_of_bounds")
                    continue
                # The preparer hashes decoded RGB without EXIF transpose.
                source.load()
                image = source.convert("RGB")
                actual_pixel_hash = pixel_hash(image)
        except (OSError, ValueError, Image.DecompressionBombError):
            problems.append(f"candidate[{index}]:pixel_sha256_decode_error")
            continue
        if actual_pixel_hash.lower() != str(row.get("pixel_sha256", "")).lower():
            problems.append(f"candidate[{index}]:pixel_sha256_mismatch")
    return problems


def _validate_unique_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Find connected duplicate sets across namespaced IDs and global content hashes.

    Every row in a duplicate set is rejected. A shared group_id is deliberately
    not part of this index: distinct images from one source group are valid.
    """
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    keyed_rows: dict[tuple[str, Any], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        for field in ("sample_id", "image_sha256", "pixel_sha256"):
            value = _identity(row, field)
            if value is not None:
                keyed_rows[(field, value)].append(index)
    for indices in keyed_rows.values():
        if len(indices) > 1:
            first = indices[0]
            for index in indices[1:]:
                union(first, index)

    components: dict[int, set[int]] = defaultdict(set)
    for index in range(len(rows)):
        components[find(index)].add(index)
    duplicate_components = [members for members in components.values() if len(members) > 1]
    groups = []
    row_fields: dict[int, set[str]] = defaultdict(set)
    for members in duplicate_components:
        repeated_keys = [key for key, indices in keyed_rows.items()
                         if len(indices) > 1 and any(index in members for index in indices)]
        fields = sorted({field for field, _ in repeated_keys})
        for field, value in repeated_keys:
            for index in keyed_rows[(field, value)]:
                if index in members:
                    row_fields[index].add(field)
        stable_keys = sorted(
            (field, list(value) if isinstance(value, tuple) else value)
            for field, value in repeated_keys
        )
        group_id = hashlib.sha256(json.dumps(stable_keys, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        stable_members = sorted(
            (str(rows[index].get("source_namespace", "")), str(rows[index].get("sample_id", "")),
             str(rows[index].get("image_sha256", "")).lower(), str(rows[index].get("pixel_sha256", "")).lower())
            for index in members
        )
        groups.append({"group_id": group_id, "member_indices": sorted(members),
                       "members": [list(member) for member in stable_members], "fields": fields})
    groups.sort(key=lambda item: item["group_id"])
    return {"groups": groups, "row_fields": row_fields}


def _check_inputs_do_not_overlap_output(output: Path, paths: list[Path], dataset_root: Path) -> None:
    out = output.resolve()
    data_root = dataset_root.resolve()
    if out == data_root or out.is_relative_to(data_root) or data_root.is_relative_to(out):
        raise ValueError("output directory must be outside dataset_root")
    for path in paths:
        source = path.resolve()
        if source == out or source.is_relative_to(out) or out.is_relative_to(source):
            raise ValueError("output directory overlaps an input file or directory")


def audit(
    candidate_manifest: Path,
    history_manifests: list[Path],
    provenance_path: Path,
    dataset_root: Path,
    output: Path,
    *,
    max_rows_per_manifest: int = 1_000_000,
    max_seconds: float = 1800,
) -> dict[str, Any]:
    candidate_manifest = candidate_manifest.resolve()
    history_manifests = [path.resolve() for path in history_manifests]
    provenance_path = provenance_path.resolve()
    dataset_root = dataset_root.resolve()
    output = output.resolve()
    if max_rows_per_manifest <= 0 or max_rows_per_manifest > 1_000_000:
        raise ValueError("max_rows_per_manifest must be in (0, 1000000]")
    if max_seconds <= 0 or max_seconds > 1800:
        raise ValueError("max_seconds must be in (0, 1800]")
    deadline = time.monotonic() + max_seconds
    if not history_manifests:
        raise ValueError("at least one --history-manifest is required; history coverage still needs human evidence")
    if not candidate_manifest.is_file() or not provenance_path.is_file() or any(not p.is_file() for p in history_manifests):
        raise FileNotFoundError("candidate, history, and provenance inputs must be existing files")
    if not dataset_root.is_dir():
        raise NotADirectoryError("dataset_root must be an existing directory")
    inputs = [candidate_manifest, *history_manifests, provenance_path]
    _check_inputs_do_not_overlap_output(output, inputs, dataset_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")

    candidates = _read_jsonl(candidate_manifest, max_rows=max_rows_per_manifest, deadline=deadline)
    histories = [(path, _read_jsonl(path, max_rows=max_rows_per_manifest, deadline=deadline))
                 for path in history_manifests]
    provenance = _read_provenance(provenance_path)
    file_problems = _verify_candidate_files(candidates, dataset_root, deadline)
    duplicate_info = _validate_unique_candidates(candidates)
    duplicate_groups = duplicate_info["groups"]
    duplicate_row_fields = duplicate_info["row_fields"]
    duplicate_problems = [
        f"candidate[{index}]:{field}"
        for index, fields in sorted(duplicate_row_fields.items())
        for field in sorted(fields)
    ]
    file_problem_rows: dict[int, list[str]] = defaultdict(list)
    for problem in file_problems:
        match = re.match(r"candidate\[(\d+)\]:(.*)", problem)
        if match:
            file_problem_rows[int(match.group(1))].append(match.group(2))
    history_index: dict[str, dict[Any, list[dict[str, Any]]]] = {field: defaultdict(list) for field in IDENTITY_FIELDS}
    history_missing = []
    history_grouping_fallbacks = []
    history_manifest_records = []
    for manifest_path, rows in histories:
        manifest_records = {"name": manifest_path.name, "sha256": _sha256_file(manifest_path, deadline), "rows": len(rows)}
        history_manifest_records.append(manifest_records)
        for index, row in enumerate(rows):
            for field in IDENTITY_FIELDS:
                key = _identity(row, field)
                if key is not None:
                    history_index[field][key].append(row)
            missing = _row_missing(row)
            if missing:
                history_missing.append({"manifest": manifest_path.name, "row": index, "missing": missing})
            if _grouping_unreliable(row):
                history_grouping_fallbacks.append({"manifest": manifest_path.name, "row": index})

    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    candidate_missing = []
    candidate_grouping_fallbacks = []
    overlap_count = 0
    for index, row in enumerate(candidates):
        reasons = []
        reasons.extend(f"candidate_file:{problem}" for problem in file_problem_rows.get(index, []))
        for field in sorted(duplicate_row_fields.get(index, set())):
            reasons.append("duplicate_sample_id" if field == "sample_id" else "duplicate_content")
        missing = _row_missing(row)
        if missing:
            candidate_missing.append({"row": index, "missing": missing})
            reasons.extend(f"missing_field:{field}" if field == "split" else f"missing_identity:{field}"
                           for field in missing)
        if _grouping_unreliable(row):
            candidate_grouping_fallbacks.append(index)
            reasons.append("unreliable_grouping")
        matching = []
        for field in IDENTITY_FIELDS:
            key = _identity(row, field)
            if key is not None and key in history_index[field]:
                prior_rows = history_index[field][key]
                roles = sorted({str(prior.get("split", "unknown")) for prior in prior_rows})
                matching.append({"field": field, "prior_roles": roles,
                                 "count": len(prior_rows), "sha256_of_identity": hashlib.sha256(
                                     json.dumps(key, separators=(",", ":") if isinstance(key, tuple) else None,
                                                ensure_ascii=False).encode()).hexdigest()})
                reasons.append(f"historical_overlap:{field}")
        if matching:
            overlap_count += 1
        if reasons:
            excluded.append({"candidate": _safe_output_row(row), "reasons": sorted(set(reasons)), "overlaps": matching})
        else:
            eligible.append(_safe_output_row(row))

    provenance_history = provenance.get("history_coverage")
    inventory_expected = []
    if isinstance(provenance_history, dict) and isinstance(provenance_history.get("manifest_inventory"), list):
        inventory_expected = sorted(
            (str(item.get("name", "")), str(item.get("sha256", "")).lower())
            for item in provenance_history["manifest_inventory"] if isinstance(item, dict)
        )
    inventory_actual = sorted((item["name"], item["sha256"].lower()) for item in history_manifest_records)
    inventory_matches = bool(inventory_actual) and inventory_expected == inventory_actual
    scope = set(provenance_history.get("scope", [])) if isinstance(provenance_history, dict) and isinstance(
        provenance_history.get("scope"), list) else set()
    history_review_ok = (
        isinstance(provenance_history, dict)
        and provenance_history.get("status") == "complete"
        and _human_review_block(provenance_history, status="complete")
        and REQUIRED_HISTORY_SCOPE.issubset(scope)
        and inventory_matches
    )
    history_coverage = history_review_ok and not history_missing

    source_evidence_ok, source_evidence_records = _verified_evidence(
        provenance.get("source_evidence"), provenance_path, deadline
    )
    source_verified = (
        provenance.get("source_status") == "source-verified"
        and source_evidence_ok
        and bool(provenance.get("provider")) and bool(provenance.get("archive_id"))
        and bool(provenance.get("source_archive_sha256"))
    )
    annotation_review = provenance.get("annotation_review")
    annotation_evidence_ok, annotation_evidence_records = _verified_evidence(
        provenance.get("annotation_evidence"), provenance_path, deadline
    )
    review_complete = _human_review_block(annotation_review) and annotation_evidence_ok
    grouping_review = provenance.get("grouping_review")
    grouping_evidence_ok, grouping_evidence_records = _verified_evidence(
        provenance.get("grouping_evidence"), provenance_path, deadline
    )
    grouping_verified = (
        not candidate_missing and not history_missing
        and not candidate_grouping_fallbacks and not history_grouping_fallbacks
        and _human_review_block(grouping_review) and grouping_evidence_ok
    )
    no_known_overlap = overlap_count == 0 and not candidate_missing and not history_missing
    sample_support = _sample_support(eligible)
    holdout_eligible = (
        bool(eligible) and not file_problems and not duplicate_problems and no_known_overlap
        and history_coverage and grouping_verified and source_verified and review_complete
        and sample_support["eligible"]
    )
    output.mkdir(parents=True, exist_ok=False)
    eligible_path = output / "eligible.jsonl"
    excluded_path = output / "excluded.jsonl"
    eligible_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in eligible),
                             encoding="utf-8")
    excluded_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in excluded),
                             encoding="utf-8")
    audit_record = {
        "schema_version": 1,
        "candidate_manifest": {"name": candidate_manifest.name, "sha256": _sha256_file(candidate_manifest, deadline),
                               "rows": len(candidates)},
        "history_manifests": history_manifest_records,
        "provenance": {"name": provenance_path.name, "sha256": _sha256_file(provenance_path, deadline)},
        "evidence": {"source": source_evidence_records, "annotation": annotation_evidence_records,
                     "grouping": grouping_evidence_records},
        "hashes": {"eligible.jsonl": _sha256_file(eligible_path, deadline),
                   "excluded.jsonl": _sha256_file(excluded_path, deadline)},
        "counts": {"candidate": len(candidates), "eligible": len(eligible), "excluded": len(excluded),
                   "historical_overlap_candidates": overlap_count},
        "candidate_content_unique": {
            "unique": not duplicate_groups,
            "duplicate_groups": len(duplicate_groups),
            "duplicate_rows": len(duplicate_row_fields),
            "groups": duplicate_groups,
        },
        "support": sample_support,
        "gates": {
            "no_known_overlap": no_known_overlap,
            "history_coverage": bool(history_coverage),
            "grouping_verified": bool(grouping_verified),
            "source_verified": bool(source_verified),
            "review_complete": bool(review_complete),
            "sample_size_eligible": bool(sample_support["eligible"]),
            "candidate_content_unique": not duplicate_groups,
            "holdout_eligible": bool(holdout_eligible),
        },
        "blockers": sorted(
            (["candidate identities overlap a historical manifest"] if not no_known_overlap else [])
            + (["history coverage attestation or manifest inventory is incomplete"] if not history_coverage else [])
            + (["grouping completeness or human evidence is incomplete"] if not grouping_verified else [])
            + (["source verification evidence is missing or incomplete"] if not source_verified else [])
            + (["annotation review evidence is missing or incomplete"] if not review_complete else [])
            + (["eligible sample count, group count, or class support is below the minimum"] if not sample_support["eligible"] else [])
            + (["candidate file checks failed"] if file_problems else [])
            + (["candidate set contains duplicate identities or content"] if duplicate_groups else [])
            + (["candidate or historical identities are incomplete"] if candidate_missing or history_missing else [])
            + (["candidate grouping uses an unreliable fallback"] if candidate_grouping_fallbacks else [])
        ),
        "input_issues": {"candidate_file_checks": file_problems, "duplicate_candidates": duplicate_problems,
                         "candidate_missing_identity": candidate_missing, "history_missing_identity": history_missing,
                         "candidate_grouping_fallbacks": candidate_grouping_fallbacks,
                         "history_grouping_fallbacks": history_grouping_fallbacks,
                         "history_inventory_matches": inventory_matches},
        "identity_rules": {
            "fields": list(IDENTITY_FIELDS),
            "sample_id_and_group_id": "compared within source_namespace",
            "pixel_sha256_and_image_sha256": "compared globally across namespaces",
            "same_role_overlaps": "excluded exactly like cross-role overlaps",
        },
        "promotion_allowed": False,
        "notes": [
            "Hash equality detects exact file/pixel identity only; it does not detect adjacent frames or perceptual near-duplicates.",
            "Evidence hashes verify referenced bytes, not the truth of their contents; reviewer attestations remain human responsibilities.",
            "Eligible rows are candidates with no known overlap; the holdout_eligible gate also requires all evidence gates.",
        ],
    }
    (output / "audit.json").write_text(json.dumps(audit_record, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                                        encoding="utf-8")
    return audit_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--history-manifest", type=Path, action="append", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="root used to check candidate image_path and image_sha256")
    parser.add_argument("--max-rows-per-manifest", type=int, default=1_000_000,
                        help="bounded JSONL rows per input (maximum 1000000)")
    parser.add_argument("--max-seconds", type=float, default=1800,
                        help="bounded wall time in seconds (maximum 1800)")
    parser.add_argument("--output", type=Path, required=True, help="new, non-existing output directory")
    args = parser.parse_args(argv)
    try:
        result = audit(args.candidate_manifest, args.history_manifest, args.provenance,
                       args.dataset_root, args.output,
                       max_rows_per_manifest=args.max_rows_per_manifest,
                       max_seconds=args.max_seconds)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"holdout audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"counts": result["counts"], "gates": result["gates"],
                      "output": args.output.name}, indent=2))
    return 0 if result["gates"]["holdout_eligible"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
