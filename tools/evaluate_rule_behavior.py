"""Bounded audit of rule traces and baseline-to-candidate output changes.

Inputs are UTF-8 JSONL prediction records containing sample_id, role, status,
risk, reason_code, and features. `role` must be development, stress, or
diagnostic; holdout rows are rejected to keep this tool away from reserved data.
Paths are resolved from the current workspace root. Outputs omit image paths.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

from traffic_risk.hard_rules import RULE_REASONS, _trace_hard_rules
from traffic_risk.inference_outcome import PROCESSING_FAILURE_CODES, REASON_CODES
from traffic_risk.schema import MODEL_FEATURES


ALLOWED_ROLES = {"development", "stress", "diagnostic"}
NORMAL_ABSTENTION_CODES = {"no_supported_detections", "no_rule_triggered"}
DEFAULT_MAX_ROWS = 2000
DEFAULT_MAX_SECONDS = 1800
MAX_REVIEW_QUEUE = 200
MAX_MANIFEST_BYTES = 100 * 1024 * 1024


def sha256_file(path: Path, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("rule evaluation time budget exceeded while hashing inputs")
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path, max_rows: int, max_bytes: int, deadline: float) -> tuple[list[dict[str, Any]], int]:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"input exceeds configured max_input_bytes={max_bytes}: {path.name}")
    rows: list[dict[str, Any]] = []
    missing_reason_code = 0
    total_bytes = 0
    with path.open("rb") as stream:
        line_number = 0
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("rule evaluation time budget exceeded while reading inputs")
            raw_line = stream.readline(max_bytes + 1)
            if not raw_line:
                break
            line_number += 1
            total_bytes += len(raw_line)
            if len(raw_line) > max_bytes or total_bytes > max_bytes:
                raise ValueError(f"input exceeds configured max_input_bytes={max_bytes}: {path.name}")
            if time.monotonic() > deadline:
                raise TimeoutError("rule evaluation time budget exceeded while reading inputs")
            try:
                line = raw_line.decode("utf-8-sig" if line_number == 1 else "utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"invalid UTF-8 at {path.name}:{line_number}") from exc
            if not line.strip():
                continue
            if len(rows) >= max_rows:
                raise ValueError(f"input exceeds configured max_rows={max_rows}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path.name}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"input row must be a JSON object at {path.name}:{line_number}")
            if not row.get("reason_code"):
                missing_reason_code += 1
                row["reason_code"] = "legacy_unspecified"
            if row["reason_code"] not in REASON_CODES:
                row["reason_code"] = "legacy_unspecified"
            role = row.get("role")
            if role == "holdout" or row.get("split") == "holdout":
                raise ValueError("reserved holdout rows are not accepted by the rule behavior tool")
            if role not in ALLOWED_ROLES:
                raise ValueError("every input row must set role to development, stress, or diagnostic")
            sample_id = row.get("sample_id")
            namespace = row.get("source_namespace")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"missing sample_id at {path.name}:{line_number}")
            if not isinstance(namespace, str) or not namespace.strip():
                raise ValueError(f"missing source_namespace at {path.name}:{line_number}")
            if not isinstance(row.get("features"), dict):
                row["features"] = {}
            if row.get("status") not in {"ok", "uncertain"}:
                row["status"] = "uncertain"
                row["risk"] = "unknown"
                row["reason_code"] = "legacy_unspecified"
            if row.get("risk") not in {"low", "medium", "high", "unknown"}:
                row["risk"] = "unknown"
                row["reason_code"] = "legacy_unspecified"
            rows.append(row)
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate sample_id values in {path.name}")
    return rows, missing_reason_code


def _label(row: dict[str, Any]) -> str:
    return f"{row.get('status', 'uncertain')}|{row.get('risk', 'unknown')}|{row.get('reason_code', 'legacy_unspecified')}"


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rule_counts: Counter[str] = Counter()
    first_counts: Counter[str] = Counter()
    overlap_sets: Counter[str] = Counter()
    high = unknown = 0
    by_role: dict[str, Counter[str]] = {}
    reason_codes: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    abstentions: Counter[str] = Counter()
    overlapping_samples = 0
    feature_complete = 0
    feature_incomplete = 0
    for row in rows:
        role = row["role"]
        role_counts = by_role.setdefault(role, Counter())
        risk = row.get("risk", "unknown")
        role_counts[f"risk:{risk}"] += 1
        role_counts["rows"] += 1
        high += risk == "high"
        unknown += risk == "unknown"
        code = row.get("reason_code", "legacy_unspecified")
        reason_codes[code] += 1
        if code in PROCESSING_FAILURE_CODES:
            failures[code] += 1
        if code in NORMAL_ABSTENTION_CODES:
            abstentions[code] += 1
        if all(name in row["features"] for name in MODEL_FEATURES):
            feature_complete += 1
            matches = _trace_hard_rules(row["features"])
        else:
            feature_incomplete += 1
            matches = []
        rule_counts.update(matches)
        if matches:
            first_counts[matches[0]] += 1
        if len(matches) > 1:
            overlapping_samples += 1
            overlap_sets[" + ".join(matches)] += 1
    return {
        "rows": len(rows),
        "feature_complete_rows": feature_complete,
        "feature_incomplete_rows": feature_incomplete,
        "risk_distribution": {"high": high, "unknown": unknown,
                              "low": sum(r.get("risk") == "low" for r in rows),
                              "medium": sum(r.get("risk") == "medium" for r in rows)},
        "reason_code_distribution": dict(sorted(reason_codes.items())),
        "normal_abstentions": dict(sorted(abstentions.items())),
        "pipeline_failures": dict(sorted(failures.items())),
        "rule_hit_counts": {rule_id: rule_counts[rule_id] for rule_id in RULE_REASONS},
        "first_match_counts": {rule_id: first_counts[rule_id] for rule_id in RULE_REASONS},
        "multiple_rule_match_samples": overlapping_samples,
        "overlap_combinations": dict(sorted(overlap_sets.items())),
        "by_role": {role: dict(sorted(counts.items())) for role, counts in sorted(by_role.items())},
    }


def _sample_key(row: dict[str, Any]) -> tuple[str, str]:
    return row["source_namespace"].strip(), row["sample_id"]


def _transition_summary(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    before = {_sample_key(row): row for row in baseline}
    after = {_sample_key(row): row for row in candidate}
    table: Counter[str] = Counter()
    changed: list[dict[str, Any]] = []
    for sample_key in sorted(before.keys() | after.keys()):
        left, right = before.get(sample_key), after.get(sample_key)
        namespace, sample_id = sample_key
        from_label = _label(left) if left else "missing"
        to_label = _label(right) if right else "missing"
        transition = f"{from_label} -> {to_label}"
        table[transition] += 1
        if from_label != to_label:
            source = right or left or {}
            changed.append({"source_namespace": namespace, "sample_id": sample_id, "role": source.get("role"),
                            "baseline": from_label, "candidate": to_label})
    return {"paired_sample_ids": len(before.keys() & after.keys()),
            "baseline_only": len(before.keys() - after.keys()),
            "candidate_only": len(after.keys() - before.keys()),
            "changed": len(changed), "transition_table": dict(sorted(table.items()))}, changed


def _review_queue(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]],
                  changed: list[dict[str, Any]], seed: int, limit: int) -> list[dict[str, Any]]:
    if limit == 0:
        return []
    by_id = {_sample_key(row): row for row in [*baseline, *candidate]}
    changed_ids = {(row["source_namespace"], row["sample_id"]) for row in changed}
    eligible = []
    for sample_key, row in by_id.items():
        namespace, sample_id = sample_key
        trace = (_trace_hard_rules(row["features"])
                 if all(name in row["features"] for name in MODEL_FEATURES) else [])
        if (sample_key in changed_ids or trace or row.get("risk") == "unknown"
                or row.get("reason_code") in PROCESSING_FAILURE_CODES):
            eligible.append((namespace, sample_id, row, trace))
    # Keep at most one reviewed item per known group; unknown groups remain separate.
    randomizer = random.Random(seed)
    eligible.sort(key=lambda item: (item[0], item[1]))
    randomizer.shuffle(eligible)
    queue: list[dict[str, Any]] = []
    seen_groups: set[tuple[str, str]] = set()
    seen_ids: set[tuple[str, str]] = set()
    for namespace, sample_id, row, matches in eligible:
        sample_key = (namespace, sample_id)
        group = row.get("group_id")
        group_key = (namespace, group) if group else None
        if group_key and group_key in seen_groups:
            continue
        if sample_key in seen_ids:
            continue
        seen_ids.add(sample_key)
        if group_key:
            seen_groups.add(group_key)
        queue.append({"source_namespace": namespace, "sample_id": sample_id, "role": row["role"],
                      "rule_matches": matches, "review_status": "awaiting-human-review"})
        if len(queue) >= limit:
            break
    return queue


def evaluate(config_path: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config_path = (root / config_path).resolve() if not config_path.is_absolute() else config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    required = {"baseline_manifest", "candidate_manifest", "seed", "max_rows", "max_seconds"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing config fields: {', '.join(missing)}")
    if config["seed"] != 42:
        raise ValueError("this plan fixes seed=42")
    max_rows = int(config["max_rows"])
    max_seconds = float(config["max_seconds"])
    max_input_bytes = int(config.get("max_input_bytes", MAX_MANIFEST_BYTES))
    queue_limit = int(config.get("max_review_queue", MAX_REVIEW_QUEUE))
    if not 1 <= max_rows <= DEFAULT_MAX_ROWS:
        raise ValueError(f"max_rows must be in [1, {DEFAULT_MAX_ROWS}]")
    if not 1 <= max_seconds <= DEFAULT_MAX_SECONDS:
        raise ValueError(f"max_seconds must be in [1, {DEFAULT_MAX_SECONDS}]")
    if not 1 <= max_input_bytes <= MAX_MANIFEST_BYTES:
        raise ValueError(f"max_input_bytes must be in [1, {MAX_MANIFEST_BYTES}]")
    if not 0 <= queue_limit <= MAX_REVIEW_QUEUE:
        raise ValueError(f"max_review_queue must be in [0, {MAX_REVIEW_QUEUE}]")
    paths = {}
    for field in ("baseline_manifest", "candidate_manifest"):
        path = Path(config[field])
        paths[field] = path.resolve() if path.is_absolute() else (root / path).resolve()
        if not paths[field].is_file():
            raise FileNotFoundError(f"{field} must be an existing file")
    output_path = output.resolve() if output.is_absolute() else (root / output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_path}")
    for source in [config_path, *paths.values()]:
        if source == output_path or source.is_relative_to(output_path) or output_path.is_relative_to(source):
            raise ValueError("output directory must not overlap an input file or directory")
    deadline = time.monotonic() + max_seconds
    baseline, missing_baseline_codes = _read_rows(paths["baseline_manifest"], max_rows, max_input_bytes, deadline)
    candidate, missing_candidate_codes = _read_rows(paths["candidate_manifest"], max_rows, max_input_bytes, deadline)
    transition, changed = _transition_summary(baseline, candidate)
    queue = _review_queue(baseline, candidate, changed, int(config["seed"]), queue_limit)
    report = {
        "schema_version": 1,
        "inputs": {field: {"name": paths[field].name, "sha256": sha256_file(paths[field], deadline)}
                   for field in paths},
        "config": {"sha256": sha256_file(config_path, deadline), "seed": config["seed"],
                   "max_rows": max_rows, "max_seconds": max_seconds,
                   "max_input_bytes": max_input_bytes, "max_review_queue": queue_limit},
        "code_sha256": {"tool": sha256_file(Path(__file__).resolve(), deadline),
                        "hard_rules": sha256_file(root / "src/traffic_risk/hard_rules.py", deadline),
                        "inference_outcome": sha256_file(root / "src/traffic_risk/inference_outcome.py", deadline)},
        "baseline": _summarize(baseline),
        "candidate": _summarize(candidate),
        "output_transitions": transition,
        "input_quality": {
            "baseline_missing_reason_code": missing_baseline_codes,
            "candidate_missing_reason_code": missing_candidate_codes,
            "legacy_unspecified_is_failure": True,
            "holdout_rows_accessed": 0,
        },
        "review_queue": {"count": len(queue), "status": "awaiting-human-review",
                         "contains_model_outputs_only": True},
        "promotion_allowed": False,
        "notes": [
            "Rule counts describe matches to the existing deterministic predicates; they do not measure real-world safety or accident risk.",
            "Unknown is reported separately from processing failures; neither is treated as an accuracy score.",
            "Input rows must be development, stress, or diagnostic. This tool rejects reserved holdout rows.",
            "The review queue has no human labels and must not be reported as reviewed evidence.",
        ],
    }
    output_path.mkdir(parents=True, exist_ok=False)
    queue_path = output_path / "review_queue.jsonl"
    queue_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in queue), encoding="utf-8")
    report["review_queue"]["sha256"] = sha256_file(queue_path, deadline)
    (output_path / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                                               encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing paths are rejected")
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.config, args.output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, TimeoutError) as exc:
        print(f"rule behavior evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"baseline_rows": result["baseline"]["rows"],
                      "candidate_rows": result["candidate"]["rows"],
                      "transitions": result["output_transitions"]["changed"],
                      "review_queue": result["review_queue"]["count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
