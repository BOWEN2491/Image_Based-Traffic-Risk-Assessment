from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

import pytest
from PIL import Image

from traffic_risk.prepare_evaluation import pixel_hash
from tools.audit_independent_holdout import _sample_support, audit


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _row(root: Path, name: str, *, sample_id: str | None = None, group_id: str | None = None,
         image_bytes: bytes | None = None, image_sha256: str | None = None,
         pixel_sha256: str | None = None, namespace: str = "bdd100k", split: str = "development",
         grouping: str = "rideID", include_group: bool = True) -> dict:
    if image_bytes is None:
        color = tuple(hashlib.sha256(name.encode()).digest()[:3])
        image = Image.new("RGB", (7, 5), color)
        stream = BytesIO()
        image.save(stream, format="PNG")
        image_bytes = stream.getvalue()
    else:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    path = root / f"{name}.png"
    path.write_bytes(image_bytes)
    row = {
        "sample_id": sample_id or f"val/{name}",
        "group_id": group_id or f"ride:{name}",
        "source_namespace": namespace,
        "image_path": path.name,
        "image_sha256": image_sha256 or _sha(image_bytes),
        "pixel_sha256": pixel_sha256 or pixel_hash(image),
        "split": split,
        "grouping": grouping,
        "objects": [],
    }
    if not include_group:
        row.pop("group_id")
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _provenance(path: Path, **overrides) -> Path:
    value = {
        "schema_version": 1,
        "source_status": "local-unverified",
        "provider": None,
        "archive_id": None,
        "source_url": None,
        "acquired_date": None,
        "acquisition_and_conversion": [],
        "source_archive_sha256": None,
        "verified_by": None,
        "verified_date": None,
        "annotation_review": {
            "status": "awaiting-human-review", "reviewer": None,
            "reviewed_date": None, "scope": None,
        },
        "history_coverage": {"status": "incomplete"},
    }
    value.update(overrides)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _run(tmp_path: Path, candidates: list[dict], history: list[dict], *, provenance: dict | None = None,
         name: str = "run") -> dict:
    candidate_path = _write_jsonl(tmp_path / "candidate.jsonl", candidates)
    history_path = _write_jsonl(tmp_path / "history.jsonl", history)
    provenance_path = _provenance(tmp_path / "provenance.json", **(provenance or {}))
    return audit(candidate_path, [history_path], provenance_path, tmp_path / "data", tmp_path / name)


def test_same_role_historical_holdout_is_excluded(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    candidate = _row(root, "candidate", sample_id="same-id", group_id="same-group", split="holdout")
    history = {**candidate, "image_path": "historical-copy.jpg", "split": "holdout"}
    # Historical files are not read, so its path may refer to an older archive.
    result = _run(tmp_path, [candidate], [history])

    excluded = json.loads((tmp_path / "run/excluded.jsonl").read_text().splitlines()[0])
    assert "historical_overlap:sample_id" in excluded["reasons"]
    assert "historical_overlap:group_id" in excluded["reasons"]
    assert result["gates"]["no_known_overlap"] is False


@pytest.mark.parametrize("field", ["sample_id", "group_id", "pixel_sha256", "image_sha256"])
def test_each_identity_field_excludes_overlap_across_roles(tmp_path, field):
    root = tmp_path / "data"
    root.mkdir()
    candidate = _row(root, "candidate")
    history = _row(root, "historical", split="development")
    history[field] = candidate[field]
    if field == "image_sha256":
        # The candidate file itself must still hash to its declared identity.
        history["image_sha256"] = candidate["image_sha256"]
    result = _run(tmp_path, [candidate], [history])

    excluded = json.loads((tmp_path / "run/excluded.jsonl").read_text().splitlines()[0])
    assert f"historical_overlap:{field}" in excluded["reasons"]
    assert result["counts"]["eligible"] == 0


def test_identity_namespaces_prevent_cross_source_id_collision(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    candidate = _row(root, "candidate", sample_id="same", group_id="group", namespace="source-a")
    history = _row(root, "historical", sample_id="same", group_id="group", namespace="source-b")

    result = _run(tmp_path, [candidate], [history])

    assert result["counts"]["eligible"] == 1
    assert result["gates"]["no_known_overlap"] is True


def test_candidate_missing_group_is_not_eligible_and_disables_group_gate(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    candidate = _row(root, "candidate", include_group=False)
    history = _row(root, "historical")

    result = _run(tmp_path, [candidate], [history])

    assert result["counts"]["eligible"] == 0
    assert result["gates"]["grouping_verified"] is False
    assert result["gates"]["no_known_overlap"] is False


def test_fallback_grouping_is_not_group_verified(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    candidate = _row(root, "candidate", grouping="image-id-fallback")
    history = _row(root, "historical")

    result = _run(tmp_path, [candidate], [history])

    assert result["gates"]["grouping_verified"] is False
    excluded = json.loads((tmp_path / "run/excluded.jsonl").read_text().splitlines()[0])
    assert "unreliable_grouping" in excluded["reasons"]


def test_missing_historical_identity_blocks_no_overlap_and_coverage(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    candidate = _row(root, "candidate")
    history = _row(root, "historical", include_group=False)

    result = _run(tmp_path, [candidate], [history])

    assert result["gates"]["no_known_overlap"] is False
    assert result["gates"]["history_coverage"] is False
    assert result["input_issues"]["history_missing_identity"]


def test_missing_source_namespace_is_not_inferred(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    candidate = _row(root, "candidate")
    candidate.pop("source_namespace")
    history = _row(root, "historical")

    result = _run(tmp_path, [candidate], [history])

    assert result["gates"]["no_known_overlap"] is False
    assert result["counts"]["eligible"] == 0
    assert "source_namespace" in result["input_issues"]["candidate_missing_identity"][0]["missing"]


def test_duplicate_candidate_sample_id_is_blocked(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    first = _row(root, "candidate-1", sample_id="same-id")
    second = _row(root, "candidate-2", sample_id="same-id")
    history = _row(root, "historical")

    result = _run(tmp_path, [first, second], [history])

    assert result["input_issues"]["duplicate_candidates"]
    assert result["gates"]["holdout_eligible"] is False
    excluded = [json.loads(line) for line in (tmp_path / "run/excluded.jsonl").read_text().splitlines()]
    assert any("duplicate_sample_id" in row["reasons"] for row in excluded)


@pytest.mark.parametrize("identity", ["image_sha256", "pixel_sha256"])
def test_duplicate_candidate_content_excludes_every_member(tmp_path, identity):
    root = tmp_path / "data"
    root.mkdir()
    first = _row(root, "candidate-1", sample_id="one", group_id="group-1")
    second = _row(root, "candidate-2", sample_id="two", group_id="group-2",
                  image_bytes=(root / "candidate-1.png").read_bytes())
    history = _row(root, "historical")

    result = _run(tmp_path, [first, second], [history])

    assert result["counts"]["eligible"] == 0
    assert result["candidate_content_unique"]["unique"] is False
    assert result["candidate_content_unique"]["duplicate_rows"] == 2
    assert result["candidate_content_unique"]["duplicate_groups"] == 1
    excluded = [json.loads(line) for line in (tmp_path / "run/excluded.jsonl").read_text().splitlines()]
    assert len(excluded) == 2
    assert all("duplicate_content" in row["reasons"] for row in excluded)
    assert result["gates"]["holdout_eligible"] is False


def test_different_file_encodings_with_same_pixels_are_duplicates(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    first = _row(root, "candidate-png")
    with Image.open(root / "candidate-png.png") as image:
        encoded = BytesIO()
        image.save(encoded, format="BMP")
    second = _row(root, "candidate-bmp", image_bytes=encoded.getvalue())
    history = _row(root, "historical")

    result = _run(tmp_path, [first, second], [history])

    assert first["image_sha256"] != second["image_sha256"]
    assert first["pixel_sha256"] == second["pixel_sha256"]
    assert result["candidate_content_unique"]["duplicate_rows"] == 2
    assert result["candidate_content_unique"]["groups"][0]["fields"] == ["pixel_sha256"]
    assert result["counts"]["eligible"] == 0


def test_duplicate_content_detection_is_global_across_namespaces(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    first = _row(root, "candidate-1", namespace="source-a")
    second = _row(root, "candidate-2", namespace="source-b", image_bytes=(root / "candidate-1.png").read_bytes())
    history = _row(root, "historical")

    result = _run(tmp_path, [first, second], [history])

    assert result["counts"]["eligible"] == 0
    assert result["candidate_content_unique"]["duplicate_rows"] == 2
    assert "image_sha256" in result["candidate_content_unique"]["groups"][0]["fields"]
    assert "pixel_sha256" in result["candidate_content_unique"]["groups"][0]["fields"]


def test_500_rows_reusing_one_image_cannot_pass_support_gate(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    first = _row(root, "candidate-0")
    image_bytes = (root / "candidate-0.png").read_bytes()
    candidates = [first]
    candidates.extend(_row(root, f"candidate-{i}", image_bytes=image_bytes,
                           sample_id=f"sample-{i}", group_id=f"group-{i}") for i in range(1, 500))
    history = _row(root, "historical")

    result = _run(tmp_path, candidates, [history])

    assert result["candidate_content_unique"]["unique"] is False
    assert result["candidate_content_unique"]["duplicate_rows"] == 500
    assert result["support"]["images"] == 0
    assert result["gates"]["sample_size_eligible"] is False
    assert result["gates"]["holdout_eligible"] is False


def test_same_group_distinct_content_is_not_a_duplicate():
    rows = [
        {"source_namespace": "provider/archive", "sample_id": "s1", "group_id": "ride-1",
         "image_sha256": "1" * 64, "pixel_sha256": "a" * 64, "objects": []},
        {"source_namespace": "provider/archive", "sample_id": "s2", "group_id": "ride-1",
         "image_sha256": "2" * 64, "pixel_sha256": "b" * 64, "objects": []},
    ]

    support = _sample_support(rows)

    assert support["images"] == 2
    assert support["groups"] == 1
    assert support["duplicate_rows"] == 0


def test_duplicate_exclusion_is_order_independent_by_identity(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    first = _row(root, "candidate-1", sample_id="one")
    second = _row(root, "candidate-2", sample_id="two", image_bytes=(root / "candidate-1.png").read_bytes())
    third = _row(root, "candidate-3", sample_id="three")
    history = _row(root, "historical")

    forward = _run(tmp_path, [first, second, third], [history], name="run-forward")
    reverse = _run(tmp_path, [third, second, first], [history], name="run-reverse")

    def identities(name):
        return {row["candidate"]["sample_id"] for row in map(json.loads, (tmp_path / name / "excluded.jsonl").read_text().splitlines())
                if "duplicate_content" in row["reasons"]}

    assert identities("run-forward") == identities("run-reverse") == {"one", "two"}
    assert forward["candidate_content_unique"]["groups"][0]["group_id"] == reverse["candidate_content_unique"]["groups"][0]["group_id"]


def test_source_and_annotation_fields_without_evidence_do_not_pass(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    candidate = _row(root, "candidate")
    history = _row(root, "historical")
    provenance = {
        "source_status": "source-verified", "provider": "Provider", "archive_id": "archive-v1",
        "source_url": "https://example.test/archive", "acquired_date": "2026-01-01",
        "acquisition_and_conversion": ["downloaded", "converted"], "source_archive_sha256": "a" * 64,
        "verified_by": "Reviewer", "verified_date": "2026-01-02",
        "annotation_review": {"status": "reviewed", "reviewer": "Reviewer",
                               "reviewed_date": "2026-01-03", "scope": "all candidate labels"},
    }

    result = _run(tmp_path, [candidate], [history], provenance=provenance)

    assert result["gates"]["source_verified"] is False
    assert result["gates"]["review_complete"] is False
    assert result["gates"]["holdout_eligible"] is False


def test_complete_fields_require_exact_history_inventory_and_evidence_files(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    candidate = _row(root, "candidate")
    history = _row(root, "historical")
    candidate_path = _write_jsonl(tmp_path / "candidate.jsonl", [candidate])
    history_path = _write_jsonl(tmp_path / "history.jsonl", [history])
    evidence = tmp_path / "source-note.txt"
    evidence.write_text("Human reviewed source archive record.", encoding="utf-8")
    evidence_sha = _sha(evidence.read_bytes())
    provenance = {
        "schema_version": 1, "source_status": "source-verified", "provider": "Provider",
        "archive_id": "archive-v1", "source_url": "https://example.test/archive",
        "acquired_date": "2026-01-01", "acquisition_and_conversion": ["downloaded", "converted"],
        "source_archive_sha256": "a" * 64, "verified_by": "Reviewer", "verified_date": "2026-01-02",
        "source_evidence": [{"path": evidence.name, "sha256": evidence_sha}],
        "annotation_review": {"status": "reviewed", "reviewer": "Reviewer",
                               "reviewed_date": "2026-01-03", "scope": "all candidate labels"},
        "annotation_evidence": [{"path": evidence.name, "sha256": evidence_sha}],
        "grouping_review": {"status": "reviewed", "reviewer": "Reviewer",
                             "reviewed_date": "2026-01-03", "scope": "rideID mapping"},
        "grouping_evidence": [{"path": evidence.name, "sha256": evidence_sha}],
        "history_coverage": {"status": "complete", "reviewer": "Reviewer", "reviewed_date": "2026-01-03",
                             "scope": ["training", "development", "prior-holdouts", "error-analysis",
                                       "manual-review", "performance-samples"],
                             "manifest_inventory": [{"name": history_path.name, "sha256": _sha(history_path.read_bytes())}]},
    }
    provenance_path = _provenance(tmp_path / "provenance.json", **provenance)

    result = audit(candidate_path, [history_path], provenance_path, root, tmp_path / "run")

    assert result["gates"]["source_verified"] is True
    assert result["gates"]["review_complete"] is True
    assert result["gates"]["grouping_verified"] is True
    assert result["gates"]["history_coverage"] is True
    assert result["gates"]["sample_size_eligible"] is False
    assert result["gates"]["holdout_eligible"] is False


def test_repeat_runs_have_deterministic_outputs(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    candidate = _row(root, "candidate")
    history = _row(root, "historical")
    candidate_path = _write_jsonl(tmp_path / "candidate.jsonl", [candidate])
    history_path = _write_jsonl(tmp_path / "history.jsonl", [history])
    provenance_path = _provenance(tmp_path / "provenance.json")

    audit(candidate_path, [history_path], provenance_path, root, tmp_path / "run-a")
    audit(candidate_path, [history_path], provenance_path, root, tmp_path / "run-b")

    for name in ("eligible.jsonl", "excluded.jsonl", "audit.json"):
        assert (tmp_path / "run-a" / name).read_bytes() == (tmp_path / "run-b" / name).read_bytes()


def test_refuse_existing_or_dataset_nested_output(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    candidate = _row(root, "candidate")
    history = _row(root, "historical")
    candidate_path = _write_jsonl(tmp_path / "candidate.jsonl", [candidate])
    history_path = _write_jsonl(tmp_path / "history.jsonl", [history])
    provenance_path = _provenance(tmp_path / "provenance.json")
    existing = tmp_path / "exists"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        audit(candidate_path, [history_path], provenance_path, root, existing)
    with pytest.raises(ValueError, match="outside dataset_root"):
        audit(candidate_path, [history_path], provenance_path, root, root / "out")

    with pytest.raises(ValueError, match="outside dataset_root"):
        audit(candidate_path, [history_path], provenance_path, root, tmp_path)


def test_candidate_changed_file_is_excluded(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    candidate = _row(root, "candidate")
    candidate["image_sha256"] = "f" * 64
    history = _row(root, "historical")

    result = _run(tmp_path, [candidate], [history])

    assert result["input_issues"]["candidate_file_checks"] == ["candidate[0]:image_sha256_mismatch"]
    assert result["counts"]["eligible"] == 0
    assert result["gates"]["holdout_eligible"] is False


def test_candidate_false_pixel_hash_is_recomputed_and_excluded(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    candidate = _row(root, "candidate", pixel_sha256="f" * 64)
    history = _row(root, "historical")

    result = _run(tmp_path, [candidate], [history])

    assert result["input_issues"]["candidate_file_checks"] == ["candidate[0]:pixel_sha256_mismatch"]
    assert result["counts"]["eligible"] == 0
    assert result["gates"]["holdout_eligible"] is False


def test_manifest_time_budget_is_enforced(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    candidate = _row(root, "candidate")
    history = _row(root, "historical")
    candidate_path = _write_jsonl(tmp_path / "candidate.jsonl", [candidate])
    history_path = _write_jsonl(tmp_path / "history.jsonl", [history])
    provenance_path = _provenance(tmp_path / "provenance.json")
    output = tmp_path / "timeout-output"

    with pytest.raises(TimeoutError, match="time budget"):
        audit(candidate_path, [history_path], provenance_path, root, output, max_seconds=1e-12)

    assert not output.exists()


def test_minimum_sample_support_requires_count_groups_and_categories():
    rows = []
    for index in range(500):
        group = f"ride:{index // 5}"
        objects = ([{"category": "person"}, {"category": "car"}, {"category": "traffic light"}]
                   if index < 125 else [])
        rows.append({"source_namespace": "provider/archive", "group_id": group, "objects": objects})

    support = _sample_support(rows)

    assert support["images"] == 500
    assert support["groups"] == 100
    assert support["eligible"] is True


def test_manifest_row_limit_stops_before_creating_output(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    candidates = [_row(root, "candidate-1"), _row(root, "candidate-2")]
    history = _row(root, "historical")
    candidate_path = _write_jsonl(tmp_path / "candidate.jsonl", candidates)
    history_path = _write_jsonl(tmp_path / "history.jsonl", [history])
    provenance_path = _provenance(tmp_path / "provenance.json")
    output = tmp_path / "bounded-output"

    with pytest.raises(ValueError, match="row limit exceeded"):
        audit(candidate_path, [history_path], provenance_path, root, output, max_rows_per_manifest=1)

    assert not output.exists()


def test_absolute_candidate_paths_are_excluded_and_redacted_from_output(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    candidate = _row(root, "candidate")
    candidate["image_path"] = str(root / "candidate.jpg")
    history = _row(root, "historical")

    result = _run(tmp_path, [candidate], [history])

    assert result["counts"]["eligible"] == 0
    excluded = json.loads((tmp_path / "run/excluded.jsonl").read_text().splitlines()[0])
    assert excluded["candidate"]["image_path"] is None
    assert "candidate_file:absolute_image_path" in excluded["reasons"]
