"""Stable dataset identities and collision checks for data tooling.

The functions in this module deliberately fail before any copy/move operation.
They are usable by feature generation, ROI extraction, and review merge tools.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


class IdentityConflict(ValueError):
    """Raised when sample identity or destination safety checks fail."""


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    relative_path: str
    sha256: str
    group_id: str | None = None
    label: str | None = None
    source_image_id: str | None = None
    object_id: str | None = None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id_for(root: Path, path: Path) -> str:
    """Return a POSIX relative path, preserving its extension."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise IdentityConflict(f"Path is outside data root: {path}") from exc
    return relative.as_posix()


def build_records(root: Path, paths: Iterable[Path], labels: dict[str, str] | None = None) -> list[SampleRecord]:
    labels = labels or {}
    records: list[SampleRecord] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        sid = sample_id_for(root, path)
        records.append(SampleRecord(sid, sid, sha256_file(path), label=labels.get(sid)))
    validate_records(records)
    return records


def validate_records(records: Iterable[SampleRecord]) -> None:
    records = list(records)
    seen_ids: set[str] = set()
    by_hash: dict[str, set[str | None]] = {}
    for record in records:
        if record.sample_id in seen_ids:
            raise IdentityConflict(f"Duplicate sample_id: {record.sample_id}")
        seen_ids.add(record.sample_id)
        by_hash.setdefault(record.sha256, set()).add(record.label)
    conflicts = [digest for digest, labels in by_hash.items() if len(labels) > 1]
    if conflicts:
        raise IdentityConflict(f"Same content has conflicting labels: {conflicts[0]}")

    # Same directory + stem with multiple extensions is ambiguous for legacy tools.
    stems: dict[tuple[str, str], set[str]] = {}
    for record in records:
        path = Path(record.relative_path)
        stems.setdefault((path.parent.as_posix(), path.stem.lower()), set()).add(path.suffix.lower())
    duplicate_stems = [key for key, extensions in stems.items() if len(extensions) > 1]
    if duplicate_stems:
        raise IdentityConflict(f"Same directory/stem has multiple extensions: {duplicate_stems[0]}")


def write_jsonl(records: Iterable[SampleRecord], destination: Path) -> None:
    records = list(records)
    validate_records(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)


def assert_destinations_free(destinations: Iterable[Path]) -> None:
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise IdentityConflict(f"Destination already exists: {existing[0]}")

