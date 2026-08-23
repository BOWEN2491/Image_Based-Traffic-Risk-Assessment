#!/usr/bin/env python
"""
Merge reviewed ROIs back into the main ROI_labeled folder.

- REVIEW root: the folder you opened for review (e.g., ROI_labeled_unknown_only).
  It must contain subfolders: red/, yellow/, green/, unknown/ (after relabeling).
- TARGET root: the original, full ROI_labeled folder (to be updated).

For each file in REVIEW:
  1) Find a file with the same filename in TARGET (search all class subfolders).
  2) If found and its class != new class -> move to the new class in TARGET.
  3) If not found:
       - If --copy-missing: copy it into the new class in TARGET.
       - Else: warn and skip.

A CSV log (merge_log.csv) will be written under TARGET root.

Usage:
  python merge_review_back.py ^
    --review-root "data/reviewed" ^
    --target-root "data/roi_labeled" ^
    --copy-missing
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import csv

CLASSES = ["red","yellow","green","unknown"]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def index_target(target_root: Path) -> dict[str, tuple[Path, str]]:
    """
    Build an index: filename -> (full_path, class_name) for all files under TARGET class folders.
    """
    idx: dict[str, tuple[Path,str]] = {}
    for c in CLASSES:
        d = target_root / c
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file():
                idx[p.name] = (p, c)
    return idx

def iter_review_items(review_root: Path):
    """
    Yield (path, class_name) for all files under REVIEW class folders.
    """
    for c in CLASSES:
        d = review_root / c
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file():
                yield p, c

def move_to(dst_dir: Path, src: Path) -> Path:
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.move(str(src), str(dst))
    return dst

def copy_to(dst_dir: Path, src: Path) -> Path:
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.copy2(str(src), str(dst))
    return dst

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge reviewed unknown-only folder back into main ROI_labeled.")
    ap.add_argument("--review-root", type=str, required=True, help="Reviewed folder (e.g., ROI_labeled_unknown_only).")
    ap.add_argument("--target-root", type=str, required=True, help="Main ROI_labeled folder to update.")
    ap.add_argument("--dry-run", action="store_true", help="Preview actions without moving/copying files.")
    ap.add_argument("--copy-missing", action="store_true", help="Copy reviewed files not found in target into the new class.")
    return ap.parse_args()

def main():
    args = parse_args()
    review_root = Path(args.review_root)
    target_root = Path(args.target_root)

    if not review_root.exists():
        raise FileNotFoundError(f"Review root not found: {review_root}")
    if not target_root.exists():
        raise FileNotFoundError(f"Target root not found: {target_root}")

    # index target
    idx = index_target(target_root)

    log_path = target_root / "merge_log.csv"
    n_found = n_moved = n_copied = n_missing = 0

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename","old_class_in_target","new_class_from_review","action","target_path"])

        for p, new_cls in iter_review_items(review_root):
            filename = p.name
            rec = idx.get(filename, None)
            if rec is not None:
                # exists in target
                tpath, old_cls = rec
                n_found += 1
                if old_cls != new_cls:
                    if args.dry_run:
                        writer.writerow([filename, old_cls, new_cls, "DRY-MOVE", str(target_root/new_cls/filename)])
                    else:
                        # move inside target to new class
                        new_path = move_to(target_root / new_cls, tpath)
                        writer.writerow([filename, old_cls, new_cls, "MOVE", str(new_path)])
                        n_moved += 1
                else:
                    # same class, nothing to do
                    writer.writerow([filename, old_cls, new_cls, "KEEP", str(tpath)])
            else:
                # not in target
                n_missing += 1
                if args.copy_missing:
                    if args.dry_run:
                        writer.writerow([filename, "", new_cls, "DRY-COPY", str(target_root/new_cls/filename)])
                    else:
                        new_path = copy_to(target_root / new_cls, p)
                        writer.writerow([filename, "", new_cls, "COPY", str(new_path)])
                        n_copied += 1
                else:
                    writer.writerow([filename, "", new_cls, "MISSING_SKIP", ""])

    print(f"[DONE] merge_log.csv -> {log_path}")
    print(f"Found in target: {n_found}, moved: {n_moved}, copied (missing): {n_copied}, missing(skipped): {n_missing}")

if __name__ == "__main__":
    main()
