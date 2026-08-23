#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROI Labeled Review Tool (keyboard-driven with OpenCV).

Usage:
  python review_roi_labeler.py ^
    --root "data/roi_labeled" ^
    --focus unknown

Requires:
  pip install opencv-python pillow numpy tqdm

Folders expected under --root:
  red/  yellow/  green/  unknown/   (created by the extractor)
Optional:
  _trash/ (for discarded ROIs)

Hotkeys:
  r / y / g / u  -> relabel & move file to that folder
  RIGHT / d / SPACE -> next
  LEFT / a / BACKSPACE -> prev
  x -> move file to _trash
  s -> save manifest
  f -> toggle focus: unknown-only vs all
  q -> quit (auto-save)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import csv
import os
from typing import List, Tuple

CLASSES = ["red","yellow","green","unknown"]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_items(root: Path, focus: str | None) -> List[Tuple[Path,str]]:
    items: List[Tuple[Path,str]] = []
    for c in CLASSES:
        cls_dir = root / c
        if not cls_dir.exists():
            continue
        for ext in ("*.jpg","*.png","*.jpeg","*.bmp","*.tif","*.tiff"):
            for p in cls_dir.rglob(ext):
                if focus is None or c == focus:
                    items.append((p, c))
    items.sort(key=lambda x: str(x[0]))
    return items

def draw_info(img: np.ndarray, text_lines: List[str]) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    y0 = 28
    for i, line in enumerate(text_lines):
        y = y0 + i*22
        cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return overlay

def imread_resize(p: Path, max_side: int = 520) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((320,320,3), dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(max_side / max(h,w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img

def move_to(dst_dir: Path, p: Path) -> Path:
    ensure_dir(dst_dir)
    new_path = dst_dir / p.name
    if new_path.exists():
        raise FileExistsError(f"Destination already exists: {new_path}")
    os.replace(str(p), str(new_path))
    return new_path

def save_manifest(root: Path, items_all: List[Tuple[Path,str]], manifest: Path) -> None:
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path","label"])
        for p, lab in items_all:
            # only write that still exists
            if p.exists():
                writer.writerow([str(p), lab])
    os.replace(str(temporary), str(manifest))

def main():
    ap = argparse.ArgumentParser(description="Review & relabel ROI_labeled images (OpenCV).")
    ap.add_argument("--root", type=str, required=True, help="ROI_labeled root folder")
    ap.add_argument("--focus", type=str, default=None, choices=[None, "red","yellow","green","unknown"], nargs='?',
                    help="Only load this class at start (e.g., unknown). Press 'f' to toggle focus at runtime.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    focus = args.focus if args.focus in CLASSES else None
    items = load_items(root, focus)  # list of (path, label)
    if not items:
        print("[INFO] No images found. Check your ROI_labeled structure.")
        return

    trash_dir = root / "_trash"
    manifest = root / "review_manifest.csv"

    idx = 0
    focus_unknown_only = (focus == "unknown")

    cv2.namedWindow("ROI Labeler", cv2.WINDOW_AUTOSIZE)
    try:
      while True:
        # guard index
        idx = max(0, min(idx, len(items)-1))
        p, lab = items[idx]

        img = imread_resize(p)
        info = [
            f"[{idx+1}/{len(items)}] {p.name} | label={lab}",
            "Hotkeys: r/y/g/u -> relabel | Left/Right or a/d or Backspace/Space -> nav",
            "x delete to _trash | s save manifest | f toggle focus | q quit",
            f"Focus: {'unknown-only' if focus_unknown_only else 'all'}"
        ]
        disp = draw_info(img, info)
        cv2.imshow("ROI Labeler", disp)

        key = cv2.waitKey(0) & 0xFF
        # Navigation
        if key in (ord('d'), 255, 83, 54, 32):  # right/space; some codes vary by platform
            idx += 1
            continue
        if key in (ord('a'), 81, 52, 8):  # left/backspace
            idx -= 1
            continue

        # Relabel
        if key in (ord('r'), ord('y'), ord('g'), ord('u')):
            new_lab = {'r':'red','y':'yellow','g':'green','u':'unknown'}[chr(key)]
            if new_lab != lab:
                new_path = move_to(root / new_lab, p)
                items[idx] = (new_path, new_lab)
                lab = new_lab
            # auto next
            idx += 1
            continue

        # Delete to trash
        if key == ord('x'):
            new_path = move_to(trash_dir, p)
            items.pop(idx)
            if idx >= len(items):
                idx = len(items)-1
            continue

        # Save
        if key == ord('s'):
            save_manifest(root, items, manifest)
            print(f"[SAVED] {manifest}")
            continue

        # Toggle focus
        if key == ord('f'):
            focus_unknown_only = not focus_unknown_only
            items = load_items(root, "unknown" if focus_unknown_only else None)
            if not items:
                print("[INFO] No items for current focus.")
                continue
            idx = 0
            continue

        # Quit
        if key == ord('q'):
            break
    finally:
      save_manifest(root, items, manifest)
      cv2.destroyAllWindows()
    print(f"[DONE] Manifest saved to: {manifest}")

if __name__ == "__main__":
    main()

