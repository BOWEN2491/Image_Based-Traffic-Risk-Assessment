"""Feature extraction from the compact BDD-style detector output."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


VEHICLES = {"car", "bus", "truck", "train"}
RIDERS = {"rider", "motorcycle", "bicycle", "cyclist"}


def _box(obj: Mapping[str, Any], height: int, width: int) -> tuple[float, ...] | None:
    raw = obj.get("box2d")
    if not isinstance(raw, Mapping):
        return None
    try:
        x1, y1, x2, y2 = (float(raw[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return None
    values = np.asarray([x1, y1, x2, y2], dtype=float)
    if not np.isfinite(values).all():
        return None
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    return None if x2 <= x1 or y2 <= y1 else (float(x1), float(y1), float(x2), float(y2))


def _mean_pairwise_iou(boxes: list[tuple[float, ...]]) -> float:
    overlaps: list[float] = []
    for index, first in enumerate(boxes):
        ax1, ay1, ax2, ay2 = first
        for bx1, by1, bx2, by2 in boxes[index + 1 :]:
            intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
                0.0, min(ay2, by2) - max(ay1, by1)
            )
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
            overlaps.append(intersection / union if union else 0.0)
    return float(np.mean(overlaps)) if overlaps else 0.0


def compute_features(
    objects: Iterable[Mapping[str, Any]],
    height: int,
    width: int,
    *,
    center_width_pct: float = 0.33,
    near_thresh: float = 0.05,
) -> dict[str, Any]:
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    if not 0 < center_width_pct <= 1 or not 0 <= near_thresh <= 1:
        raise ValueError("Feature thresholds are outside their supported range")

    boxes: list[tuple[float, ...]] = []
    people: list[tuple[float, float, float]] = []
    vehicles: list[tuple[float, float, float]] = []
    lights: list[tuple[float, float, float, str]] = []
    n_rider = n_sign = 0
    image_area = float(height * width)

    for obj in objects:
        box = _box(obj, height, width)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1) / image_area
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        category = str(obj.get("category", "")).lower().replace("_", " ")
        boxes.append(box)
        if category == "person":
            people.append((area, cx, cy))
        elif category in VEHICLES:
            vehicles.append((area, cx, cy))
        elif category in RIDERS:
            n_rider += 1
        elif category == "traffic light":
            attributes = obj.get("attributes", {})
            color = str(attributes.get("trafficLightColor", "unknown")).lower()
            color = color if color in {"red", "yellow", "green"} else "unknown"
            lights.append((area, cx, cy, color))
        elif category == "traffic sign":
            n_sign += 1

    if not boxes:
        raise ValueError("Detector output contains no valid objects")

    counts = {color: sum(item[3] == color for item in lights) for color in ("red", "yellow", "green")}
    central_color, central_distance, central_area = "unknown", 1.0, 0.0
    candidates: list[tuple[float, float, str, float]] = []
    diagonal = max(1.0, float(np.hypot(width, height)))
    for area, cx, cy, color in lights:
        x_offset = abs(cx - width / 2) / width
        distance = float(np.hypot(cx - width / 2, cy - height / 2) / diagonal)
        if x_offset <= 0.18 and cy / height <= 0.75 and distance <= 0.35 and area >= 0.0006:
            score = 0.55 * x_offset + 0.25 * distance + 0.15 * (cy / height)
            candidates.append((score, distance, color, area))
    candidates.sort()
    if candidates and (len(candidates) == 1 or candidates[1][0] - candidates[0][0] >= 0.03):
        _, central_distance, central_color, central_area = candidates[0]

    person_areas = [item[0] for item in people]
    vehicle_areas = [item[0] for item in vehicles]
    half_center = width * center_width_pct / 2
    return {
        "n_person": len(people),
        "n_vehicle": len(vehicles),
        "n_rider": n_rider,
        "n_human": len(people) + n_rider,
        "n_tl": len(lights),
        "n_ts": n_sign,
        "n_tl_red": counts["red"],
        "n_tl_yellow": counts["yellow"],
        "n_tl_green": counts["green"],
        "central_tl_color": central_color,
        "central_tl_code": {"green": 1, "yellow": 2, "red": 3}.get(central_color, 0),
        "central_tl_is_red": int(central_color == "red"),
        "central_tl_is_yellow": int(central_color == "yellow"),
        "central_tl_is_green": int(central_color == "green"),
        "central_tl_center_dist": central_distance,
        "central_tl_area": central_area,
        "near_person_count": sum(area > near_thresh for area in person_areas),
        "near_vehicle_count": sum(area > near_thresh for area in vehicle_areas),
        "max_box_area_person": max(person_areas, default=0.0),
        "max_box_area_vehicle": max(vehicle_areas, default=0.0),
        "sum_box_area_person": float(sum(person_areas)),
        "sum_box_area_vehicle": float(sum(vehicle_areas)),
        "bottom_half_person_ratio": float(np.mean([cy >= height / 2 for _, _, cy in people])) if people else 0.0,
        "center_region_vehicle_ratio": float(np.mean([abs(cx - width / 2) < half_center for _, cx, _ in vehicles])) if vehicles else 0.0,
        "object_density_per_mp": len(boxes) / max(1.0, image_area / 1_000_000),
        "mean_overlap_iou": _mean_pairwise_iou(boxes),
    }


def build_features_from_bdd(bdd_json: Mapping[str, Any], image_path: str | Path) -> dict[str, Any]:
    frames = bdd_json.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Detector output contains no frames")
    objects = frames[0].get("objects")
    if not isinstance(objects, list):
        raise ValueError("Detector output contains no object list")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Image could not be decoded")
    height, width = image.shape[:2]
    return compute_features(objects, height, width)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features from one BDD-style JSON file")
    parser.add_argument("--bdd-json", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()
    features = build_features_from_bdd(json.loads(args.bdd_json.read_text(encoding="utf-8")), args.image)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([features]).to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()

