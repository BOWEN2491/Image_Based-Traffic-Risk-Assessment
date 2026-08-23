"""Deterministic weak-label generation for the installable package."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

WEAK_LABEL_VERSION = "weak-label-v2"
WEAK_LABEL_SEED = 42
WEAK_LABEL_N_CLUSTERS = 3
WEAK_LABEL_N_INIT = 50
WEAK_LABEL_FEATURES = (
    "n_person", "n_vehicle", "n_rider", "n_ts", "near_person_count",
    "near_vehicle_count", "max_box_area_person", "max_box_area_vehicle",
    "sum_box_area_person", "sum_box_area_vehicle", "bottom_half_person_ratio",
    "center_region_vehicle_ratio", "object_density_per_mp", "mean_overlap_iou",
)
WEAK_LABEL_DANGER_WEIGHTS = {
    "n_person": 2.0, "n_rider": 1.5, "n_ts": 1.0,
    "max_box_area_person": 2.0, "sum_box_area_person": 2.0,
    "sum_box_area_vehicle": 1.5, "bottom_half_person_ratio": 2.0,
    "object_density_per_mp": 1.5, "mean_overlap_iou": 1.0,
}


class WeakLabelError(ValueError):
    """Input data cannot satisfy the deterministic weak-label contract."""


def generate_weak_label(df1: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df1, pd.DataFrame) or df1.empty:
        raise WeakLabelError("weak-label input must be a non-empty pandas DataFrame")
    missing = [column for column in WEAK_LABEL_FEATURES if column not in df1.columns]
    if missing:
        raise WeakLabelError(f"Missing required columns (missing): {', '.join(missing)}")
    df = df1.copy()
    values = df.loc[:, list(WEAK_LABEL_FEATURES)].copy()
    for column in WEAK_LABEL_FEATURES:
        converted = pd.to_numeric(values[column], errors="coerce")
        invalid = converted.isna() | ~np.isfinite(converted.to_numpy(dtype=float))
        if invalid.any():
            rows = converted.index[invalid].tolist()
            if "sample_id" in df.columns:
                rows = df.loc[rows, "sample_id"].tolist()
            raise WeakLabelError(f"weak-label column {column!r} contains non-numeric or non-finite values (rows: {rows[:5]})")
        values[column] = converted.astype(float)
    if len(values) < WEAK_LABEL_N_CLUSTERS:
        raise WeakLabelError(f"at least {WEAK_LABEL_N_CLUSTERS} valid samples are required")
    if len(values.drop_duplicates()) < WEAK_LABEL_N_CLUSTERS:
        raise WeakLabelError(f"unable to form {WEAK_LABEL_N_CLUSTERS} distinct clusters")
    scaled = StandardScaler().fit_transform(values)
    try:
        labels = KMeans(n_clusters=WEAK_LABEL_N_CLUSTERS, random_state=WEAK_LABEL_SEED, n_init=WEAK_LABEL_N_INIT).fit_predict(scaled)
    except (ValueError, RuntimeError) as exc:
        raise WeakLabelError(f"unable to form {WEAK_LABEL_N_CLUSTERS} clusters") from exc
    columns = list(WEAK_LABEL_DANGER_WEIGHTS)
    weights = np.array([WEAK_LABEL_DANGER_WEIGHTS[column] for column in columns], dtype=float)
    means = values.groupby(labels)[columns].mean()
    scores = (means * weights).sum(axis=1)
    if len(scores) != WEAK_LABEL_N_CLUSTERS or not np.isfinite(scores.to_numpy()).all():
        raise WeakLabelError(f"unable to form {WEAK_LABEL_N_CLUSTERS} risk clusters")
    order = scores.sort_values().index.tolist()
    risk_map = {cluster: index for index, cluster in enumerate(order)}
    df["cluster"] = labels
    df["risk_weak"] = pd.Series(labels, index=values.index).map(risk_map)
    return df

