"""Train the weak-label XGBoost baseline using explicit input and output paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .schema import MODEL_FEATURES, SCHEMA_VERSION


def train(features_csv: Path, output: Path, seed: int = 42) -> None:
    frame = pd.read_csv(features_csv)
    if "risk_weak" not in frame:
        raise ValueError("Training CSV must contain risk_weak")
    x = frame.loc[:, MODEL_FEATURES].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(frame["risk_weak"], errors="raise").astype(int)
    classes = sorted(y.unique().tolist())
    if classes != [0, 1, 2]:
        raise ValueError(f"risk_weak must contain exactly classes 0, 1, 2; got {classes}")
    if y.value_counts().min() < 2:
        raise ValueError("Each risk_weak class needs at least two samples for a stratified split")
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    model = XGBClassifier(objective="multi:softprob", num_class=3, n_estimators=400, learning_rate=0.05, max_depth=7, tree_method="hist", eval_metric="mlogloss", random_state=seed)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    output.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(output / "risk_xgb.ubj")
    (output / "feature_order.json").write_text(json.dumps(list(MODEL_FEATURES), indent=2), encoding="utf-8")
    (output / "model_metadata.json").write_text(
        json.dumps(
            {
                "contract": "traffic-risk",
                "schema_version": SCHEMA_VERSION,
                "feature_order": list(MODEL_FEATURES),
                "train_cols_nosignal": list(MODEL_FEATURES),
                "label_type": "weak",
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.features_csv, args.output, args.seed)


if __name__ == "__main__":
    main()

