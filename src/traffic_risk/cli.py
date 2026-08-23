"""Single public command line entry point for traffic-risk."""
from __future__ import annotations

import argparse
import json
import pandas as pd
import sys
import hashlib
from pathlib import Path

from .config import Settings
from .model_runtime import ModelRuntime, ModelUnavailableError
from .pipeline import run_single_image


COMMANDS = ("predict", "download-models", "build-features", "generate-weak-labels", "train-risk", "train-traffic-light", "build-local-manifest", "review-roi", "merge-review")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="traffic-risk", description="Traffic scene risk assessment tools")
    sub = parser.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict", help="predict risk for one image")
    predict.add_argument("image", type=Path)
    for name in COMMANDS[1:]:
        workflow = sub.add_parser(name, help=f"{name.replace('-', ' ')} local workflow")
        workflow.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(argv if argv is not None else sys.argv[1:])
    command = raw[0] if raw else None
    if command and command != "predict":
        args = argparse.Namespace(command=command, args=raw[1:])
    else:
        args = build_parser().parse_args(argv)
    if args.command != "predict":
        forwarded = list(getattr(args, "args", []))
        if args.command == "generate-weak-labels":
            parser = argparse.ArgumentParser(prog="traffic-risk generate-weak-labels")
            parser.add_argument("--input-csv", type=Path, required=True)
            parser.add_argument("--output-csv", type=Path, required=True)
            options = parser.parse_args(forwarded)
            from .weak_label import generate_weak_label
            result = generate_weak_label(pd.read_csv(options.input_csv))
            options.output_csv.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(options.output_csv, index=False)
            return 0
        modules = {
            "build-features": "src.build_features",
            "train-risk": "src.build_XGBoost",
            "train-traffic-light": "src.train_tl_cnn",
            "review-roi": "src.review_roi_labeler",
            "merge-review": "src.merge_review_back",
        }
        if args.command == "build-local-manifest":
            parser = argparse.ArgumentParser(prog="traffic-risk build-local-manifest")
            parser.add_argument("--model-dir", type=Path, required=True)
            parser.add_argument("--output", type=Path, required=True)
            options = parser.parse_args(forwarded)
            destinations = (
                "yolo/yolo11n.pt", "cnn/best_model.pth", "cnn/class_indices.json",
                "risk/risk_xgb.ubj", "risk/feature_order.json", "risk/model_metadata.json",
            )
            assets = []
            for relative in destinations:
                path = options.model_dir / relative
                if not path.is_file():
                    raise FileNotFoundError(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                assets.append({"destination": relative, "size": path.stat().st_size, "sha256": digest})
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_text(json.dumps({"contract": "traffic-risk", "schema_version": "2.0.0", "distribution_status": "local", "model_versions": {"mode": "models", "feature_schema": "2.0.0"}, "assets": assets}, indent=2), encoding="utf-8")
            return 0
        if args.command == "download-models":
            from tools.download_models import main as workflow_main
        else:
            module_name = modules[args.command]
            module = __import__(module_name, fromlist=["main"])
            workflow_main = module.main
        old_argv = sys.argv
        try:
            sys.argv = [f"traffic-risk {args.command}", *forwarded]
            workflow_main()
        finally:
            sys.argv = old_argv
        return 0
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    runtime = ModelRuntime(Settings())
    try:
        runtime.load()
    except ModelUnavailableError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(run_single_image(args.image, runtime), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    main()

