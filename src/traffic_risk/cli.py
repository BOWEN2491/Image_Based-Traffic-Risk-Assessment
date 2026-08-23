"""Single public command line entry point for traffic-risk."""
from __future__ import annotations

import argparse
import json
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
        sub.add_parser(name, help=f"{name.replace('-', ' ')} (local workflow)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "predict":
        raise SystemExit(f"{args.command} is an offline workflow and must be invoked with its dedicated module")
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

