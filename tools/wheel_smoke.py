"""Verify a built wheel from an isolated working directory and interpreter."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def smoke(wheel: Path, lock: Path) -> None:
    wheel = wheel.resolve()
    lock = lock.resolve()
    with tempfile.TemporaryDirectory(prefix="risk-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        working = root / "outside-source"
        working.mkdir()
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", "-r", str(lock)],
            check=True,
            cwd=working,
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
            check=True,
            cwd=working,
        )
        verify_installed(python, working)


def verify_installed(python: Path, working: Path) -> None:
    """Verify the current wheel installation without importing a source checkout."""
    scripts = python.parent
    subprocess.run([str(python), "-m", "pip", "check"], check=True, cwd=working)
    check = (
            "import json; from importlib.metadata import distributions; from importlib.resources import files; "
            "from pathlib import Path; "
            "from src.config import Settings; "
            "s=Settings(); m=json.loads(files('tools').joinpath('model_manifest.json').read_text()); "
            "assert s.model_dir.resolve() == (Path.cwd() / 'models').resolve(); "
            "assert 'traffic-risk-assessment' in str(s.upload_dir); "
            "assert m['feature_schema'] == '1.0.0'; "
            "opencv=sorted(d.metadata['Name'].lower() for d in distributions() if d.metadata['Name'].lower().startswith('opencv-')); "
            "assert opencv == ['opencv-python'], opencv; "
            "print(json.dumps({'model_dir': str(s.model_dir), 'upload_dir': str(s.upload_dir)}))"
    )
    subprocess.run([str(python), "-c", check], check=True, cwd=working)
    for command in ("risk-assess", "download-risk-models"):
        executable = scripts / (f"{command}.exe" if os.name == "nt" else command)
        subprocess.run([str(executable), "--help"], check=True, cwd=working)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, nargs="?")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    if args.installed:
        verify_installed(Path(sys.executable), Path.cwd())
    elif args.wheel is not None and args.lock is not None:
        smoke(args.wheel, args.lock)
    else:
        parser.error("wheel and --lock are required unless --installed is used")


if __name__ == "__main__":
    main()
