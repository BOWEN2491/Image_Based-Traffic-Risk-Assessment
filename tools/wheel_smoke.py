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
        env = isolated_subprocess_environment()
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", "-r", str(lock)],
            check=True,
            cwd=working,
            env=env,
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
            check=True,
            cwd=working,
            env=env,
        )
        verify_installed(python, working, env=env)


def isolated_subprocess_environment() -> dict[str, str]:
    """Remove source/config overrides while keeping the normal process environment."""
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() != "PYTHONPATH" and not key.upper().startswith("RISK_")
    }


def verify_installed(python: Path, working: Path, *, env: dict[str, str] | None = None) -> None:
    """Verify the current wheel installation without importing a source checkout."""
    env = env if env is not None else isolated_subprocess_environment()
    scripts = python.parent
    subprocess.run([str(python), "-m", "pip", "check"], check=True, cwd=working, env=env)
    check = (
            "import json; from importlib.metadata import distributions; from importlib.resources import files; "
            "from pathlib import Path; "
            "from traffic_risk.config import Settings; "
            "s=Settings(); m=json.loads(files('traffic_risk').joinpath('model_manifest.json').read_text()); "
            "assert s.model_dir.resolve() == (Path.cwd() / 'models').resolve(); "
            "assert 'traffic-risk-assessment' in str(s.upload_dir); "
            "assert m['feature_schema'] == '2.0.0'; "
            "opencv=sorted(d.metadata['Name'].lower() for d in distributions() if d.metadata['Name'].lower().startswith('opencv-')); "
            "assert opencv == ['opencv-python'], opencv; "
            "print(json.dumps({'model_dir': str(s.model_dir), 'upload_dir': str(s.upload_dir)}))"
    )
    subprocess.run([str(python), "-c", check], check=True, cwd=working, env=env)
    for command in ("traffic-risk",):
        executable = scripts / (f"{command}.exe" if os.name == "nt" else command)
        subprocess.run([str(executable), "--help"], check=True, cwd=working, env=env)


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

