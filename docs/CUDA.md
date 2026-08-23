# Optional NVIDIA CUDA setup

CPU inference is the supported default. `requirements-cpu.lock` explicitly uses `https://download.pytorch.org/whl/cpu`, pins `torch`/`torchvision` CPU wheels, and selects the official `xgboost-cpu==3.2.0` distribution for the shared Windows/Linux environment; it must not be used for macOS, other platforms, or to construct a CUDA environment. Project metadata falls back to the official full `xgboost==3.2.0` distribution outside Windows/Linux, with the same `import xgboost` API. CUDA compatibility depends on the operating system, NVIDIA driver, GPU architecture, and the versions supported by PyTorch and Ultralytics at installation time.

1. Confirm the driver with `nvidia-smi`.
2. Use the [official PyTorch installer](https://pytorch.org/get-started/locally/) to select a wheel compatible with the machine.
3. Install that PyTorch build in a fresh Python 3.11 environment.
4. Install this project's dependencies from `pyproject.toml` after the selected PyTorch pair is present, then install the project with `python -m pip install --no-deps -e .` and run `python -m pip check`. Do not install `requirements-cpu.lock` in this environment.
5. Run the manual real-model smoke test before relying on acceleration.

Example commands in third-party documentation change over time; copy the current command from PyTorch rather than hard-coding a `+cuXXX` requirement in this project's default lock or metadata.

CUDA does not change the project's research-only status. GPU use should default to a single in-flight inference unless memory and concurrency have been measured.
