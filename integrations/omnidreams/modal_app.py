# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Modal serverless entry point for interactive-drive with --stream-mjpeg.

Build image only:
    cd /tmp/flashdream
    modal shell integrations/omnidreams/modal_app.py --cmd true

Shell into built image:
    modal shell integrations/omnidreams/modal_app.py

Run:
    modal run integrations/omnidreams/modal_app.py
"""

from __future__ import annotations

import subprocess

import modal

app = modal.App("omnidreams-interactive-drive")

_INF = 3600  # 1-hour container timeout

OMNIDREAMS_VOL = modal.Volume.from_name("omnidreams", create_if_missing=True)

_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("build-essential", "git", "curl")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
    )
    .add_local_dir(
        ".",
        "/workspace",
        copy=True,
        ignore=[
            ".venv", "__pycache__", ".git", ".mypy_cache",
            ".pytest_cache", ".ruff_cache", "*.pyc",
        ],
    )
    .env({"UV_LINK_MODE": "copy"})
    .workdir("/workspace")
    .run_commands(
        "uv sync --package flashdreams-omnidreams "
        "--extra interactive-drive --frozen",
        "uv cache clean",
    )
)


@app.function(
    image=_IMAGE,
    gpu="RTX-PRO-6000",
    timeout=_INF,
    scaledown_window=120,
    volumes={
        "/mnt/omnidreams": OMNIDREAMS_VOL,
    },
)
@modal.web_server(port=8000, startup_timeout=300)
def interactive_drive() -> None:
    """Run interactive-drive MJPEG stream proxied to a Modal web endpoint."""
    import os

    os.chdir("/workspace")

    for _d in (
        "/mnt/omnidreams/flashdreams-cache",
        "/mnt/omnidreams/hf-cache",
        "/mnt/omnidreams/triton-cache",
    ):
        os.makedirs(_d, exist_ok=True)

    _env_base = {
        "FLASHDREAMS_CACHE_DIR": "/mnt/omnidreams/flashdreams-cache",
        "HF_HOME": "/mnt/omnidreams/hf-cache",
        "TRITON_CACHE_DIR": "/mnt/omnidreams/triton-cache",
    }

    scene_dir = os.environ.get("SCENE_DIR", "/mnt/omnidreams/flashdream")
    manifest = os.environ.get("MANIFEST", "example_world_model.yaml")

    proc = subprocess.Popen(
        [
            "uv", "run",
            "--package", "flashdreams-omnidreams",
            "interactive-drive",
            f"--scene-dir={scene_dir}",
            "--stream-mjpeg=:8000",
            "--backend=omnidreams",
            f"--manifest={manifest}",
        ],
        env={**os.environ, **_env_base},
    )
    proc.wait()


@app.function(
    image=_IMAGE,
    gpu="RTX-PRO-6000",
    timeout=_INF,
    scaledown_window=120,
    volumes={
        "/mnt/omnidreams": OMNIDREAMS_VOL,
    },
)
def cli(
    scene_dir: str = "/mnt/omnidreams/flashdream",
    manifest: str = "example_world_model.yaml",
    backend: str = "omnidreams",
) -> None:
    """Non-web variant — runs process and streams logs to terminal."""
    import os

    os.chdir("/workspace")

    for _d in (
        "/mnt/omnidreams/flashdreams-cache",
        "/mnt/omnidreams/hf-cache",
        "/mnt/omnidreams/triton-cache",
    ):
        os.makedirs(_d, exist_ok=True)

    subprocess.run(
        [
            "uv", "run",
            "--package", "flashdreams-omnidreams",
            "interactive-drive",
            f"--scene-dir={scene_dir}",
            "--stream-mjpeg=:8000",
            f"--backend={backend}",
            f"--manifest={manifest}",
        ],
        check=True,
        env={
            **os.environ,
            "FLASHDREAMS_CACHE_DIR": "/mnt/omnidreams/flashdreams-cache",
            "HF_HOME": "/mnt/omnidreams/hf-cache",
            "TRITON_CACHE_DIR": "/mnt/omnidreams/triton-cache",
        },
    )


@app.local_entrypoint()
def main(
    scene_dir: str = "/mnt/omnidreams/flashdream",
    manifest: str = "example_world_model.yaml",
) -> None:
    cli.remote(scene_dir=scene_dir, manifest=manifest)
