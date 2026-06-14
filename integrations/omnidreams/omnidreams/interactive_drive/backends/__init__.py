# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Renderer backends."""

from omnidreams.interactive_drive.backends.base import RenderBackend

__all__ = [
    "RenderBackend",
    "GrpcWorldModelRenderBackend",
    "RasterRenderBackend",
    "WorldModelRenderBackend",
]


def RasterRenderBackend(*args, **kwargs):  # type: ignore[no-untyped-def]
    from omnidreams.interactive_drive.backends.raster import (
        RasterRenderBackend as _cls,
    )

    return _cls(*args, **kwargs)


def WorldModelRenderBackend(*args, **kwargs):  # type: ignore[no-untyped-def]
    from omnidreams.interactive_drive.backends.world_model import (
        WorldModelRenderBackend as _cls,
    )

    return _cls(*args, **kwargs)


def GrpcWorldModelRenderBackend(*args, **kwargs):  # type: ignore[no-untyped-def]
    from omnidreams.interactive_drive.backends.grpc_world_model import (
        GrpcWorldModelRenderBackend as _cls,
    )

    return _cls(*args, **kwargs)

