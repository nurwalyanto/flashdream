# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for GrpcWorldModelRenderBackend internals.

All tests run on CPU — no GPU, no Vulkan, no cloud server.
Helper functions and standalone logic are tested directly without
importing the full backend module (which pulls in torch via Ludus).
"""

from __future__ import annotations

import io
import sys
import types
import unittest
import zipfile
from unittest.mock import MagicMock

import numpy as np
from omnidreams.grpc.protos import common_pb2, video_model_pb2
from PIL import Image


def _import_grpc_module() -> types.ModuleType:
    """Import grpc_world_model module with GPU deps stubbed out."""
    import importlib.util

    # Stub torch and GPU-heavy packages before anything touches them
    torch_stub = types.ModuleType("torch")
    torch_stub.device = lambda x: x  # type: ignore[attr-defined]
    sys.modules["torch"] = torch_stub

    raster_stub = types.ModuleType("rasterizer")
    raster_stub.LudusConditionRasterizer = MagicMock
    sys.modules["omnidreams.interactive_drive.rasterizer"] = raster_stub

    # Stub backends.world_model so __init__ doesn't chase torch
    wm_stub = types.ModuleType("world_model")
    wm_stub.WorldModelRenderBackend = MagicMock
    sys.modules["omnidreams.interactive_drive.backends.world_model"] = wm_stub

    # grpc_world_model imports base.RenderBackend via backends.__init__.
    # We need a real RenderBackend, so we must let base import through.
    # Patch out world_model before __init__ triggers.

    path = (
        "/tmp/flashdreams/integrations/omnidreams/omnidreams/"
        "interactive_drive/backends/grpc_world_model.py"
    )
    spec = importlib.util.spec_from_file_location("grpc_world_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_grpc = _import_grpc_module()


def _make_fake_jpeg_bytes(
    width: int = 128, height: int = 72, seed: int = 0
) -> bytes:
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 255, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


class TestMatrixToPoseProto(unittest.TestCase):
    def test_identity(self) -> None:
        pose = _grpc._matrix_to_pose_proto(np.eye(4, dtype=np.float32))
        self.assertAlmostEqual(pose.quat.w, 1.0, places=6)
        self.assertAlmostEqual(pose.vec.x, 0.0, places=6)

    def test_translation(self) -> None:
        mat = np.eye(4, dtype=np.float32)
        mat[:3, 3] = [1.0, 2.0, 3.0]
        pose = _grpc._matrix_to_pose_proto(mat)
        self.assertAlmostEqual(pose.vec.x, 1.0)
        self.assertAlmostEqual(pose.vec.y, 2.0)
        self.assertAlmostEqual(pose.vec.z, 3.0)

    def test_rotation_90z(self) -> None:
        mat = np.array(
            [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        pose = _grpc._matrix_to_pose_proto(mat)
        self.assertAlmostEqual(pose.quat.w, 0.7071, places=3)
        self.assertAlmostEqual(pose.quat.z, 0.7071, places=3)


class TestDecodeCameraOutput(unittest.TestCase):
    def test_single_frame(self) -> None:
        cam_out = video_model_pb2.CameraOutput(
            camera_logical_id="cam",
            rgb_frames=[
                video_model_pb2.Image(
                    data=_make_fake_jpeg_bytes(128, 72, seed=0),
                    format=video_model_pb2.JPEG,
                )
            ],
        )
        resp = video_model_pb2.VideoChunkReturn(camera_outputs=[cam_out])
        frames = _grpc._decode_camera_output(resp)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].shape, (72, 128, 3))

    def test_multi_frame(self) -> None:
        cam_out = video_model_pb2.CameraOutput(
            camera_logical_id="cam",
            rgb_frames=[
                video_model_pb2.Image(
                    data=_make_fake_jpeg_bytes(128, 72, seed=i),
                    format=video_model_pb2.JPEG,
                )
                for i in range(5)
            ],
        )
        resp = video_model_pb2.VideoChunkReturn(camera_outputs=[cam_out])
        frames = _grpc._decode_camera_output(resp)
        self.assertEqual(len(frames), 5)

    def test_multi_camera(self) -> None:
        resp = video_model_pb2.VideoChunkReturn(
            camera_outputs=[
                video_model_pb2.CameraOutput(
                    camera_logical_id=f"cam{i}",
                    rgb_frames=[
                        video_model_pb2.Image(
                            data=_make_fake_jpeg_bytes(128, 72),
                            format=video_model_pb2.JPEG,
                        )
                    ],
                )
                for i in range(2)
            ]
        )
        frames = _grpc._decode_camera_output(resp)
        self.assertEqual(len(frames), 2)


class TestParquetExtraction(unittest.TestCase):
    def test_extracts_only_parquets(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".usdz") as f:
            path = f.name
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("clipgt/lane_line.parquet", b"lane")
                zf.writestr("clipgt/calibration_estimate.parquet", b"calib")
                zf.writestr("clipgt/frames/frame_0.jpeg", b"image")
                zf.writestr("other.txt", b"other")
            result = _grpc._extract_hdmap_parquets(path)

        with zipfile.ZipFile(io.BytesIO(result)) as zf:
            names = zf.namelist()
        self.assertIn("clipgt/lane_line.parquet", names)
        self.assertIn("clipgt/calibration_estimate.parquet", names)
        self.assertNotIn("clipgt/frames/frame_0.jpeg", names)
        self.assertNotIn("other.txt", names)
        self.assertEqual(len(names), 2)


if __name__ == "__main__":
    unittest.main()
