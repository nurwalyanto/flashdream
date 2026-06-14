#!/usr/bin/env python3
"""CPU-only fake gRPC server for testing GrpcWorldModelRenderBackend.

Usage:
    uv run --package flashdreams-omnidreams python scripts/fake_grpc_server.py

Then in another terminal:
    uv run --package flashdreams-omnidreams interactive-drive \
      --grpc-endpoint localhost:50052 --manifest example_world_model_perf.yaml
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
from omnidreams.grpc.protos import (
    common_pb2,
    video_model_pb2,
    video_model_pb2_grpc,
)
from PIL import Image

_FRAME_SIZE = (1280, 704)


def _make_fake_frame(seed: int = 0) -> bytes:
    arr = np.zeros((_FRAME_SIZE[1], _FRAME_SIZE[0], 3), dtype=np.uint8)
    rng = np.random.RandomState(seed)
    arr[:] = rng.randint(0, 255, size=3, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=50)
    return buf.getvalue()


class FakeWorldModelService(video_model_pb2_grpc.WorldModelServiceServicer):
    def start_session(self, request, context):
        print(
            f"[fake] start_session: camera_specs={len(request.camera_specs)}, "
            f"initial_frames={len(request.initial_frames)}, "
            f"hdmap={len(request.static_world_map.hdmap_parquets)} bytes"
        )
        return video_model_pb2.SessionId(session_id="fake-session-001")

    def render_video_chunk(self, request, context):
        n_poses = len(request.rig_trajectory.poses)
        print(f"[fake] render_video_chunk: {n_poses} poses")
        cam_out = video_model_pb2.CameraOutput(
            camera_logical_id="camera_front_wide_120fov",
        )
        for i in range(n_poses):
            cam_out.rgb_frames.append(
                video_model_pb2.Image(
                    data=_make_fake_frame(i),
                    format=video_model_pb2.JPEG,
                )
            )
        return video_model_pb2.VideoChunkReturn(camera_outputs=[cam_out])

    def close_session(self, request, context):
        print(f"[fake] close_session: {request.session_id}")
        return common_pb2.Empty()

    def get_version(self, request, context):
        return common_pb2.VersionId(version_id="fake-0.1.0")


def main() -> None:
    import grpc

    port = 50052
    server = grpc.server(thread_pool=threading.ThreadPoolExecutor(max_workers=4))
    video_model_pb2_grpc.add_WorldModelServiceServicer_to_server(
        FakeWorldModelService(), server
    )
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    print(f"[fake] gRPC server listening on 0.0.0.0:{port} (CPU-only)")
    print("[fake] Press Ctrl+C to stop.")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    main()
