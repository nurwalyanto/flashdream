# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import io
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path

import grpc
import numpy as np
from loguru import logger
from omnidreams.grpc.protos import (
    camera_pb2,
    common_pb2,
    video_model_pb2,
    video_model_pb2_grpc,
)
from omnidreams.interactive_drive.backends.base import RenderBackend
from omnidreams.interactive_drive.config import BevConfig, ChunkConfig, RasterConfig
from omnidreams.interactive_drive.rasterizer import LudusConditionRasterizer
from omnidreams.interactive_drive.types import (
    FrameChunk,
    PresentedFrame,
    SceneBundle,
    TrajectoryChunk,
    VideoModelTimings,
)
from PIL import Image
from scipy.spatial.transform import Rotation

_FIRST_STEADY_STATE_WARMUP_MESSAGE = "Optimizing world model..."


def _matrix_to_pose_proto(matrix: np.ndarray) -> common_pb2.Pose:
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    return common_pb2.Pose(
        vec=common_pb2.Vec3(x=float(t[0]), y=float(t[1]), z=float(t[2])),
        quat=common_pb2.Quat(
            w=float(quat_xyzw[3]),
            x=float(quat_xyzw[0]),
            y=float(quat_xyzw[1]),
            z=float(quat_xyzw[2]),
        ),
    )


def _extract_hdmap_parquets(scene_path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(scene_path, "r") as src, zipfile.ZipFile(buf, "w") as dst:
        for name in src.namelist():
            if name.startswith("clipgt/") and name.endswith(".parquet"):
                dst.writestr(name, src.read(name))
    return buf.getvalue()


def _calibration_to_camera_spec(
    calib: object, resolution_wh: tuple[int, int]
) -> camera_pb2.CameraSpec:
    w, h = resolution_wh
    poly = list(calib.polynomial)  # type: ignore[attr-defined]
    poly_padded = (poly + [0.0] * 6)[:6]
    return camera_pb2.CameraSpec(
        logical_id=calib.logical_name,  # type: ignore[attr-defined]
        resolution_h=h,
        resolution_w=w,
        ftheta_param=camera_pb2.FthetaCameraParam(
            principal_point_x=float(calib.cx),  # type: ignore[attr-defined]
            principal_point_y=float(calib.cy),  # type: ignore[attr-defined]
            reference_poly=(
                camera_pb2.FthetaCameraParam.PIXELDIST_TO_ANGLE
                if calib.is_backward_polynomial  # type: ignore[attr-defined]
                else camera_pb2.FthetaCameraParam.ANGLE_TO_PIXELDIST
            ),
            pixeldist_to_angle_poly=poly_padded if calib.is_backward_polynomial else [],  # type: ignore[attr-defined]
            angle_to_pixeldist_poly=[] if calib.is_backward_polynomial else poly_padded,  # type: ignore[attr-defined]
            max_angle=1.57,
            linear_cde=camera_pb2.LinearCde(
                linear_c=float(calib.linear_cde[0]),  # type: ignore[attr-defined]
                linear_d=float(calib.linear_cde[1]),  # type: ignore[attr-defined]
                linear_e=float(calib.linear_cde[2]),  # type: ignore[attr-defined]
            ),
        ),
    )


class GrpcWorldModelRenderBackend(RenderBackend):
    def __init__(
        self,
        endpoint: str,
        scene_path: Path,
        chunk: ChunkConfig,
        raster: RasterConfig,
        bev: BevConfig | None = None,
    ) -> None:
        super().__init__(chunk=chunk, raster=raster)
        self._channel = grpc.insecure_channel(endpoint)
        self._stub = video_model_pb2_grpc.WorldModelServiceStub(self._channel)
        self._scene_path = scene_path
        self._hdmap_parquet_bytes: bytes | None = None
        self._rasterizer = LudusConditionRasterizer(raster, bev=bev)
        self._session_id: str | None = None
        self._scene: SceneBundle | None = None
        self._next_chunk_count = 0

    @property
    def can_prewarm(self) -> bool:
        return True

    @property
    def optimizes_on_first_chunk(self) -> bool:
        return False

    def warmup_model(self) -> None:
        self._hdmap_parquet_bytes = _extract_hdmap_parquets(self._scene_path)
        try:
            self._stub.get_version(common_pb2.Empty(), timeout=10)
        except Exception as e:
            logger.warning(f"[grpc-cloud] server reachable check: {e}")
        logger.info("[grpc-cloud] model warmup complete (server-side)")

    def load_scene(self, scene: SceneBundle) -> None:
        self._scene = scene
        self._next_chunk_count = 0
        load_start = time.perf_counter()

        self._rasterizer.load_scene(scene)
        rasterizer_end = time.perf_counter()

        w, h = self._raster.resolution_wh
        spec = _calibration_to_camera_spec(scene.selected_camera, (w, h))

        sensor_to_rig = scene.selected_camera.sensor_to_rig_flu
        rig_to_camera = np.linalg.inv(sensor_to_rig)
        rig_to_cam_pose = _matrix_to_pose_proto(rig_to_camera)

        buf = io.BytesIO()
        Image.fromarray(scene.initial_rgb).save(buf, format="JPEG", quality=95)
        initial_frame_bytes = buf.getvalue()

        req = video_model_pb2.SessionRequest(
            static_world_map=video_model_pb2.StaticWorldMap(
                hdmap_parquets=self._hdmap_parquet_bytes,
            ),
            text_prompt=video_model_pb2.TextPrompt(positive=scene.prompt),
            camera_specs=[spec],
            initial_frames=[
                video_model_pb2.Image(
                    data=initial_frame_bytes,
                    format=video_model_pb2.JPEG,
                )
            ],
            rig_to_camera=[rig_to_cam_pose],
        )
        resp = self._stub.start_session(req)
        self._session_id = resp.session_id

        session_end = time.perf_counter()
        logger.info(
            "[grpc-cloud] load_scene "
            f"rasterizer_ms={(rasterizer_end - load_start) * 1000.0:.1f} "
            f"session_ms={(session_end - rasterizer_end) * 1000.0:.1f} "
            f"total_ms={(session_end - load_start) * 1000.0:.1f}",
        )

    def render_first_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        return self._render_chunk(trajectory, annotate_first_transition=True)

    def render_next_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        return self._render_chunk(trajectory, annotate_first_transition=False)

    def _render_chunk(
        self,
        trajectory: TrajectoryChunk,
        *,
        annotate_first_transition: bool,
    ) -> FrameChunk:
        scene = self._require_scene()
        chunk_start = time.perf_counter()

        raster_chunk = self._rasterizer.render_chunk(
            rig_poses_world=trajectory.rig_poses_world,
            timestamps_us=trajectory.timestamps_us,
        )
        raster_end = time.perf_counter()

        traj_proto = common_pb2.Trajectory(
            poses=[
                common_pb2.PoseAtTime(
                    timestamp_us=int(ts),
                    pose=_matrix_to_pose_proto(pose),
                )
                for ts, pose in zip(
                    trajectory.timestamps_us, trajectory.rig_poses_world
                )
            ]
        )
        chunk_req = video_model_pb2.VideoChunkRequest(
            session_id=video_model_pb2.SessionId(session_id=self._session_id),
            rig_trajectory=traj_proto,
        )
        chunk_resp = self._stub.render_video_chunk(chunk_req)
        model_end = time.perf_counter()

        model_frames = _decode_camera_output(chunk_resp)

        if len(model_frames) != len(raster_chunk.frames):
            logger.warning(
                f"[grpc-cloud] frame count mismatch: "
                f"cloud={len(model_frames)} raster={len(raster_chunk.frames)}"
            )

        merged_frames = self._merge_frames(
            raster_chunk.frames,
            model_frames,
            annotate_first_transition=annotate_first_transition,
        )
        merge_end = time.perf_counter()

        self._next_chunk_count += 1
        total_ms = (merge_end - chunk_start) * 1000.0
        if (
            self._next_chunk_count <= 3
            or self._next_chunk_count % 10 == 0
            or total_ms > 500.0
        ):
            logger.info(
                "[grpc-cloud] chunk "
                f"index={self._next_chunk_count} "
                f"frames={len(trajectory.timestamps_us)} "
                f"raster_ms={(raster_end - chunk_start) * 1000.0:.1f} "
                f"model_ms={(model_end - raster_end) * 1000.0:.1f} "
                f"merge_ms={(merge_end - model_end) * 1000.0:.1f} "
                f"total_ms={total_ms:.1f}",
            )

        return FrameChunk(
            frames=merged_frames,
            boundary_state_after_chunk=trajectory.boundary_state_after_chunk,
            source_name="grpc-cloud",
            video_model_timings=VideoModelTimings(
                condition_start_time=chunk_start,
                condition_ready_time=raster_end,
                model_start_time=raster_end,
                model_ready_time=model_end,
                merge_start_time=model_end,
                merge_ready_time=merge_end,
            ),
        )

    def reset(self) -> None:
        if self._session_id is not None:
            try:
                self._stub.close_session(
                    video_model_pb2.SessionCloseRequest(
                        session_id=self._session_id
                    )
                )
            except Exception as e:
                logger.warning(f"[grpc-cloud] close_session error: {e}")
            self._session_id = None
        self._next_chunk_count = 0

    def reset_scene_conditioning(self) -> None:
        self.reset()

    def close(self) -> None:
        self.reset()
        self._rasterizer.cleanup()
        self._channel.close()

    def _require_scene(self) -> SceneBundle:
        if self._scene is None:
            raise RuntimeError(
                "warmup() must be called before rendering world-model chunks"
            )
        return self._scene

    def _merge_frames(
        self,
        raster_frames: Sequence[PresentedFrame],
        model_frames: Sequence[np.ndarray],
        *,
        annotate_first_transition: bool = False,
    ) -> tuple[PresentedFrame, ...]:
        merged: list[PresentedFrame] = []
        last_index = len(raster_frames) - 1
        for index, (rf, model_rgb) in enumerate(
            zip(raster_frames, model_frames, strict=True)
        ):
            merged.append(
                PresentedFrame(
                    timestamp_us=rf.timestamp_us,
                    rgb_host_uint8=rf.rgb_host_uint8,
                    depth_host_f32=rf.depth_host_f32,
                    rgb_native=rf.rgb_native,
                    depth_native=rf.depth_native,
                    model_rgb_host_uint8=model_rgb,
                    bev_host_uint8=rf.bev_host_uint8,
                    status_message=(
                        _FIRST_STEADY_STATE_WARMUP_MESSAGE
                        if annotate_first_transition and index == last_index
                        else None
                    ),
                )
            )
        return tuple(merged)


def _decode_camera_output(
    resp: video_model_pb2.VideoChunkReturn,
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for cam_out in resp.camera_outputs:
        for img_msg in cam_out.rgb_frames:
            arr = np.frombuffer(img_msg.data, dtype=np.uint8)
            pil = Image.open(io.BytesIO(arr.tobytes()))
            frames.append(np.array(pil.convert("RGB"), dtype=np.uint8))
    return frames
