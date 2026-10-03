import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Set

from backend.app.config import (
    DEFAULT_VIDEO_CHUNK_SIZE,
    DEFAULT_VIDEO_OVERLAP,
    VIDEO_OUTPUTS_DIR,
    WORK_DIR,
)
from backend.app.model_registry import (
    activate_model,
    get_inference_lock,
    get_inference_semaphore,
    list_available_models,
    resolve_model_id,
)
from backend.app.video.chunk_stream import compute_chunk_windows, stream_rb_video_chunks
from backend.app.video.decode import decode_video_frames
from backend.app.video.encode import encode_video_from_frames
from backend.app.video.probe import VideoProbeResult, probe_video

logger = logging.getLogger(__name__)


@dataclass
class VideoJob:
    job_id: str
    source_filename: str
    input_path: Path
    model_id: str = ""
    checkpoint_filename: str = ""
    device: Optional[str] = None
    chunk_size: int = DEFAULT_VIDEO_CHUNK_SIZE
    overlap: int = DEFAULT_VIDEO_OVERLAP
    output_path: Optional[Path] = None
    status: str = "pending"  # pending | probing | decoding | inferring | encoding | completed | failed
    progress_percent: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    current_chunk: int = 0
    total_chunks: int = 0
    eta_seconds: Optional[float] = None
    error_message: Optional[str] = None
    input_meta: Optional[dict] = None
    output_meta: Optional[dict] = None
    timing: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    subscribers: List[asyncio.Queue] = field(default_factory=list)

    def to_status_dict(self) -> dict:
        result_url = f"/api/video/results/{self.job_id}" if self.status == "completed" else None
        download_url = f"/api/video/results/{self.job_id}/download" if self.status == "completed" else None
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "checkpoint_filename": self.checkpoint_filename,
            "device": self.device,
            "status": self.status,
            "progress_percent": round(self.progress_percent, 1),
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "current_chunk": self.current_chunk,
            "total_chunks": self.total_chunks,
            "eta_seconds": round(self.eta_seconds, 1) if self.eta_seconds is not None else None,
            "error_message": self.error_message,
            "result_url": result_url,
            "download_url": download_url,
            "input_meta": self.input_meta,
            "output_meta": self.output_meta,
            "timing": self.timing,
        }

    def notify_subscribers(self):
        data_str = json.dumps(self.to_status_dict())
        for q in list(self.subscribers):
            try:
                q.put_nowait(data_str)
            except Exception:
                pass


class VideoJobManager:
    """Manages asynchronous video super-resolution background jobs with SSE broadcasting."""

    def __init__(self):
        self._jobs: Dict[str, VideoJob] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    def start(self):
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("VideoJobManager background worker started.")

    async def stop(self):
        self._running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            logger.info("VideoJobManager background worker stopped.")

    def submit_job(
        self,
        input_path: Path,
        source_filename: str,
        chunk_size: int = DEFAULT_VIDEO_CHUNK_SIZE,
        overlap: int = DEFAULT_VIDEO_OVERLAP,
        model_id: str = "",
    ) -> str:
        model_id = resolve_model_id(model_id)
        checkpoint = next(item for item in list_available_models() if item.id == model_id)
        job_id = uuid.uuid4().hex
        out_video_path = VIDEO_OUTPUTS_DIR / f"{job_id}_x4.mp4"

        job = VideoJob(
            job_id=job_id,
            source_filename=source_filename,
            input_path=Path(input_path),
            model_id=model_id,
            checkpoint_filename=checkpoint.checkpoint_filename,
            chunk_size=chunk_size,
            overlap=overlap,
            output_path=out_video_path,
        )
        self._jobs[job_id] = job
        self._queue.put_nowait(job_id)
        logger.info("Submitted video job %s (file: '%s', chunk: %d, overlap: %d)",
                    job_id, source_filename, chunk_size, overlap)
        return job_id

    def get_job(self, job_id: str) -> Optional[VideoJob]:
        return self._jobs.get(job_id)

    async def subscribe(self, job_id: str) -> AsyncGenerator[str, None]:
        job = self.get_job(job_id)
        if not job:
            return

        queue: asyncio.Queue = asyncio.Queue()
        job.subscribers.append(queue)

        # Send initial status immediately
        yield f"data: {json.dumps(job.to_status_dict())}\n\n"

        try:
            while True:
                msg = await queue.get()
                yield f"data: {msg}\n\n"
                if job.status in ("completed", "failed"):
                    break
        finally:
            if queue in job.subscribers:
                job.subscribers.remove(queue)

    async def _worker_loop(self):
        """Sequential single-job background processing loop."""
        while self._running:
            try:
                job_id = await self._queue.get()
            except asyncio.CancelledError:
                break

            job = self._jobs.get(job_id)
            if not job:
                self._queue.task_done()
                continue

            try:
                await self._process_job(job)
            except Exception as exc:
                logger.exception("Unexpected error processing video job %s: %s", job_id, exc)
                job.status = "failed"
                job.error_message = str(exc)
                job.updated_at = time.time()
                job.notify_subscribers()
            finally:
                self._queue.task_done()

    async def _process_job(self, job: VideoJob):
        job_work_dir = WORK_DIR / f"job_{job.job_id}"
        raw_frames_dir = job_work_dir / "raw_frames"
        sr_frames_dir = job_work_dir / "sr_frames"
        job_work_dir.mkdir(parents=True, exist_ok=True)

        t_start_total = time.perf_counter()

        try:
            # 1. Probing
            job.status = "probing"
            job.progress_percent = 5.0
            job.updated_at = time.time()
            job.notify_subscribers()

            t0 = time.perf_counter()
            probe_res = await asyncio.to_thread(probe_video, job.input_path, enforce_limits=True)
            job.input_meta = probe_res.to_dict()
            job.total_frames = probe_res.total_frames
            job.timing["probe_seconds"] = round(time.perf_counter() - t0, 2)

            # 2. Decoding
            job.status = "decoding"
            job.progress_percent = 15.0
            job.updated_at = time.time()
            job.notify_subscribers()

            t_dec = time.perf_counter()
            decoded_count = await asyncio.to_thread(
                decode_video_frames,
                job.input_path,
                raw_frames_dir,
                probe_res.total_frames,
            )
            job.total_frames = decoded_count
            job.timing["decode_seconds"] = round(time.perf_counter() - t_dec, 2)

            # 3. Super-Resolution Inference
            windows = compute_chunk_windows(decoded_count, job.chunk_size, job.overlap)
            job.total_chunks = len(windows)
            job.status = "inferring"
            job.progress_percent = 20.0
            job.updated_at = time.time()
            job.notify_subscribers()

            t_inf_start = time.perf_counter()

            def on_inference_progress(done_frames: int, total: int, msg: str):
                # Map inference progress from 20% to 85%
                job.current_frame = done_frames
                infer_ratio = done_frames / max(total, 1)
                job.progress_percent = 20.0 + (infer_ratio * 65.0)

                # Estimate ETA
                elapsed = time.perf_counter() - t_inf_start
                if done_frames > 0 and infer_ratio < 1.0:
                    speed_fps = done_frames / elapsed
                    remaining_frames = total - done_frames
                    job.eta_seconds = remaining_frames / max(speed_fps, 0.01)

                job.updated_at = time.time()
                job.notify_subscribers()

            def run_stream_chunks():
                with get_inference_lock():
                    net, device = activate_model(job.model_id)
                    job.device = str(device)
                    return stream_rb_video_chunks(
                        net=net,
                        input_frames_dir=raw_frames_dir,
                        output_frames_dir=sr_frames_dir,
                        total_frames=decoded_count,
                        chunk_size=job.chunk_size,
                        overlap=job.overlap,
                        device=device,
                        progress_callback=on_inference_progress,
                    )

            semaphore = get_inference_semaphore()
            async with semaphore:
                await asyncio.to_thread(run_stream_chunks)

            job.timing["infer_seconds"] = round(time.perf_counter() - t_inf_start, 2)

            # 4. Encoding
            job.status = "encoding"
            job.progress_percent = 90.0
            job.eta_seconds = 5.0
            job.updated_at = time.time()
            job.notify_subscribers()

            t_enc = time.perf_counter()
            out_probe = await asyncio.to_thread(
                encode_video_from_frames,
                frames_dir=sr_frames_dir,
                output_video_path=job.output_path,
                fps_str=probe_res.fps,
                source_video_path=job.input_path if probe_res.has_audio else None,
                crf=18,
                preset="medium",
            )
            job.output_meta = out_probe.to_dict()
            job.timing["encode_seconds"] = round(time.perf_counter() - t_enc, 2)
            job.timing["total_seconds"] = round(time.perf_counter() - t_start_total, 2)

            # 5. Completed
            job.status = "completed"
            job.progress_percent = 100.0
            job.eta_seconds = 0.0
            job.updated_at = time.time()
            job.notify_subscribers()
            logger.info("Video job %s completed successfully in %.2fs!", job.job_id, job.timing["total_seconds"])

        except Exception as exc:
            logger.exception("Error processing video job %s: %s", job.job_id, exc)
            job.status = "failed"
            job.error_message = str(exc)
            job.updated_at = time.time()
            job.notify_subscribers()

        finally:
            # Always clean up temporary uncompressed PNG frames to save disk space
            if job_work_dir.exists():
                shutil.rmtree(job_work_dir, ignore_errors=True)
                logger.debug("Cleaned up temp frames directory '%s'", job_work_dir)


# Global job manager singleton instance
video_job_manager = VideoJobManager()
