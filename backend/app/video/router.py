import asyncio
import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse

from backend.app.config import (
    DEFAULT_VIDEO_CHUNK_SIZE,
    DEFAULT_VIDEO_OVERLAP,
    UPLOADS_DIR,
    VIDEO_CHUNK_PRESETS,
    VIDEO_OUTPUTS_DIR,
    WORK_DIR,
)
from backend.app.schemas import (
    VideoJobCreateResponse,
    VideoJobStatusResponse,
    VideoProbeResponse,
)
from backend.app.model_registry import resolve_model_id
from backend.app.video.job_manager import video_job_manager
from backend.app.video.probe import VideoProbeError, probe_video

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/video", tags=["Video Super-Resolution"])


def sanitize_filename(filename: Optional[str]) -> str:
    """Sanitize uploaded filename, stripping directory traversal characters."""
    if not filename:
        return "uploaded_video.mp4"
    name = Path(filename).name
    # Keep alphanumeric, dots, underscores, dashes
    sanitized = re.sub(r"[^\w\.\-]", "_", name)
    return sanitized if sanitized else "uploaded_video.mp4"


@router.post("/probe", response_model=VideoProbeResponse)
async def probe_video_endpoint(file: UploadFile = File(...)):
    """Analyze and validate video file without running super-resolution.

    Useful for frontend pre-flight validation (dimensions, CFR, duration, codec).
    """
    temp_id = uuid.uuid4().hex[:8]
    temp_file = WORK_DIR / f"probe_{temp_id}_{sanitize_filename(file.filename)}"

    try:
        # Stream upload directly to disk
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run ffprobe verification
        probe_res = await asyncio.to_thread(probe_video, temp_file, enforce_limits=True)
        return VideoProbeResponse(**probe_res.to_dict())

    except VideoProbeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error probing video: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống khi phân tích video: {exc}",
        )
    finally:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)


@router.post(
    "/upscale",
    response_model=VideoJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_video_upscale_job(
    file: UploadFile = File(...),
    chunk_size: int = Form(DEFAULT_VIDEO_CHUNK_SIZE),
    overlap: int = Form(DEFAULT_VIDEO_OVERLAP),
    model_id: str = Form(""),
):
    """Submit a video for asynchronous x4 super-resolution.

    Returns HTTP 202 Accepted with a job_id. Clients can poll GET /api/video/jobs/{job_id}
    or listen to SSE events at GET /api/video/jobs/{job_id}/stream.
    """
    # Validate chunk_size and overlap
    if chunk_size < 4 or chunk_size > 32:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"chunk_size ({chunk_size}) phải nằm trong khoảng [4, 32].",
        )

    if overlap < 1 or overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"overlap ({overlap}) phải >= 1 và nhỏ hơn chunk_size ({chunk_size}).",
        )

    try:
        model_id = resolve_model_id(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    safe_name = sanitize_filename(file.filename)
    job_uuid = uuid.uuid4().hex
    input_path = UPLOADS_DIR / f"{job_uuid}_{safe_name}"

    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Quick pre-validation probe to fail fast on invalid video formats
        probe_res = await asyncio.to_thread(probe_video, input_path, enforce_limits=True)

    except VideoProbeError as exc:
        if input_path.exists():
            input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        if input_path.exists():
            input_path.unlink(missing_ok=True)
        logger.exception("Error saving upload video: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Không thể lưu tệp video tải lên: {exc}",
        )

    # Submit to asynchronous background worker queue
    job_id = video_job_manager.submit_job(
        input_path=input_path,
        source_filename=safe_name,
        chunk_size=chunk_size,
        overlap=overlap,
        model_id=model_id,
    )

    return VideoJobCreateResponse(
        job_id=job_id,
        status="pending",
        message="Yêu cầu siêu phân giải video đã được đưa vào hàng đợi xử lý.",
    )


@router.get("/jobs/{job_id}", response_model=VideoJobStatusResponse)
async def get_video_job_status(job_id: str):
    """Get the current progress status and metadata of a video job."""
    job = video_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy tác vụ video với ID '{job_id}'.",
        )
    return VideoJobStatusResponse(**job.to_status_dict())


@router.get("/jobs/{job_id}/stream")
async def stream_video_job_progress(job_id: str):
    """Server-Sent Events (SSE) endpoint providing realtime progress updates."""
    job = video_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy tác vụ video với ID '{job_id}'.",
        )

    return StreamingResponse(
        video_job_manager.subscribe(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/results/{job_id}")
async def get_video_result(job_id: str, download: bool = False):
    """Stream or download the completed super-resolved MP4 video."""
    video_path = VIDEO_OUTPUTS_DIR / f"{job_id}_x4.mp4"
    if not video_path.is_file():
        # Check if job exists and still in progress
        job = video_job_manager.get_job(job_id)
        if job and job.status != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tác vụ video đang ở trạng thái '{job.status}'. Kết quả chưa sẵn sàng.",
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tệp video kết quả không tồn tại hoặc đã bị xóa.",
        )

    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{job_id}_x4.mp4"'

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"{job_id}_x4.mp4" if download else None,
        headers=headers,
    )


@router.get("/results/{job_id}/download")
async def download_video_result(job_id: str):
    """Convenience endpoint to directly trigger file download of super-resolved video."""
    return await get_video_result(job_id, download=True)
