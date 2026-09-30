import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.app.config import (
    CLEANUP_INTERVAL_MINUTES,
    CLEANUP_MAX_AGE_HOURS,
    CORS_ORIGINS,
    CUDA_DEVICE_INDEX,
    DEFAULT_VIDEO_CHUNK_SIZE,
    DEFAULT_VIDEO_OVERLAP,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_PIXELS_GPU,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_VIDEO_DURATION_SECONDS,
    MAX_VIDEO_FRAMES,
    MAX_VIDEO_INPUT_PIXELS,
    MAX_VIDEO_UPLOAD_MB,
    MIN_IMAGE_DIMENSION,
    VIDEO_CHUNK_PRESETS,
)
from backend.app.inference.realbasicvsr_pipeline import infer_rb_single
from backend.app.model_registry import (
    get_active_model_ids,
    get_inference_semaphore,
    get_inference_lock,
    get_model,
    list_available_models,
    load_model_registry,
)

from backend.app.schemas import (
    HealthResponse,
    ImageLimits,
    ImageSize,
    ModelsResponse,
    UpscaleResponse,
    VideoLimits,
)
from backend.app.video.job_manager import video_job_manager
from backend.app.video.router import router as video_router
from backend.app.storage import (
    cleanup_old_files,
    get_result_path,
    is_valid_uuid,
    save_output_image,
    validate_and_read_image,
)

logger = logging.getLogger(__name__)


async def _cleanup_worker(interval_minutes: int, max_age_hours: int):
    """Background loop to periodically purge expired result and temporary files."""
    while True:
        try:
            await asyncio.sleep(interval_minutes * 60)
            removed = await asyncio.to_thread(cleanup_old_files, max_age_hours)
            if removed > 0:
                logger.info("[Cleanup] Automatically removed %d expired file(s).", removed)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("[Cleanup] Error during periodic cleanup: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan context manager: Load model once at startup and schedule cleanup."""
    print("=" * 60)
    print("Starting Post-Production AI Upscale Server...")
    count, device = load_model_registry()
    print(f"Server ready with {count} active model(s) on device: {device}")

    # Initial cleanup on server boot
    initial_removed = await asyncio.to_thread(cleanup_old_files, CLEANUP_MAX_AGE_HOURS)
    if initial_removed > 0:
        print(f"[Cleanup] Startup cleanup removed {initial_removed} expired file(s).")

    # Start periodic background cleanup worker
    cleanup_task = asyncio.create_task(
        _cleanup_worker(CLEANUP_INTERVAL_MINUTES, CLEANUP_MAX_AGE_HOURS)
    )
    # Start video job background manager
    video_job_manager.start()
    print("=" * 60)
    try:
        yield
    finally:
        print("Shutting down Post-Production AI Upscale Server...")
        cleanup_task.cancel()
        await video_job_manager.stop()
        await asyncio.gather(cleanup_task, return_exceptions=True)


app = FastAPI(
    title="AI Super-Resolution API",
    description="Local AI Super-Resolution API using RealBasicVSR Anime V3 (x4).",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint: returns server status, active device, GPU name, and loaded models."""
    cuda_avail = torch.cuda.is_available()
    active_model = get_model("realbasicvsr_anime_v3")
    actual_device = str(active_model[1]) if active_model else "none"
    gpu_name = None
    if cuda_avail and 0 <= CUDA_DEVICE_INDEX < torch.cuda.device_count():
        gpu_name = torch.cuda.get_device_name(CUDA_DEVICE_INDEX)
    max_pixels = MAX_IMAGE_PIXELS_GPU if active_model and active_model[1].type == "cuda" else MAX_IMAGE_PIXELS
    loaded = get_active_model_ids()
    return HealthResponse(
        status="ok" if active_model else "degraded",
        version="1.0.0",
        device=actual_device,
        torch_version=torch.__version__,
        cuda_available=cuda_avail,
        gpu_name=gpu_name,
        active_models_loaded=loaded,
        image_limits=ImageLimits(
            max_upload_size_bytes=MAX_UPLOAD_SIZE_BYTES,
            max_image_dimension=MAX_IMAGE_DIMENSION,
            max_image_pixels=max_pixels,
            min_image_dimension=MIN_IMAGE_DIMENSION,
        ),
        video_limits=VideoLimits(
            max_upload_mb=MAX_VIDEO_UPLOAD_MB,
            max_duration_seconds=MAX_VIDEO_DURATION_SECONDS,
            max_frames=MAX_VIDEO_FRAMES,
            max_input_pixels=MAX_VIDEO_INPUT_PIXELS,
            default_chunk_size=DEFAULT_VIDEO_CHUNK_SIZE,
            default_overlap=DEFAULT_VIDEO_OVERLAP,
            chunk_presets=VIDEO_CHUNK_PRESETS,
        ),
    )


@app.get("/api/models", response_model=ModelsResponse, tags=["Models"])
async def get_models():
    """Get list of registered models, supported scales, and real availability."""
    models = list_available_models()
    return ModelsResponse(models=models)


@app.post("/api/upscale", response_model=UpscaleResponse, tags=["Inference"])
async def upscale_image(
    image: UploadFile = File(..., description="Image file to upscale"),
    model_id: str = Form("realbasicvsr_anime_v3", description="ID of the model to use"),
):
    """Upscale a single image by 4x using specified AI model."""
    # 1. Look up requested model in registry
    model_entry = get_model(model_id)
    if model_entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' không khả dụng hoặc chưa được nạp checkpoint.",
        )
    net, device = model_entry

    # 2. Validate and read input image
    max_pixels = MAX_IMAGE_PIXELS_GPU if device.type == "cuda" else MAX_IMAGE_PIXELS
    img_rgb, width, height = await validate_and_read_image(image, max_pixels=max_pixels)

    # 3. Concurrency control: serialize CPU/GPU heavy inference
    semaphore = get_inference_semaphore()
    def run_inference():
        with get_inference_lock():
            return infer_rb_single(net, img_rgb, device)

    async with semaphore:
        t_start = time.perf_counter()
        try:
            out_rgb = await asyncio.to_thread(run_inference)
        except (MemoryError, torch.cuda.OutOfMemoryError) as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.exception("Insufficient memory during inference")
            raise HTTPException(status_code=503, detail="Không đủ bộ nhớ (RAM/VRAM) để xử lý ảnh này. Hãy chọn ảnh nhỏ hơn.") from exc
        except RuntimeError as exc:
            logger.exception("Inference failed")
            if "out of memory" in str(exc).lower() or "memory" in str(exc).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise HTTPException(status_code=503, detail="Không đủ bộ nhớ (RAM/VRAM) để xử lý ảnh này. Hãy chọn ảnh nhỏ hơn.") from exc
            raise HTTPException(status_code=500, detail="Model không thể xử lý ảnh này.") from exc
    elapsed_ms = int((time.perf_counter() - t_start) * 1000)

    # 4. Save result safely with random UUID4
    result_id = str(uuid.uuid4())
    try:
        save_output_image(out_rgb, result_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lưu tệp ảnh kết quả: {e}",
        )

    out_h, out_w = out_rgb.shape[:2]
    result_url = f"/api/results/{result_id}"

    return UpscaleResponse(
        result_id=result_id,
        model_id=model_id,
        scale=4,
        input_size=ImageSize(width=width, height=height),
        output_size=ImageSize(width=out_w, height=out_h),
        device=str(device),
        inference_time_ms=elapsed_ms,
        result_url=result_url,
    )


@app.get("/api/results/{result_id}", tags=["Results"])
async def get_result_file(
    result_id: str,
    download: bool = Query(False, description="Set true to force download"),
):
    """Retrieve preview or download of super-resolved result image."""
    if not is_valid_uuid(result_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mã kết quả (result_id) không đúng định dạng UUID.",
        )

    file_path = get_result_path(result_id)
    if file_path is None or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy tệp kết quả. Tệp có thể đã hết hạn hoặc bị xóa.",
        )

    filename = f"upscale_x4_{result_id}.png" if download else None
    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        filename=filename,
    )


# Register Video API router
app.include_router(video_router)

