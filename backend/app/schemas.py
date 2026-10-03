from typing import List, Optional
from pydantic import BaseModel, Field


class ImageLimits(BaseModel):
    max_upload_size_bytes: int
    max_image_dimension: int
    max_image_pixels: int
    min_image_dimension: int


class VideoLimits(BaseModel):
    max_upload_mb: int
    max_duration_seconds: float
    max_frames: int
    max_input_pixels: int
    default_chunk_size: int
    default_overlap: int
    chunk_presets: List[int]


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    version: str = Field(..., example="1.0.0")
    device: str = Field(..., example="cpu")
    torch_version: str = Field(..., example="2.10.0+cpu")
    cuda_available: bool = Field(..., example=False)
    gpu_name: Optional[str] = Field(None, example="NVIDIA GeForce GTX 1650")
    active_models_loaded: List[str] = Field(default_factory=list)
    image_limits: ImageLimits
    video_limits: Optional[VideoLimits] = None


class ModelItem(BaseModel):
    id: str = Field(..., example="realbasicvsr_anime_v3")
    name: str = Field(..., example="RealBasicVSR Anime V3")
    description: str = Field(..., example="Super-resolution x4 for anime image/video.")
    scale: int = Field(4, example=4)
    available: bool = Field(True, example=True)
    device: str = Field(..., example="cpu")
    checkpoint_filename: str = ""
    checkpoint_path: str = ""
    loaded: bool = False
    error: Optional[str] = None


class ModelsResponse(BaseModel):
    models: List[ModelItem]


class ImageSize(BaseModel):
    width: int
    height: int


class UpscaleResponse(BaseModel):
    result_id: str = Field(..., example="c7b19a12-8720-4a88-8120-0082496a84d9")
    model_id: str = Field(..., example="realbasicvsr_anime_v3")
    checkpoint_filename: str = ""
    scale: int = Field(4, example=4)
    input_size: ImageSize
    output_size: ImageSize
    device: str = Field(..., example="cpu")
    inference_time_ms: int = Field(..., example=7950)
    result_url: str = Field(..., example="/api/results/c7b19a12-8720-4a88-8120-0082496a84d9")


class ErrorResponse(BaseModel):
    detail: str


class VideoProbeResponse(BaseModel):
    width: int
    height: int
    duration_seconds: float
    fps: str
    fps_float: float
    total_frames: int
    video_codec: str
    has_audio: bool
    audio_codec: Optional[str] = None
    file_size_bytes: int


class VideoJobCreateResponse(BaseModel):
    job_id: str
    status: str = "pending"
    message: str


class VideoJobStatusResponse(BaseModel):
    job_id: str
    model_id: str = ""
    checkpoint_filename: str = ""
    device: Optional[str] = None
    status: str  # pending | probing | decoding | inferring | encoding | completed | failed
    progress_percent: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    current_chunk: int = 0
    total_chunks: int = 0
    eta_seconds: Optional[float] = None
    error_message: Optional[str] = None
    result_url: Optional[str] = None
    download_url: Optional[str] = None
    input_meta: Optional[dict] = None
    output_meta: Optional[dict] = None
    timing: Optional[dict] = None

