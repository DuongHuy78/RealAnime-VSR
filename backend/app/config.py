import os
import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Base directory: repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
load_dotenv(BACKEND_DIR / ".env")


def configured_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default.resolve()
    path = Path(value)
    return (path if path.is_absolute() else BACKEND_DIR / path).resolve()

# Checkpoint paths
CHECKPOINT_PATH = configured_path("CHECKPOINT_PATH", REPO_ROOT / "best.pth")

# Storage directories
UPLOADS_DIR = configured_path("UPLOADS_DIR", REPO_ROOT / "uploads")
OUTPUTS_DIR = configured_path("OUTPUTS_DIR", REPO_ROOT / "outputs")
MODELS_DIR = configured_path("MODELS_DIR", REPO_ROOT / "models")
WORK_DIR = configured_path("WORK_DIR", REPO_ROOT / "work")
VIDEO_OUTPUTS_DIR = configured_path("VIDEO_OUTPUTS_DIR", OUTPUTS_DIR / "videos")

# Ensure storage directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# FFmpeg & FFprobe binary locations
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

# Video limits
MAX_VIDEO_UPLOAD_MB = int(os.getenv("MAX_VIDEO_UPLOAD_MB", "25"))
MAX_VIDEO_DURATION_SECONDS = float(os.getenv("MAX_VIDEO_DURATION_SECONDS", "10.0"))
MAX_VIDEO_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", "300"))
MAX_VIDEO_INPUT_PIXELS = int(os.getenv("MAX_VIDEO_INPUT_PIXELS", "76800"))
DEFAULT_VIDEO_CHUNK_SIZE = int(os.getenv("DEFAULT_VIDEO_CHUNK_SIZE", "7"))
DEFAULT_VIDEO_OVERLAP = int(os.getenv("DEFAULT_VIDEO_OVERLAP", "2"))
VIDEO_CHUNK_PRESETS = [7, 12, 16, 24]

# Conservative CPU defaults. Increase only after measuring process RSS or GPU VRAM.
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15")) * 1024 * 1024
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "512"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "102400"))
MIN_IMAGE_DIMENSION = int(os.getenv("MIN_IMAGE_DIMENSION", "16"))

# Hardware device settings (auto | cuda | cpu)
AI_DEVICE = os.getenv("AI_DEVICE", "auto").lower().strip()
CUDA_DEVICE_INDEX = int(os.getenv("CUDA_DEVICE_INDEX", "0"))
MAX_IMAGE_PIXELS_GPU = int(os.getenv("MAX_IMAGE_PIXELS_GPU", "102400"))

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Automated file cleanup settings
CLEANUP_INTERVAL_MINUTES = int(os.getenv("CLEANUP_INTERVAL_MINUTES", "60"))
CLEANUP_MAX_AGE_HOURS = int(os.getenv("CLEANUP_MAX_AGE_HOURS", "24"))

# CORS settings for frontend development
CORS_ORIGINS: List[str] = json.loads(os.getenv("CORS_ORIGINS", "null")) or [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
