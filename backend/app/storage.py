import io
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageOps

from backend.app.config import (
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_SIZE_BYTES,
    MIN_IMAGE_DIMENSION,
    OUTPUTS_DIR,
    SUPPORTED_IMAGE_EXTS,
    UPLOADS_DIR,
)

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def is_valid_uuid(val: str) -> bool:
    """Check if string is a valid UUIDv4."""
    return bool(UUID_PATTERN.match(val))


async def validate_and_read_image(
    file: UploadFile, max_pixels: int = MAX_IMAGE_PIXELS
) -> Tuple[np.ndarray, int, int]:
    """Validate uploaded image file format, size, dimensions and read as RGB uint8 array.

    Args:
        file: UploadFile from FastAPI request.

    Returns:
        Tuple of (img_rgb_array, width, height).

    Raises:
        HTTPException: If file is too large, corrupt, invalid format, or dimensions exceed limits.
    """
    # Read at most one byte beyond the limit; never buffer an unbounded upload.
    try:
        content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    finally:
        await file.close()
    file_size = len(content)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File upload rỗng (0 bytes). Vui lòng chọn một file ảnh hợp lệ.",
        )

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Dung lượng file vượt quá giới hạn cho phép ({max_mb} MB).",
        )

    extension = Path(file.filename or "").suffix.lower()
    if extension not in SUPPORTED_IMAGE_EXTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Định dạng ảnh không được hỗ trợ. Chọn PNG, JPG, WEBP hoặc BMP.",
        )

    expected_format = {
        ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG",
        ".webp": "WEBP", ".bmp": "BMP",
    }[extension]

    # Inspect the header and reject oversized images before full pixel decoding.
    try:
        with Image.open(io.BytesIO(content)) as pil_img:
            if pil_img.format != expected_format:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Định dạng nội dung ảnh không khớp phần mở rộng file.",
                )
            width, height = pil_img.size
            if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Kích thước ảnh quá nhỏ ({width}x{height}); mỗi cạnh cần ít nhất {MIN_IMAGE_DIMENSION}px.",
                )
            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Kích thước cạnh ảnh ({width}x{height}) vượt giới hạn {MAX_IMAGE_DIMENSION}px.",
                )
            if width * height > max_pixels:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tổng số pixel ({width * height:,}) vượt quá ngưỡng xử lý an toàn ({max_pixels:,} pixels).",
                )
            pil_img.verify()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tệp tải lên không phải là định dạng hình ảnh hợp lệ hoặc dữ liệu bị lỗi.",
        )

    # Re-open after verify(), then handle EXIF orientation and RGB conversion.
    try:
        with Image.open(io.BytesIO(content)) as pil_img:
            transposed = ImageOps.exif_transpose(pil_img)
            # Handle alpha channel cleanly (RGBA, LA, or palette with transparency)
            # by compositing onto a neutral white background instead of harsh black clipping
            if transposed.mode in ("RGBA", "LA") or (transposed.mode == "P" and "transparency" in transposed.info):
                rgba = transposed.convert("RGBA")
                bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                composite = Image.alpha_composite(bg, rgba)
                converted = composite.convert("RGB")
            else:
                converted = transposed.convert("RGB")
            width, height = converted.size
            img_rgb = np.asarray(converted, dtype=np.uint8).copy()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể giải mã hoặc chuyển đổi màu sắc của ảnh.",
        )

    return img_rgb, width, height


def save_output_image(img_rgb: np.ndarray, result_id: str) -> Path:
    """Save super-resolved image to outputs directory.

    Args:
        img_rgb: Numpy RGB array of shape [H, W, 3].
        result_id: Valid UUID string.

    Returns:
        Path to saved output image.
    """
    if not is_valid_uuid(result_id):
        raise ValueError("result_id must be a UUIDv4.")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"{result_id}.png"

    # Convert RGB to BGR for OpenCV imwrite
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(out_path), bgr)
    if not success:
        raise IOError(f"Không thể ghi file kết quả vào '{out_path}'.")

    return out_path


def get_result_path(result_id: str) -> Optional[Path]:
    """Retrieve output image file path by result_id safely, preventing directory traversal.

    Args:
        result_id: Unique identifier for the result.

    Returns:
        Path if file exists and is within outputs dir, else None.
    """
    # Strict UUID validation eliminates path traversal characters (/, \, .., etc.)
    if not is_valid_uuid(result_id):
        return None

    target = (OUTPUTS_DIR / f"{result_id}.png").resolve()
    base_dir = OUTPUTS_DIR.resolve()

    # Double check path containment
    if not target.is_relative_to(base_dir):
        return None

    if not target.is_file():
        return None

    return target


def cleanup_old_files(max_age_hours: int = 24) -> int:
    """Remove temporary files older than max_age_hours from uploads and outputs.

    Returns:
        Number of removed files.
    """
    removed_count = 0
    now = time.time()
    max_age_seconds = max_age_hours * 3600

    # 1. Clean generated UUID result PNGs from OUTPUTS_DIR; keep .gitkeep and experiments.
    if OUTPUTS_DIR.is_dir():
        for item in OUTPUTS_DIR.glob("*.png"):
            if not is_valid_uuid(item.stem):
                continue
            try:
                if now - item.stat().st_mtime > max_age_seconds:
                    item.unlink()
                    removed_count += 1
            except OSError:
                pass

    # 2. Clean temporary files from UPLOADS_DIR; keep .gitkeep
    if UPLOADS_DIR.is_dir():
        for item in UPLOADS_DIR.iterdir():
            if item.is_file() and item.name != ".gitkeep":
                try:
                    if now - item.stat().st_mtime > max_age_seconds:
                        item.unlink()
                        removed_count += 1
                except OSError:
                    pass

    return removed_count
