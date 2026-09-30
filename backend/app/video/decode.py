import logging
import subprocess
from pathlib import Path
from typing import List, Optional

from backend.app.config import FFMPEG_BIN

logger = logging.getLogger(__name__)


class VideoDecodeError(Exception):
    """Custom exception raised when video frame extraction fails."""
    pass


def decode_video_frames(
    input_video_path: Path,
    output_frames_dir: Path,
    expected_frames: Optional[int] = None,
) -> int:
    """Extract all video frames into sequentially numbered PNG files.

    Uses '-fps_mode passthrough' to prevent FFmpeg from duplicating or dropping frames,
    preserving exact CFR timestamps.

    Args:
        input_video_path: Absolute path to the source MP4 video.
        output_frames_dir: Directory where 'frame_%06d.png' images will be saved.
        expected_frames: Expected frame count from probe (for sanity check).

    Returns:
        int: Number of frames successfully extracted.

    Raises:
        VideoDecodeError: If FFmpeg fails or no frames are extracted.
    """
    video_path = Path(input_video_path).resolve()
    frames_dir = Path(output_frames_dir).resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_pattern = str(frames_dir / "frame_%06d.png")

    cmd = [
        FFMPEG_BIN,
        "-v", "error",
        "-y",
        "-i", str(video_path),
        "-fps_mode", "passthrough",
        "-vsync", "0",
        frame_pattern,
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            check=False,
            timeout=120,
        )
    except FileNotFoundError as exc:
        raise VideoDecodeError(f"Không tìm thấy công cụ ffmpeg tại '{FFMPEG_BIN}'.") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoDecodeError("Quá thời gian chờ khi giải mã khung hình video (FFmpeg timeout).") from exc

    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or "FFmpeg frame extraction failed"
        raise VideoDecodeError(f"Lỗi khi giải mã khung hình video: {err_msg}")

    # Count extracted PNG files
    extracted_files = sorted(frames_dir.glob("frame_*.png"))
    count = len(extracted_files)

    if count == 0:
        raise VideoDecodeError("Không có khung hình nào được giải mã từ video.")

    logger.info("Successfully decoded %d frames to '%s'", count, frames_dir)
    return count
