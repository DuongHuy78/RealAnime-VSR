import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional

from backend.app.config import (
    FFPROBE_BIN,
    MAX_VIDEO_DURATION_SECONDS,
    MAX_VIDEO_FRAMES,
    MAX_VIDEO_INPUT_PIXELS,
    MAX_VIDEO_UPLOAD_MB,
)

logger = logging.getLogger(__name__)


class VideoProbeError(Exception):
    """Custom exception raised when video file validation or probing fails."""
    pass


@dataclass
class VideoProbeResult:
    width: int
    height: int
    duration_seconds: float
    fps: str  # Rational string, e.g. "30000/1001" or "24/1"
    fps_float: float
    total_frames: int
    video_codec: str
    has_audio: bool
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None
    file_size_bytes: int = 0

    def to_dict(self):
        return asdict(self)


def parse_fractional_fps(fps_str: str) -> Fraction:
    try:
        parts = fps_str.split("/")
        if len(parts) == 2:
            num, den = int(parts[0]), int(parts[1])
            if den == 0:
                raise ValueError("Zero denominator in FPS fraction")
            return Fraction(num, den)
        return Fraction(float(fps_str)).limit_denominator(1001)
    except Exception as exc:
        raise VideoProbeError(f"Không thể đọc thông số FPS từ '{fps_str}': {exc}") from exc


def probe_video(file_path: Path, enforce_limits: bool = True) -> VideoProbeResult:
    """Probe video file using ffprobe and validate compliance with MVP requirements.

    Args:
        file_path: Path to local video file.
        enforce_limits: Whether to check max duration, pixels, frames, and file size.

    Returns:
        VideoProbeResult: Detailed metadata of validated video.

    Raises:
        VideoProbeError: If file is invalid, corrupt, unsupported codec, VFR, or exceeds limits.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise VideoProbeError(f"Tệp video không tồn tại tại '{path}'.")

    file_size = path.stat().st_size
    if file_size == 0:
        raise VideoProbeError("Tệp video rỗng (0 bytes).")

    if enforce_limits and file_size > MAX_VIDEO_UPLOAD_MB * 1024 * 1024:
        raise VideoProbeError(f"Dung lượng video vượt quá giới hạn cho phép ({MAX_VIDEO_UPLOAD_MB} MB).")

    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-of", "json",
        str(path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            check=False,
            timeout=15,
        )
    except FileNotFoundError as exc:
        raise VideoProbeError(f"Không tìm thấy công cụ ffprobe tại '{FFPROBE_BIN}'. Vui lòng kiểm tra cài đặt FFmpeg.") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoProbeError("Quá thời gian chờ khi phân tích tệp video (ffprobe timeout).") from exc

    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or "ffprobe execution failed"
        raise VideoProbeError(f"Lỗi khi đọc tệp video: {err_msg}")

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise VideoProbeError(f"Không thể phân tích dữ liệu JSON từ ffprobe: {exc}") from exc

    format_info = data.get("format", {})
    streams = data.get("streams", [])

    # Validate container format (must be MP4 / QuickTime family)
    format_name = format_info.get("format_name", "").lower()
    if not any(f in format_name for f in ("mp4", "mov")):
        raise VideoProbeError(f"Định dạng container không được hỗ trợ ('{format_name}'). Vui lòng chọn tệp MP4.")

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        raise VideoProbeError("Tệp không chứa video stream nào.")

    if len(video_streams) > 1:
        raise VideoProbeError("Tệp chứa nhiều hơn 1 video stream. Bản MVP chỉ hỗ trợ 1 video stream.")

    v_stream = video_streams[0]

    # Validate video codec: must be H.264 / AVC
    codec_name = v_stream.get("codec_name", "").lower()
    if codec_name not in ("h264", "avc"):
        raise VideoProbeError(
            f"Codec video '{codec_name}' không được hỗ trợ. Bản MVP chỉ hỗ trợ video H.264 (AVC)."
        )

    # Validate rotation metadata (must be 0 or absent)
    # Check stream tags
    tags = v_stream.get("tags", {})
    rotate_tag = tags.get("rotate")
    if rotate_tag and str(rotate_tag) not in ("0", ""):
        raise VideoProbeError(f"Video chứa cờ xoay (rotate={rotate_tag}). Bản MVP chỉ hỗ trợ video không xoay.")

    # Check side_data for display matrix rotation
    for side in v_stream.get("side_data_list", []):
        rotation = side.get("rotation")
        if rotation is not None and int(rotation) != 0:
            raise VideoProbeError(f"Video chứa ma trận xoay (rotation={rotation}). Bản MVP chỉ hỗ trợ video không xoay.")

    # Parse dimensions
    width = int(v_stream.get("width", 0))
    height = int(v_stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise VideoProbeError("Không thể xác định kích thước khung hình video.")

    # Parse duration
    try:
        duration = float(v_stream.get("duration") or format_info.get("duration") or 0.0)
    except (ValueError, TypeError):
        duration = 0.0

    if duration <= 0.0:
        raise VideoProbeError("Không thể xác định thời lượng video hoặc video có thời lượng 0s.")

    # Parse & Validate Frame Rate (CFR validation)
    r_frame_rate = v_stream.get("r_frame_rate", "")
    avg_frame_rate = v_stream.get("avg_frame_rate", "")

    frac_r = parse_fractional_fps(r_frame_rate)
    frac_avg = parse_fractional_fps(avg_frame_rate)

    float_r = float(frac_r)
    float_avg = float(frac_avg)

    # CFR tolerance check: r_frame_rate and avg_frame_rate must be close (diff < 0.05 fps)
    if abs(float_r - float_avg) > 0.05:
        raise VideoProbeError(
            f"Phát hiện video có tốc độ khung hình biến thiên (VFR: {float_r:.2f} vs {float_avg:.2f} fps). "
            "Bản MVP chỉ hỗ trợ video chuẩn Constant Frame Rate (CFR)."
        )

    fps_frac = frac_avg if frac_avg.denominator != 0 else frac_r
    fps_str = f"{fps_frac.numerator}/{fps_frac.denominator}" if fps_frac.denominator != 1 else str(fps_frac.numerator)
    fps_float = float(fps_frac)

    # Calculate or parse total frames
    nb_frames_str = v_stream.get("nb_frames")
    if nb_frames_str and nb_frames_str.isdigit() and int(nb_frames_str) > 0:
        total_frames = int(nb_frames_str)
    else:
        total_frames = max(1, round(duration * fps_float))

    # Parse audio stream if present
    has_audio = False
    audio_codec = None
    audio_sample_rate = None
    audio_channels = None

    if audio_streams:
        a_stream = audio_streams[0]
        has_audio = True
        audio_codec = a_stream.get("codec_name")
        audio_sample_rate = int(a_stream.get("sample_rate", 0)) if a_stream.get("sample_rate") else None
        audio_channels = int(a_stream.get("channels", 0)) if a_stream.get("channels") else None

    # Enforce configured limits
    if enforce_limits:
        if duration > MAX_VIDEO_DURATION_SECONDS:
            raise VideoProbeError(
                f"Thời lượng video ({duration:.1f}s) vượt quá giới hạn an toàn ({MAX_VIDEO_DURATION_SECONDS}s)."
            )
        total_pixels = width * height
        if total_pixels > MAX_VIDEO_INPUT_PIXELS:
            raise VideoProbeError(
                f"Kích thước video ({width}x{height} = {total_pixels:,} px) vượt quá giới hạn an toàn ({MAX_VIDEO_INPUT_PIXELS:,} px)."
            )
        if total_frames > MAX_VIDEO_FRAMES:
            raise VideoProbeError(
                f"Số lượng khung hình ({total_frames}) vượt quá giới hạn an toàn ({MAX_VIDEO_FRAMES} frames)."
            )

    return VideoProbeResult(
        width=width,
        height=height,
        duration_seconds=round(duration, 3),
        fps=fps_str,
        fps_float=round(fps_float, 3),
        total_frames=total_frames,
        video_codec=codec_name,
        has_audio=has_audio,
        audio_codec=audio_codec,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
        file_size_bytes=file_size,
    )
