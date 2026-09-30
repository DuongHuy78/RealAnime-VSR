import logging
import subprocess
from pathlib import Path
from typing import Optional, Union

from backend.app.config import FFMPEG_BIN
from backend.app.video.probe import VideoProbeResult, probe_video

logger = logging.getLogger(__name__)


class VideoEncodeError(Exception):
    """Custom exception raised when video encoding or audio muxing fails."""
    pass


def encode_video_from_frames(
    frames_dir: Union[str, Path],
    output_video_path: Union[str, Path],
    fps_str: str,
    source_video_path: Optional[Union[str, Path]] = None,
    crf: int = 18,
    preset: str = "medium",
) -> VideoProbeResult:
    """Encode super-resolved frame PNG sequence to MP4 H.264 video with optional audio muxing.

    Key considerations:
    - Uses '-framerate <fps_str>' on image2 demuxer to preserve exact CFR timestamps.
    - Uses '-pix_fmt yuv420p' for universal browser/HTML5 video compatibility.
    - Uses '-movflags +faststart' to enable progressive streaming in web players.
    - Muxes audio from source_video_path if available via AAC re-encode (-c:a aac -b:a 192k).
    - Uses '-shortest' to avoid trailing black freeze frames if audio/video timestamps differ slightly.
    - Validates output file using probe_video.

    Args:
        frames_dir: Directory containing 'sr_frame_%06d.png' files.
        output_video_path: Destination path for final MP4 video file.
        fps_str: Rational or numeric FPS string (e.g. "30000/1001", "24/1", "30").
        source_video_path: Optional original video path to extract and mux audio.
        crf: Constant Rate Factor for x264 (default 18 for visually lossless quality).
        preset: x264 encoding preset (default "medium").

    Returns:
        VideoProbeResult: Probe metadata of the verified output MP4 file.

    Raises:
        VideoEncodeError: If FFmpeg fails or output video cannot be verified.
    """
    f_dir = Path(frames_dir).resolve()
    out_path = Path(output_video_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_pattern = str(f_dir / "sr_frame_%06d.png")
    frames_found = list(f_dir.glob("sr_frame_*.png"))
    if not frames_found:
        raise VideoEncodeError(f"Không tìm thấy khung hình siêu phân giải nào trong '{f_dir}'.")

    cmd = [
        FFMPEG_BIN,
        "-v", "error",
        "-y",
        "-framerate", str(fps_str),
        "-i", frame_pattern,
    ]

    has_source_audio = False
    if source_video_path is not None:
        src_path = Path(source_video_path).resolve()
        if src_path.is_file():
            # Check whether source video actually contains audio
            try:
                src_probe = probe_video(src_path, enforce_limits=False)
                has_source_audio = src_probe.has_audio
            except Exception as exc:
                logger.warning("Không thể probe audio từ source video '%s': %exc", src_path, exc)
                has_source_audio = False

            if has_source_audio:
                cmd.extend(["-i", str(src_path)])

    if has_source_audio:
        # Map video from PNG sequence (0:v:0) and audio from source (1:a:0)
        cmd.extend([
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-preset", str(preset),
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(out_path),
        ])
    else:
        # Video-only encoding
        cmd.extend([
            "-map", "0:v:0",
            "-c:v", "libx264",
            "-preset", str(preset),
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ])

    logger.info("Executing FFmpeg encode command: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            check=False,
            timeout=300,
        )
    except FileNotFoundError as exc:
        raise VideoEncodeError(f"Không tìm thấy công cụ ffmpeg tại '{FFMPEG_BIN}'.") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoEncodeError("Quá thời gian chờ khi mã hóa video MP4 (FFmpeg timeout).") from exc

    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or "FFmpeg encoding failed"
        raise VideoEncodeError(f"Lỗi khi mã hóa video MP4: {err_msg}")

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise VideoEncodeError(f"Tệp video đầu ra rỗng hoặc không tồn tại tại '{out_path}'.")

    # Validate output with probe_video (limits off since output dimensions are 4x)
    try:
        output_probe = probe_video(out_path, enforce_limits=False)
    except Exception as exc:
        raise VideoEncodeError(f"Video đầu ra không hợp lệ hoặc bị lỗi sau khi mã hóa: {exc}") from exc

    logger.info(
        "Mã hóa thành công: %dx%d, %.2fs, %d frames, audio=%s, file_size=%.2f MB",
        output_probe.width,
        output_probe.height,
        output_probe.duration_seconds,
        output_probe.total_frames,
        output_probe.has_audio,
        output_probe.file_size_bytes / (1024 * 1024),
    )

    return output_probe
