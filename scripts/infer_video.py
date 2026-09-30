"""CLI tool for RealBasicVSR video super-resolution inference and profiling.

Supports:
- Running video super-resolution on any local MP4 video file.
- Generating synthetic test videos (with or without audio).
- Cropping a quick 2-second test segment from the sample anime asset.
- Profiling decode, inference, and encode stages with peak VRAM and throughput (FPS).
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from backend.app.config import (
    CHECKPOINT_PATH,
    FFMPEG_BIN,
    VIDEO_OUTPUTS_DIR,
    WORK_DIR,
)
from backend.app.inference.realbasicvsr_pipeline import (
    get_default_device,
    get_gpu_memory_stats,
    load_realbasicvsr,
)
from backend.app.video.chunk_stream import stream_rb_video_chunks
from backend.app.video.decode import decode_video_frames
from backend.app.video.encode import encode_video_from_frames
from backend.app.video.probe import probe_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("infer_video_cli")


def generate_synthetic_video(
    output_path: Path,
    duration: float = 2.0,
    fps: int = 30,
    width: int = 160,
    height: int = 120,
    include_audio: bool = True,
) -> Path:
    """Generate a synthetic CFR MP4 test video with test patterns and optional audio tone."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = int(duration * fps)
    filter_expr = f"testsrc=duration={duration}:size={width}x{height}:rate={fps}"

    cmd = [
        FFMPEG_BIN,
        "-v", "error",
        "-y",
        "-f", "lavfi",
        "-i", filter_expr,
    ]

    if include_audio:
        # Generate 440Hz sine wave tone
        cmd.extend([
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            str(output_path),
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ])

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg synthetic video generation failed: {proc.stderr}")

    logger.info("Generated synthetic video at '%s' (%dx%d, %.1fs, %d fps, audio=%s)",
                output_path, width, height, duration, fps, include_audio)
    return output_path


def crop_sample_video(
    source_path: Path,
    output_path: Path,
    duration: float = 2.0,
) -> Path:
    """Crop a short initial segment from sample anime asset for testing."""
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_BIN,
        "-v", "error",
        "-y",
        "-ss", "00:00:00",
        "-i", str(source_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg crop failed: {proc.stderr}")

    logger.info("Cropped %.1fs sample video to '%s'", duration, output_path)
    return output_path


def run_video_pipeline(
    input_video: Path,
    output_video: Path,
    chunk_size: int = 7,
    overlap: int = 2,
    device_name: Optional[str] = None,
    keep_temp: bool = False,
):
    """Execute complete end-to-end video super-resolution pipeline."""
    input_path = Path(input_video).resolve()
    output_path = Path(output_video).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n========================================================")
    print("      REALBASICVSR VIDEO SUPER-RESOLUTION PIPELINE      ")
    print("========================================================")
    print(f"Input Video : {input_path}")
    print(f"Output Video: {output_path}")

    # Step 1: Probe video
    t0 = time.perf_counter()
    probe = probe_video(input_path, enforce_limits=False)
    probe_time = time.perf_counter() - t0
    print(f"Source Specs: {probe.width}x{probe.height}, {probe.duration_seconds:.2f}s, "
          f"{probe.fps} fps ({probe.fps_float:.2f}), {probe.total_frames} frames, audio={probe.has_audio}")

    # Step 2: Prepare Device & Load Model
    if device_name:
        device = torch.device(device_name)
    else:
        device = get_default_device()

    print(f"Target Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    t_load = time.perf_counter()
    net, _ = load_realbasicvsr(checkpoint_path=CHECKPOINT_PATH, device=device)
    model_load_time = time.perf_counter() - t_load
    print(f"Model loaded in {model_load_time:.2f}s")

    # Step 3: Setup temp work directory
    session_id = f"cli_{uuid.uuid4().hex[:8]}"
    job_work_dir = WORK_DIR / session_id
    raw_frames_dir = job_work_dir / "raw_frames"
    sr_frames_dir = job_work_dir / "sr_frames"
    job_work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 4: Decode frames
        print("\n[1/3] Đang giải mã các khung hình video (FFmpeg CFR passthrough)...")
        t_dec = time.perf_counter()
        decoded_count = decode_video_frames(
            input_video_path=input_path,
            output_frames_dir=raw_frames_dir,
            expected_frames=probe.total_frames,
        )
        decode_time = time.perf_counter() - t_dec
        print(f"-> Đã trích xuất {decoded_count} khung hình trong {decode_time:.2f}s ({(decoded_count / max(decode_time, 0.001)):.1f} fps)")

        # Step 5: Streaming Chunk Inference
        print(f"\n[2/3] Đang chạy siêu phân giải AI x4 (chunk_size={chunk_size}, overlap={overlap})...")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def on_progress(done: int, total: int, msg: str):
            sys.stdout.write(f"\r   [{done}/{total}] {msg}")
            sys.stdout.flush()

        t_inf = time.perf_counter()
        sr_count = stream_rb_video_chunks(
            net=net,
            input_frames_dir=raw_frames_dir,
            output_frames_dir=sr_frames_dir,
            total_frames=decoded_count,
            chunk_size=chunk_size,
            overlap=overlap,
            device=device,
            progress_callback=on_progress,
        )
        inference_time = time.perf_counter() - t_inf
        print()  # newline after progress bar
        print(f"-> Hoàn tất inference {sr_count} khung hình trong {inference_time:.2f}s ({(sr_count / max(inference_time, 0.001)):.2f} FPS)")

        if device.type == "cuda":
            gpu_stats = get_gpu_memory_stats(device)
            print(f"   Peak VRAM allocated: {gpu_stats.get('allocated_mb', 0)} MB / {gpu_stats.get('total_mb', 0)} MB")

        # Step 6: Encode & Audio Muxing
        print("\n[3/3] Đang mã hóa MP4 (H.264 yuv420p) và ghép âm thanh AAC...")
        t_enc = time.perf_counter()
        out_probe = encode_video_from_frames(
            frames_dir=sr_frames_dir,
            output_video_path=output_path,
            fps_str=probe.fps,
            source_video_path=input_path if probe.has_audio else None,
            crf=18,
            preset="medium",
        )
        encode_time = time.perf_counter() - t_enc
        print(f"-> Mã hóa hoàn tất trong {encode_time:.2f}s")

        total_time = decode_time + inference_time + encode_time

        print("\n========================================================")
        print("                   KẾT QUẢ XỬ LÝ                        ")
        print("========================================================")
        print(f"Kích thước đầu vào : {probe.width}x{probe.height}")
        print(f"Kích thước đầu ra  : {out_probe.width}x{out_probe.height} (Tăng 4x)")
        print(f"Tổng số frame      : {out_probe.total_frames}")
        print(f"Thời lượng         : {out_probe.duration_seconds:.2f}s (FPS: {out_probe.fps_float:.2f})")
        print(f"Âm thanh           : {'Có (AAC)' if out_probe.has_audio else 'Không'}")
        print(f"Dung lượng video   : {out_probe.file_size_bytes / (1024 * 1024):.2f} MB")
        print(f"Tốc độ Inference   : {(sr_count / max(inference_time, 0.001)):.2f} FPS")
        print(f"Tổng thời gian     : {total_time:.2f}s (Decode: {decode_time:.2f}s | Infer: {inference_time:.2f}s | Encode: {encode_time:.2f}s)")
        print(f"Đường dẫn kết quả  : {output_path}")
        print("========================================================\n")

    finally:
        if not keep_temp and job_work_dir.exists():
            shutil.rmtree(job_work_dir, ignore_errors=True)
            logger.info("Cleaned up temp directory '%s'", job_work_dir)


def main():
    parser = argparse.ArgumentParser(description="RealBasicVSR Video Super-Resolution CLI")
    parser.add_argument("--input", "-i", type=str, help="Path to input MP4 video")
    parser.add_argument("--output", "-o", type=str, help="Path to save output MP4 video")
    parser.add_argument("--chunk-size", "-c", type=int, default=7, help="Chunk size (default 7)")
    parser.add_argument("--overlap", type=int, default=2, help="Overlap between chunks (default 2)")
    parser.add_argument("--device", "-d", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test video")
    parser.add_argument("--no-audio", action="store_true", help="Synthetic video without audio")
    parser.add_argument("--crop-sample", action="store_true", help="Crop 2s sample from assets/ anime video")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration for test clips (default 2.0s)")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary frame PNG directories")

    args = parser.parse_args()

    VIDEO_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    input_video_path = None
    output_video_path = None

    if args.synthetic:
        # Generate synthetic video
        test_dir = WORK_DIR / "test_clips"
        test_dir.mkdir(parents=True, exist_ok=True)
        has_audio = not args.no_audio
        suffix = "audio" if has_audio else "no_audio"
        test_input = test_dir / f"synthetic_{suffix}.mp4"
        generate_synthetic_video(
            output_path=test_input,
            duration=args.duration,
            fps=30,
            width=160,
            height=120,
            include_audio=has_audio,
        )
        input_video_path = test_input
        output_video_path = VIDEO_OUTPUTS_DIR / f"synthetic_{suffix}_x4.mp4"

    elif args.crop_sample:
        # Crop sample from assets
        sample_asset = PROJECT_ROOT / "assets" / "ebd9028b-a231-4b6a-b4a3-acdba777e7f5.mp4"
        if not sample_asset.is_file():
            print(f"Lỗi: Không tìm thấy video mẫu tại '{sample_asset}'")
            sys.exit(1)

        test_dir = WORK_DIR / "test_clips"
        test_dir.mkdir(parents=True, exist_ok=True)
        cropped_input = test_dir / "anime_sample_2s.mp4"
        crop_sample_video(sample_asset, cropped_input, duration=args.duration)
        input_video_path = cropped_input
        output_video_path = VIDEO_OUTPUTS_DIR / "anime_sample_2s_x4.mp4"

    elif args.input:
        input_video_path = Path(args.input)
        if args.output:
            output_video_path = Path(args.output)
        else:
            output_video_path = VIDEO_OUTPUTS_DIR / f"{input_video_path.stem}_x4.mp4"

    else:
        parser.print_help()
        sys.exit(0)

    run_video_pipeline(
        input_video=input_video_path,
        output_video=output_video_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        device_name=args.device,
        keep_temp=args.keep_temp,
    )


if __name__ == "__main__":
    main()
