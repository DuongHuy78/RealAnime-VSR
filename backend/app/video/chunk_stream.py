import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from backend.app.inference.realbasicvsr_pipeline import (
    pad_lq,
    read_image_rgb,
    save_image_rgb,
)
from backend.app.models.realbasicvsr_net import RealBasicVSRNet

logger = logging.getLogger(__name__)


class VideoChunkStreamError(Exception):
    """Custom exception raised during sliding window chunk inference."""
    pass


def compute_chunk_windows(
    total_frames: int,
    chunk_size: int = 7,
    overlap: int = 2,
) -> List[Tuple[int, int]]:
    """Compute start and end frame indices for sliding window inference.

    Args:
        total_frames: Total number of frames in sequence.
        chunk_size: Number of frames per chunk.
        overlap: Number of overlapping frames between consecutive chunks.

    Returns:
        List of (start, end) tuples covering [0, total_frames).
    """
    if total_frames <= 0:
        return []

    if chunk_size <= overlap:
        raise VideoChunkStreamError(
            f"chunk_size ({chunk_size}) must be strictly greater than overlap ({overlap})."
        )

    if total_frames <= chunk_size:
        return [(0, total_frames)]

    stride = max(1, chunk_size - overlap)
    windows: List[Tuple[int, int]] = []

    for start in range(0, total_frames, stride):
        end = min(total_frames, start + chunk_size)
        windows.append((start, end))
        if end == total_frames:
            break

    return windows


@torch.inference_mode()
def stream_rb_video_chunks(
    net: RealBasicVSRNet,
    input_frames_dir: Union[str, Path],
    output_frames_dir: Union[str, Path],
    total_frames: Optional[int] = None,
    chunk_size: int = 7,
    overlap: int = 2,
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> int:
    """Run RealBasicVSR super-resolution with bounded-memory streaming chunk buffer.

    Instead of accumulating all 4x output frames in RAM (which causes OOM on videos
    longer than a few seconds), this function uses a sliding window buffer.
    Frames that will receive no further overlap are finalized, saved to disk, and
    immediately purged from memory. Peak memory is strictly bounded by O(chunk_size * H * W).

    Args:
        net: Instantiated RealBasicVSRNet model.
        input_frames_dir: Directory containing extracted source frames (PNG).
        output_frames_dir: Directory to save super-resolved 'sr_frame_%06d.png'.
        total_frames: Expected total frame count. If None, derived from files on disk.
        chunk_size: Number of frames per temporal chunk (default 7).
        overlap: Overlap between adjacent chunks for temporal consistency (default 2).
        device: PyTorch device to execute on. If None, uses net's device.
        progress_callback: Optional callback func(processed_frames, total_frames, message).

    Returns:
        int: Total number of super-resolved frames written to disk.

    Raises:
        VideoChunkStreamError: If input frames are missing, inconsistent, or inference fails.
    """
    in_dir = Path(input_frames_dir).resolve()
    out_dir = Path(output_frames_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(in_dir.glob("frame_*.png"))
    n = len(input_files)

    if n == 0:
        raise VideoChunkStreamError(f"Không tìm thấy khung hình đầu vào nào trong '{in_dir}'.")

    if total_frames is not None and total_frames != n:
        logger.warning(
            "Số khung hình tìm thấy (%d) khác với kỳ vọng (%d). Sẽ sử dụng %d khung hình.",
            n, total_frames, n
        )

    if device is None:
        device = next(net.parameters()).device

    windows = compute_chunk_windows(total_frames=n, chunk_size=chunk_size, overlap=overlap)
    total_chunks = len(windows)
    logger.info(
        "Bắt đầu xử lý video %d khung hình qua %d chunks (chunk_size=%d, overlap=%d, device=%s)",
        n, total_chunks, chunk_size, overlap, device
    )

    accum_dict: Dict[int, np.ndarray] = {}
    count_dict: Dict[int, int] = {}
    written_frames = 0

    for chunk_idx, (start, end) in enumerate(windows):
        chunk_len = end - start
        is_last_chunk = (chunk_idx == total_chunks - 1)
        next_start = windows[chunk_idx + 1][0] if not is_last_chunk else n

        logger.debug(
            "Processing chunk %d/%d: frames [%d, %d)",
            chunk_idx + 1, total_chunks, start, end
        )

        # 1. Load input frames for current chunk
        chunk_imgs: List[np.ndarray] = []
        for i in range(start, end):
            try:
                img = read_image_rgb(input_files[i])
                chunk_imgs.append(img)
            except Exception as exc:
                raise VideoChunkStreamError(
                    f"Lỗi khi đọc khung hình {i+1} ('{input_files[i].name}'): {exc}"
                ) from exc

        # 2. Normalize and prepare tensor [1, T, 3, H, W]
        arr = np.stack(chunk_imgs).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to(device)

        # 3. Model forward pass with padding
        x_pad, (orig_h, orig_w) = pad_lq(x)
        sr = net(x_pad).clamp(0.0, 1.0)
        sr = sr[:, :, :, : orig_h * 4, : orig_w * 4]

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Convert chunk result to numpy uint8
        sr_np = sr[0].permute(0, 2, 3, 1).cpu().numpy()
        sr_uint8 = np.clip(sr_np * 255.0, 0.0, 255.0).round().astype(np.uint8)

        # Explicitly free PyTorch GPU tensors
        del x, x_pad, sr
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 4. Accumulate into memory buffer
        for j in range(chunk_len):
            frame_i = start + j
            if frame_i not in accum_dict:
                accum_dict[frame_i] = sr_uint8[j].astype(np.float32)
                count_dict[frame_i] = 1
            else:
                accum_dict[frame_i] += sr_uint8[j].astype(np.float32)
                count_dict[frame_i] += 1

        # 5. Flush and write frames that will receive no further overlaps
        flush_limit = next_start if not is_last_chunk else n

        for frame_to_save in range(written_frames, flush_limit):
            if frame_to_save not in accum_dict:
                raise VideoChunkStreamError(
                    f"Khung hình {frame_to_save + 1} bị thiếu trong bộ đệm kết quả."
                )

            avg_frame = accum_dict[frame_to_save] / max(count_dict[frame_to_save], 1)
            final_uint8 = np.clip(avg_frame, 0.0, 255.0).round().astype(np.uint8)

            out_filename = f"sr_frame_{frame_to_save + 1:06d}.png"
            out_filepath = out_dir / out_filename
            save_image_rgb(out_filepath, final_uint8)

            # Purge from memory immediately to keep memory bounded
            del accum_dict[frame_to_save]
            del count_dict[frame_to_save]
            written_frames += 1

        if progress_callback:
            percent_msg = f"Đã xử lý {written_frames}/{n} khung hình (Chunk {chunk_idx + 1}/{total_chunks})"
            progress_callback(written_frames, n, percent_msg)

    if written_frames != n:
        raise VideoChunkStreamError(
            f"Số lượng khung hình đã ghi ({written_frames}) không khớp tổng số khung hình ({n})."
        )

    logger.info("Hoàn tất xử lý video: %d khung hình siêu phân giải x4 đã lưu tại '%s'", written_frames, out_dir)
    return written_frames
