from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from backend.app.models.realbasicvsr_net import RealBasicVSRNet


def get_default_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_realbasicvsr(
    checkpoint_path: Union[str, Path] = "best.pth",
    device: Optional[torch.device] = None,
    mid_channels: int = 64,
    num_blocks: int = 20,
) -> Tuple[RealBasicVSRNet, torch.device]:
    """Load RealBasicVSRNet model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pth).
        device: Device to place the model on. If None, auto-selects CUDA if available, else CPU.
        mid_channels: Number of intermediate channels (default 64).
        num_blocks: Number of residual blocks (default 20).

    Returns:
        Tuple of (model, device).

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
        RuntimeError: If checkpoint loading fails or keys mismatch.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found at '{path.resolve()}'. "
            "Please ensure the valid 'best.pth' checkpoint is present."
        )

    if device is None:
        device = get_default_device()

    net = RealBasicVSRNet(mid_channels=mid_channels, num_blocks=num_blocks)

    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint file '{path}': {e}") from e

    if isinstance(ckpt, dict) and "net" in ckpt:
        state = ckpt["net"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise RuntimeError(f"Unexpected checkpoint format in '{path}'. Expected dict containing 'net'.")

    # Strip 'module.' prefix if model was saved using DataParallel
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    net.load_state_dict(state, strict=True)
    net = net.to(device).eval()

    return net, device


def pad_lq(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad input tensor so that height and width are divisible by 32.

    Args:
        x: Tensor of shape [B, T, C, H, W].

    Returns:
        Tuple of (padded_tensor, (original_H, original_W)).
    """
    b, t, c, h, w = x.shape

    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    if pad_h == 0 and pad_w == 0:
        return x, (h, w)

    flat = x.reshape(b * t, c, h, w)

    # Reflect padding requires input dimensions to be strictly greater than padding amount.
    # Fallback to replicate mode if image dimension is smaller than required padding.
    mode = "reflect" if (h > pad_h and w > pad_w) else "replicate"

    flat = F.pad(flat, (0, pad_w, 0, pad_h), mode=mode)
    padded = flat.reshape(b, t, c, h + pad_h, w + pad_w)

    return padded, (h, w)


def read_image_rgb(path: Union[str, Path]) -> np.ndarray:
    """Read an image file as RGB uint8 array, handling EXIF orientation.

    Args:
        path: Path to image file.

    Returns:
        np.ndarray: Image array of shape [H, W, 3] with dtype uint8 (RGB).
    """
    path_str = str(path)
    # Using Pillow to reliably handle EXIF orientation and multiple formats
    with Image.open(path_str) as img:
        transposed = ImageOps.exif_transpose(img)
        if transposed.mode in ("RGBA", "LA") or (transposed.mode == "P" and "transparency" in transposed.info):
            rgba = transposed.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            composite = Image.alpha_composite(bg, rgba)
            converted = composite.convert("RGB")
        else:
            converted = transposed.convert("RGB")
        return np.array(converted, dtype=np.uint8)


def save_image_rgb(path: Union[str, Path], img_rgb: np.ndarray) -> None:
    """Save an RGB uint8 array to image file.

    Args:
        path: Target file path.
        img_rgb: Image array of shape [H, W, 3] with dtype uint8 (RGB).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite expects BGR order
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(target), bgr)
    if not success:
        raise IOError(f"Failed to write image to '{target}'.")


def get_gpu_memory_stats(device: Optional[torch.device] = None) -> dict:
    """Return dictionary of GPU memory stats in MB if device is CUDA, else empty dict."""
    if not torch.cuda.is_available() or device is None or device.type != "cuda":
        return {}
    allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_mb": round(allocated, 1),
        "reserved_mb": round(reserved, 1),
        "free_mb": round(free / (1024 * 1024), 1),
        "total_mb": round(total / (1024 * 1024), 1),
    }


@torch.inference_mode()
def infer_rb_single(
    net: RealBasicVSRNet,
    image_rgb: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Run RealBasicVSR inference on a single image (T=1).

    Args:
        net: Instantiated RealBasicVSRNet model.
        image_rgb: RGB image numpy array of shape [H, W, 3], uint8.
        device: Device to run on. If None, inferred from model parameters.

    Returns:
        RGB image numpy array of shape [H*4, W*4, 3], uint8.
    """
    if device is None:
        device = next(net.parameters()).device

    h, w, _ = image_rgb.shape

    # Normalize [0, 1] and reshape to [1, 1, 3, H, W]
    arr = image_rgb.astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)

    # Pad dimensions to multiples of 32
    x_pad, (orig_h, orig_w) = pad_lq(x)

    # Forward pass
    sr = net(x_pad).clamp(0.0, 1.0)

    # Crop out padding to exact 4x dimension
    sr = sr[:, :, :, : orig_h * 4, : orig_w * 4]

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Convert back to uint8 numpy array [H*4, W*4, 3]
    sr_np = sr[0, 0].permute(1, 2, 0).cpu().numpy()
    out = np.clip(sr_np * 255.0, 0.0, 255.0).round().astype(np.uint8)

    return out



@torch.inference_mode()
def infer_rb_sequence(
    net: RealBasicVSRNet,
    frames: List[np.ndarray],
    chunk_size: int = 7,
    overlap: int = 2,
    device: Optional[torch.device] = None,
) -> List[np.ndarray]:
    """Run RealBasicVSR inference on a sequence of frames with sliding temporal window.

    Args:
        net: Instantiated RealBasicVSRNet model.
        frames: List of RGB uint8 images of shape [H, W, 3].
        chunk_size: Number of frames per chunk (default 7).
        overlap: Overlap between consecutive chunks (default 2).
        device: Device to run on. If None, inferred from model parameters.

    Returns:
        List of RGB uint8 images of shape [H*4, W*4, 3].
    """
    n = len(frames)
    if n == 0:
        raise ValueError("frames list cannot be empty.")

    if chunk_size <= overlap:
        raise ValueError(f"chunk_size ({chunk_size}) must be greater than overlap ({overlap}).")

    if device is None:
        device = next(net.parameters()).device

    h, w = frames[0].shape[:2]

    # Verify all frames have the same shape
    for idx, f in enumerate(frames):
        if f.shape[:2] != (h, w):
            raise ValueError(
                f"Frame {idx} shape {f.shape[:2]} does not match first frame shape {(h, w)}."
            )

    accum = [np.zeros((h * 4, w * 4, 3), dtype=np.float32) for _ in range(n)]
    count = np.zeros(n, dtype=np.float32)

    stride = max(1, chunk_size - overlap)
    starts = range(0, n, stride)

    for start in starts:
        end = min(n, start + chunk_size)
        current = frames[start:end]

        arr = np.stack(current).astype(np.float32) / 255.0
        # Shape: [1, T, 3, H, W]
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to(device)

        x_pad, (orig_h, orig_w) = pad_lq(x)

        sr = net(x_pad).clamp(0.0, 1.0)
        # Crop out padding
        sr = sr[:, :, :, : orig_h * 4, : orig_w * 4]

        # Convert chunk result to numpy
        sr_np = sr[0].permute(0, 2, 3, 1).cpu().numpy()
        sr_uint8 = np.clip(sr_np * 255.0, 0.0, 255.0).round().astype(np.uint8)

        # Accumulate overlapping frames
        chunk_len = end - start
        for j in range(chunk_len):
            frame_idx = start + j
            accum[frame_idx] += sr_uint8[j].astype(np.float32)
            count[frame_idx] += 1.0

        if end == n:
            break

    output = []
    for i in range(n):
        avg = accum[i] / max(count[i], 1.0)
        output.append(np.clip(avg, 0.0, 255.0).round().astype(np.uint8))

    return output
