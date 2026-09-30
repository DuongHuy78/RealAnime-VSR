import asyncio
import threading
from typing import Dict, List, Optional, Tuple

import torch

from backend.app.config import AI_DEVICE, CHECKPOINT_PATH, CUDA_DEVICE_INDEX
from backend.app.inference.realbasicvsr_pipeline import (
    get_default_device,
    load_realbasicvsr,
)
from backend.app.models.realbasicvsr_net import RealBasicVSRNet
from backend.app.schemas import ModelItem

# In-memory singleton cache for models
_MODEL_CACHE: Dict[str, RealBasicVSRNet] = {}
_ACTIVE_DEVICE: Optional[torch.device] = None

# Concurrency lock to prevent concurrent CPU/GPU inference overload
# Serializes requests safely
_INFERENCE_SEMAPHORE: Optional[asyncio.Semaphore] = None
_INFERENCE_LOCK = threading.Lock()


def get_inference_semaphore() -> asyncio.Semaphore:
    """Get or create singleton asyncio Semaphore for inference concurrency control."""
    global _INFERENCE_SEMAPHORE
    if _INFERENCE_SEMAPHORE is None:
        _INFERENCE_SEMAPHORE = asyncio.Semaphore(1)
    return _INFERENCE_SEMAPHORE


def get_inference_lock() -> threading.Lock:
    """Keep the model serialized even if an awaiting HTTP task is cancelled."""
    return _INFERENCE_LOCK


def resolve_preferred_device() -> Tuple[torch.device, str]:
    """Resolve target execution device according to AI_DEVICE setting and hardware availability."""
    pref = AI_DEVICE.lower().strip()
    if pref not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"AI_DEVICE='{AI_DEVICE}' không hợp lệ; chọn auto, cuda hoặc cpu.")
    if pref in ("cuda", "auto"):
        if torch.cuda.is_available():
            if not 0 <= CUDA_DEVICE_INDEX < torch.cuda.device_count():
                if pref == "cuda":
                    raise RuntimeError(f"CUDA_DEVICE_INDEX={CUDA_DEVICE_INDEX} không tồn tại.")
                print(f"[ModelRegistry] [WARN] CUDA_DEVICE_INDEX={CUDA_DEVICE_INDEX} không tồn tại; dùng CPU.")
                return torch.device("cpu"), "cpu"
            dev = torch.device(f"cuda:{CUDA_DEVICE_INDEX}")
            return dev, "cuda"
        elif pref == "cuda":
            raise RuntimeError("AI_DEVICE='cuda' được yêu cầu nhưng CUDA không khả dụng trên hệ thống.")
    return torch.device("cpu"), "cpu"


def load_model_registry() -> Tuple[int, str]:
    """Load and cache supported checkpoints into memory on server startup with GPU warmup & fallback.

    Returns:
        Tuple of (number_of_models_loaded, device_str).
    """
    global _MODEL_CACHE, _ACTIVE_DEVICE, _INFERENCE_SEMAPHORE

    _MODEL_CACHE.clear()
    _INFERENCE_SEMAPHORE = asyncio.Semaphore(1)

    model_id = "realbasicvsr_anime_v3"

    if not CHECKPOINT_PATH.is_file():
        print(f"[ModelRegistry] [WARN] Checkpoint not found at '{CHECKPOINT_PATH}'.")
        _ACTIVE_DEVICE = torch.device("cpu")
        return 0, str(_ACTIVE_DEVICE)

    target_dev, dev_type = resolve_preferred_device()

    try:
        print(f"[ModelRegistry] Loading '{model_id}' from {CHECKPOINT_PATH} to {target_dev} ({dev_type.upper()})...")
        net, dev = load_realbasicvsr(CHECKPOINT_PATH, device=target_dev)

        # T=2 exercises optical flow and grid_sample as well as the image path.
        if dev.type == "cuda":
            dummy_x = torch.zeros(1, 2, 3, 32, 32, device=dev, dtype=torch.float32)
            with torch.inference_mode():
                _ = net(dummy_x)
                torch.cuda.synchronize(dev)
            print(f"[ModelRegistry] [CUDA Warmup OK] {torch.cuda.get_device_name(dev)} is fully operational!")

        _MODEL_CACHE[model_id] = net
        _ACTIVE_DEVICE = dev
        print(f"[ModelRegistry] [OK] Successfully loaded and cached '{model_id}' on {dev}!")
    except Exception as e:
        print(f"[ModelRegistry] [FAIL] Failed to initialize on {target_dev}: {e}")
        # Automatic CPU fallback if AI_DEVICE is 'auto' and CUDA failed
        if target_dev.type == "cuda" and AI_DEVICE == "auto":
            print(f"[ModelRegistry] [FALLBACK] Automatically falling back to CPU...")
            try:
                cpu_dev = torch.device("cpu")
                net, dev = load_realbasicvsr(CHECKPOINT_PATH, device=cpu_dev)
                _MODEL_CACHE[model_id] = net
                _ACTIVE_DEVICE = dev
                print(f"[ModelRegistry] [OK] Successfully loaded '{model_id}' on fallback CPU!")
            except Exception as cpu_err:
                print(f"[ModelRegistry] [FAIL] CPU fallback also failed: {cpu_err}")
                _ACTIVE_DEVICE = torch.device("cpu")
        else:
            _ACTIVE_DEVICE = target_dev

    return len(_MODEL_CACHE), str(_ACTIVE_DEVICE)


def get_model(model_id: str) -> Optional[Tuple[RealBasicVSRNet, torch.device]]:
    """Retrieve cached model instance and device by model ID.

    Returns:
        Tuple of (model, device) or None if model_id is not loaded.
    """
    global _MODEL_CACHE, _ACTIVE_DEVICE
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id], _ACTIVE_DEVICE or get_default_device()
    return None


def get_active_model_ids() -> List[str]:
    """List of currently loaded model IDs."""
    return list(_MODEL_CACHE.keys())


def list_available_models() -> List[ModelItem]:
    """Return catalog of models and their availability."""
    global _MODEL_CACHE, _ACTIVE_DEVICE
    is_rb_available = "realbasicvsr_anime_v3" in _MODEL_CACHE
    dev_str = str(_ACTIVE_DEVICE) if is_rb_available else "none"

    return [
        ModelItem(
            id="realbasicvsr_anime_v3",
            name="RealBasicVSR Anime V3",
            description="Siêu phân giải x4 chuyên biệt cho nét vẽ và màu sắc anime (Single Frame Mode).",
            scale=4,
            available=is_rb_available,
            device=dev_str,
        )
    ]
