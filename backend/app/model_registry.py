import asyncio
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from backend.app.config import AI_DEVICE, CHECKPOINT_PATH, CUDA_DEVICE_INDEX, MODELS_DIR, REPO_ROOT
from backend.app.inference.realbasicvsr_pipeline import load_realbasicvsr
from backend.app.models.realbasicvsr_net import RealBasicVSRNet
from backend.app.schemas import ModelItem

DEFAULT_MODEL_ID = "realbasicvsr_anime_v3"  # Preserve the original API ID.
_MODEL_CACHE: Dict[str, RealBasicVSRNet] = {}
_MODEL_DEVICES: Dict[str, torch.device] = {}
_MODEL_SIGNATURES: Dict[str, Tuple[int, int]] = {}
_MODEL_ERRORS: Dict[str, Tuple[Tuple[int, int], str]] = {}
_CHECKPOINTS: Dict[str, Path] = {}
_DEFAULT_CHECKPOINT: Optional[Path] = None
_ACTIVE_MODEL_ID: Optional[str] = None
_ACTIVE_DEVICE: Optional[torch.device] = None
_INFERENCE_SEMAPHORE: Optional[asyncio.Semaphore] = None
# Activation and inference share this lock, including cancelled HTTP requests.
_INFERENCE_LOCK = threading.RLock()


class ModelLoadError(RuntimeError):
    """Checkpoint validation or device initialization failed."""


def get_inference_semaphore() -> asyncio.Semaphore:
    global _INFERENCE_SEMAPHORE
    if _INFERENCE_SEMAPHORE is None:
        _INFERENCE_SEMAPHORE = asyncio.Semaphore(1)
    return _INFERENCE_SEMAPHORE


def get_inference_lock():
    return _INFERENCE_LOCK


def resolve_preferred_device() -> Tuple[torch.device, str]:
    pref = AI_DEVICE.lower().strip()
    if pref not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"AI_DEVICE='{AI_DEVICE}' không hợp lệ; chọn auto, cuda hoặc cpu.")
    if pref in ("cuda", "auto"):
        if torch.cuda.is_available():
            if not 0 <= CUDA_DEVICE_INDEX < torch.cuda.device_count():
                if pref == "cuda":
                    raise RuntimeError(f"CUDA_DEVICE_INDEX={CUDA_DEVICE_INDEX} không tồn tại.")
                print(f"[ModelRegistry] CUDA_DEVICE_INDEX={CUDA_DEVICE_INDEX} không tồn tại; dùng CPU.")
                return torch.device("cpu"), "cpu"
            return torch.device(f"cuda:{CUDA_DEVICE_INDEX}"), "cuda"
        if pref == "cuda":
            raise RuntimeError("AI_DEVICE='cuda' được yêu cầu nhưng CUDA không khả dụng trên hệ thống.")
    return torch.device("cpu"), "cpu"


def _signature(path: Path) -> Tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def discover_checkpoints() -> Dict[str, Path]:
    """Discover local .pth files; client input is never interpreted as a path."""
    global _CHECKPOINTS, _DEFAULT_CHECKPOINT
    paths = set()
    for directory, recursive in ((REPO_ROOT, False), (MODELS_DIR, True)):
        directory = directory.resolve()
        entries = directory.rglob("*") if recursive else directory.iterdir()
        for entry in entries:
            if entry.suffix.lower() != ".pth" or not entry.is_file():
                continue
            resolved = entry.resolve()
            if resolved.is_relative_to(directory):
                paths.add(resolved)
    if CHECKPOINT_PATH.is_file() and CHECKPOINT_PATH.suffix.lower() == ".pth":
        paths.add(CHECKPOINT_PATH.resolve())
    ordered = sorted(paths, key=lambda path: (path.name.casefold(), str(path).casefold()))
    if _DEFAULT_CHECKPOINT is None and ordered:
        configured = CHECKPOINT_PATH.resolve()
        _DEFAULT_CHECKPOINT = configured if configured in paths else ordered[0]
    checkpoints = {}
    for path in ordered:
        model_id = DEFAULT_MODEL_ID if path == _DEFAULT_CHECKPOINT else (
            "realbasicvsr_" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
        )
        checkpoints[model_id] = path
    _CHECKPOINTS = checkpoints
    return checkpoints


def resolve_model_id(model_id: str = "") -> str:
    checkpoints = discover_checkpoints()
    resolved_id = model_id or _ACTIVE_MODEL_ID or DEFAULT_MODEL_ID
    if resolved_id not in checkpoints:
        raise FileNotFoundError(f"Model '{resolved_id}' không tồn tại. Hãy làm mới danh sách file .pth.")
    return resolved_id


def _warmup(net: RealBasicVSRNet, device: torch.device):
    if device.type == "cuda":
        # T=2 also validates optical flow/grid_sample used by video inference.
        with torch.inference_mode():
            net(torch.zeros(1, 2, 3, 32, 32, device=device, dtype=torch.float32))
            torch.cuda.synchronize(device)


def activate_model(model_id: str = "") -> Tuple[RealBasicVSRNet, torch.device]:
    """Cache used checkpoints on CPU and keep only the active model on GPU.

    Callers can hold the shared reentrant lock throughout activation AND inference.
    """
    global _ACTIVE_MODEL_ID, _ACTIVE_DEVICE
    with _INFERENCE_LOCK:
        model_id = resolve_model_id(model_id)
        path = _CHECKPOINTS[model_id]
        signature = _signature(path)
        if model_id == _ACTIVE_MODEL_ID and _MODEL_SIGNATURES.get(model_id) == signature:
            return _MODEL_CACHE[model_id], _MODEL_DEVICES[model_id]

        try:
            target_dev, _ = resolve_preferred_device()
        except (ValueError, RuntimeError) as exc:
            raise ModelLoadError(str(exc)) from exc

        previous_id, previous_device = _ACTIVE_MODEL_ID, _ACTIVE_DEVICE
        previous_net = _MODEL_CACHE.get(previous_id)
        previous_signature = _MODEL_SIGNATURES.get(previous_id)
        net = _MODEL_CACHE.get(model_id) if _MODEL_SIGNATURES.get(model_id) == signature else None
        if net is None:
            try:
                net, _ = load_realbasicvsr(path, device=torch.device("cpu"))
            except Exception as exc:
                message = f"Không thể nạp '{path.name}': checkpoint lỗi hoặc không tương thích RealBasicVSR x4 (64 channels, 20 blocks)."
                _MODEL_ERRORS[model_id] = (signature, message)
                print(f"[ModelRegistry] {message} {exc}")
                raise ModelLoadError(message) from exc

        # Validate new weights before releasing the previous GPU model.
        if previous_net is not None and previous_device is not None and previous_device.type == "cuda":
            previous_net.to("cpu")
            _MODEL_DEVICES[previous_id] = torch.device("cpu")
            torch.cuda.empty_cache()
        try:
            net.to(target_dev).eval()
            _warmup(net, target_dev)
        except Exception as exc:
            net.to("cpu")
            if target_dev.type == "cuda":
                torch.cuda.empty_cache()
            if target_dev.type == "cuda" and AI_DEVICE == "auto":
                print(f"[ModelRegistry] CUDA failed for '{path.name}'; using CPU: {exc}")
                target_dev = torch.device("cpu")
            else:
                if previous_net is not None:
                    previous_net.to(previous_device)
                    _MODEL_CACHE[previous_id] = previous_net
                    _MODEL_DEVICES[previous_id] = previous_device
                    _MODEL_SIGNATURES[previous_id] = previous_signature
                message = f"Không thể kích hoạt '{path.name}' trên {target_dev}: {str(exc)[:250]}"
                _MODEL_ERRORS[model_id] = (signature, message)
                raise ModelLoadError(message) from exc

        _MODEL_CACHE[model_id] = net
        _MODEL_DEVICES[model_id] = target_dev
        _MODEL_SIGNATURES[model_id] = signature
        _MODEL_ERRORS.pop(model_id, None)
        _ACTIVE_MODEL_ID, _ACTIVE_DEVICE = model_id, target_dev
        print(f"[ModelRegistry] Active checkpoint: '{path.name}' on {target_dev}.")
        return net, target_dev


def load_model_registry() -> Tuple[int, str]:
    """Load the configured checkpoint, or the first valid discovered .pth file."""
    global _DEFAULT_CHECKPOINT, _ACTIVE_MODEL_ID, _ACTIVE_DEVICE, _INFERENCE_SEMAPHORE
    with _INFERENCE_LOCK:
        _MODEL_CACHE.clear()
        _MODEL_DEVICES.clear()
        _MODEL_SIGNATURES.clear()
        _MODEL_ERRORS.clear()
        _DEFAULT_CHECKPOINT = _ACTIVE_MODEL_ID = _ACTIVE_DEVICE = None
        _INFERENCE_SEMAPHORE = asyncio.Semaphore(1)
        checkpoints = discover_checkpoints()
        ordered_ids = sorted(checkpoints, key=lambda item: item != DEFAULT_MODEL_ID)
        for model_id in ordered_ids:
            try:
                activate_model(model_id)
                break
            except (FileNotFoundError, ModelLoadError) as exc:
                print(f"[ModelRegistry] {exc}")
        return len(get_active_model_ids()), str(_ACTIVE_DEVICE or "none")


def get_model(model_id: str = "") -> Optional[Tuple[RealBasicVSRNet, torch.device]]:
    """Read the cache without loading weights (safe for health checks)."""
    model_id = model_id or _ACTIVE_MODEL_ID
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id], _MODEL_DEVICES[model_id]
    return None


def get_active_model_ids() -> List[str]:
    return [_ACTIVE_MODEL_ID] if _ACTIVE_MODEL_ID else []


def list_available_models() -> List[ModelItem]:
    models = []
    for model_id, path in discover_checkpoints().items():
        try:
            signature = _signature(path)
        except OSError:
            continue
        error_signature, message = _MODEL_ERRORS.get(model_id, (None, None))
        error = message if error_signature == signature else None
        loaded = model_id == _ACTIVE_MODEL_ID and _MODEL_SIGNATURES.get(model_id) == signature
        display_path = path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else path.name
        models.append(ModelItem(
            id=model_id,
            name=f"RealBasicVSR · {path.name}",
            description="Checkpoint RealBasicVSR x4 cho ảnh anime (T=1) và video.",
            scale=4,
            available=error is None,
            device=str(_MODEL_DEVICES[model_id]) if loaded else "none",
            checkpoint_filename=path.name,
            checkpoint_path=display_path,
            loaded=loaded,
            error=error,
        ))
    return models
