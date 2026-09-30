import os
import sys
import time
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import psutil
import torch

from backend.app.config import CUDA_DEVICE_INDEX
from backend.app.inference.realbasicvsr_pipeline import infer_rb_single, load_realbasicvsr


def benchmark_device(net, device, image, runs=3):
    # Warm up with the same resolution as the timed runs.
    _ = infer_rb_single(net, image, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    out = None
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = infer_rb_single(net, image, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

    median_time = statistics.median(times)
    mem_info = {}
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        reserved = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        mem_info = {"allocated_mb": round(allocated, 1), "reserved_mb": round(reserved, 1)}
    else:
        process = psutil.Process(os.getpid())
        rss_mb = process.memory_info().rss / (1024 * 1024)
        mem_info = {"rss_mb": round(rss_mb, 1)}

    return median_time, mem_info, out


def main():
    print("=" * 70)
    print("RealBasicVSR Anime V3 — CPU vs CUDA (NVIDIA GTX 1650) Benchmark")
    print("=" * 70)

    ckpt_path = REPO_ROOT / "best.pth"
    if not ckpt_path.is_file():
        print(f"Error: Checkpoint file not found at '{ckpt_path}'")
        sys.exit(1)

    cuda_available = torch.cuda.is_available()
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {cuda_available}")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(CUDA_DEVICE_INDEX)
        total_vram_gb = torch.cuda.get_device_properties(CUDA_DEVICE_INDEX).total_memory / (1024**3)
        print(f"GPU Hardware:    {gpu_name} ({total_vram_gb:.1f} GB VRAM)")
    else:
        print("Note: Running in CPU-only mode. CUDA build needed for GPU benchmarking.")

    # Load CPU model
    print("\n[1/3] Loading RealBasicVSRNet on CPU...")
    net_cpu, cpu_device = load_realbasicvsr(ckpt_path, device=torch.device("cpu"))

    net_cuda, cuda_device = None, None
    if cuda_available:
        print(f"[2/3] Loading RealBasicVSRNet on CUDA:{CUDA_DEVICE_INDEX}...")
        net_cuda, cuda_device = load_realbasicvsr(ckpt_path, device=torch.device(f"cuda:{CUDA_DEVICE_INDEX}"))
    else:
        print("[2/3] Skipped CUDA model loading (CUDA not active in this Python environment).")

    test_resolutions = [
        (64, 64),
        (128, 128),
        (224, 224),
        (320, 320),
    ]

    print("\n[3/3] Running Benchmarks across resolutions...")
    print("-" * 70)
    print(f"{'Input Size':<12} | {'CPU Time':<10} | {'GPU Time':<10} | {'Speedup':<8} | {'GPU VRAM Peak':<14} | {'Diff (MAE)':<12}")
    print("-" * 70)

    rng = np.random.default_rng(1234)
    for h, w in test_resolutions:
        dummy_in = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        cpu_time, _cpu_mem, out_cpu = benchmark_device(net_cpu, cpu_device, dummy_in, runs=2)

        if cuda_available and net_cuda is not None:
            gpu_time, gpu_mem, out_gpu = benchmark_device(net_cuda, cuda_device, dummy_in, runs=2)
            vram_mb = gpu_mem["allocated_mb"]
            speedup = cpu_time / max(gpu_time, 1e-5)
            # Compare the final uint8 pixels, not internal floating-point tensors.
            mae = np.mean(np.abs(out_cpu.astype(np.float32) - out_gpu.astype(np.float32)))

            print(
                f"{w}x{h:<8} | {cpu_time:>8.2f}s | {gpu_time:>8.3f}s | {speedup:>6.1f}x | {vram_mb:>10.1f} MB | {mae:>10.6f}"
            )
        else:
            print(f"{w}x{h:<8} | {cpu_time:>8.2f}s | {'N/A':>10} | {'N/A':>8} | {'N/A':>14} | {'N/A':>10}")

    print("-" * 70)
    print("Benchmark complete!\n")


if __name__ == "__main__":
    main()
