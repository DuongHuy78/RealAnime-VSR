import sys
import time
import os
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import psutil
import torch
from backend.app.inference.realbasicvsr_pipeline import infer_rb_single, load_realbasicvsr

def benchmark_resolution(net, device, h, w):
    dummy_img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    
    # Warmup
    _ = infer_rb_single(net, dummy_img[:64, :64], device=device)
    
    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss
    peak_rss = [baseline_rss]
    stop_sampling = threading.Event()

    def sample_rss():
        while not stop_sampling.wait(0.01):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    t0 = time.perf_counter()
    try:
        out = infer_rb_single(net, dummy_img, device=device)
        elapsed = time.perf_counter() - t0
    finally:
        stop_sampling.set()
        sampler.join()
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    to_mb = 1024 * 1024
    baseline_mb = baseline_rss / to_mb
    peak_mb = peak_rss[0] / to_mb
    delta_mb = (peak_rss[0] - baseline_rss) / to_mb
    print(
        f"Input: {w}x{h} -> Output: {out.shape[1]}x{out.shape[0]} | "
        f"Time: {elapsed:.2f}s | Process RSS baseline/peak/delta: "
        f"{baseline_mb:.1f}/{peak_mb:.1f}/{delta_mb:.1f} MB"
    )
    return elapsed, peak_mb

def main():
    print("=" * 60)
    print("RealBasicVSR Anime V3 — CPU Resource Profiling")
    print("=" * 60)
    
    net, device = load_realbasicvsr("best.pth", device=torch.device("cpu"))
    print(f"Device: {device}")
    
    test_sizes = [
        (128, 128),
        (224, 224),
        (320, 320),
    ]
    
    for h, w in test_sizes:
        benchmark_resolution(net, device, h, w)

if __name__ == "__main__":
    main()
