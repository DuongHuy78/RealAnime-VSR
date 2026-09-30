import argparse
import sys
import time
from pathlib import Path
from typing import List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from backend.app.inference.realbasicvsr_pipeline import (
    get_default_device,
    infer_rb_sequence,
    infer_rb_single,
    load_realbasicvsr,
    read_image_rgb,
    save_image_rgb,
)

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="RealBasicVSR Anime V3 Super-Resolution CLI Tool (x4 upscale)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best.pth",
        help="Path to RealBasicVSR checkpoint file (default: best.pth).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "sequence", "auto"],
        default="auto",
        help="Inference mode: 'single' (one image T=1), 'sequence' (frame folder), or 'auto' (detect from input).",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image file or directory containing frame sequence.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output image file or output directory for sequence.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=7,
        help="Chunk size (number of frames per sliding window) for sequence mode (default: 7).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=2,
        help="Overlap frames between consecutive chunks for sequence mode (default: 2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Computation device (default: auto).",
    )
    return parser.parse_args()


def get_image_files(folder: Path) -> List[Path]:
    """Find and naturally sort image files in directory."""
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    return sorted(files, key=lambda x: x.name)


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine mode
    mode = args.mode
    if mode == "auto":
        if input_path.is_dir():
            mode = "sequence"
        elif input_path.is_file():
            mode = "single"
        else:
            print(f"[ERROR] Invalid input path: {input_path}", file=sys.stderr)
            sys.exit(1)

    # Resolve device
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("[ERROR] CUDA requested but not available. Aborting.", file=sys.stderr)
            sys.exit(1)
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = get_default_device()

    print("=" * 60)
    print("RealBasicVSR Anime V3 — Super-Resolution CLI (x4)")
    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode:       {mode}")
    print(f"Input:      {input_path}")
    print(f"Output:     {args.output}")
    if mode == "sequence":
        print(f"Chunk Size: {args.chunk_size}, Overlap: {args.overlap}")
    print("=" * 60)

    # Load model
    print("[1/3] Loading checkpoint...")
    t0 = time.time()
    try:
        net, actual_device = load_realbasicvsr(args.checkpoint, device=device)
    except Exception as e:
        print(f"[ERROR] Checkpoint loading failed: {e}", file=sys.stderr)
        sys.exit(1)
    load_time = time.time() - t0
    print(f"       Checkpoint loaded successfully in {load_time:.2f}s.")

    # Execute inference
    output_path = Path(args.output)
    if mode == "single":
        if not input_path.is_file():
            print(f"[ERROR] Single image mode expects a file, but got directory: {input_path}", file=sys.stderr)
            sys.exit(1)

        print("[2/3] Reading input image...")
        try:
            img_rgb = read_image_rgb(input_path)
        except Exception as e:
            print(f"[ERROR] Failed to read input image: {e}", file=sys.stderr)
            sys.exit(1)

        h, w = img_rgb.shape[:2]
        expected_h, expected_w = h * 4, w * 4
        print(f"       Input resolution:  {w}x{h}")
        print(f"       Target resolution: {expected_w}x{expected_h} (x4)")

        print("[3/3] Running inference (T=1)...")
        t_start = time.time()
        out_rgb = infer_rb_single(net, img_rgb, device=actual_device)
        infer_time = time.time() - t_start

        actual_h, actual_w = out_rgb.shape[:2]
        print(f"       Inference completed in {infer_time:.2f}s.")
        print(f"       Output resolution: {actual_w}x{actual_h}")

        if (actual_h, actual_w) != (expected_h, expected_w):
            print(
                f"[ERROR] Output shape mismatch! Expected ({expected_h}, {expected_w}), got ({actual_h}, {actual_w})",
                file=sys.stderr,
            )
            sys.exit(1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image_rgb(output_path, out_rgb)
        print(f"[SUCCESS] Result saved to: {output_path.resolve()}")

    elif mode == "sequence":
        if not input_path.is_dir():
            print(f"[ERROR] Sequence mode expects a directory, but got: {input_path}", file=sys.stderr)
            sys.exit(1)

        image_files = get_image_files(input_path)
        if not image_files:
            print(f"[ERROR] No supported image files found in {input_path}", file=sys.stderr)
            sys.exit(1)

        print(f"[2/3] Reading {len(image_files)} frames...")
        frames = [read_image_rgb(f) for f in image_files]
        h, w = frames[0].shape[:2]
        expected_h, expected_w = h * 4, w * 4
        print(f"       Frame resolution:  {w}x{h}")
        print(f"       Target resolution: {expected_w}x{expected_h} (x4)")

        print(f"[3/3] Running sequence inference (chunk={args.chunk_size}, overlap={args.overlap})...")
        t_start = time.time()
        out_frames = infer_rb_sequence(
            net,
            frames,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            device=actual_device,
        )
        infer_time = time.time() - t_start
        fps = len(frames) / max(infer_time, 0.001)
        print(f"       Sequence inference completed in {infer_time:.2f}s ({fps:.2f} FPS).")

        output_path.mkdir(parents=True, exist_ok=True)
        for idx, (f_path, out_rgb) in enumerate(zip(image_files, out_frames)):
            out_file = output_path / f_path.name
            save_image_rgb(out_file, out_rgb)

        print(f"[SUCCESS] {len(out_frames)} frames saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
