import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.inference.realbasicvsr_pipeline import (
    infer_rb_sequence,
    infer_rb_single,
    load_realbasicvsr,
    pad_lq,
)
from backend.app.models.realbasicvsr_net import RealBasicVSRNet


class TestRealBasicVSR(unittest.TestCase):
    """Sanity checks and unit tests for RealBasicVSR architecture and pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = REPO_ROOT / "best.pth"
        if cls.ckpt_path.is_file():
            cls.model, cls.device = load_realbasicvsr(cls.ckpt_path, device=torch.device("cpu"))
        else:
            cls.model = None
            cls.device = torch.device("cpu")

    def test_checkpoint_strict_loading(self):
        """Verify best.pth exists and loads strictly into RealBasicVSRNet."""
        self.assertTrue(
            self.ckpt_path.is_file(),
            f"Checkpoint file '{self.ckpt_path}' must exist on disk.",
        )
        self.assertIsNotNone(self.model, "Model should be successfully instantiated.")
        self.assertIsInstance(self.model, RealBasicVSRNet)

    def test_missing_checkpoint_raises_error(self):
        """Verify that loading a non-existent checkpoint raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_realbasicvsr("non_existent_checkpoint.pth")

    def test_padding_divisible_by_32(self):
        """Input already divisible by 32 should have 0 padding."""
        x = torch.zeros(1, 1, 3, 64, 64)
        x_pad, (h, w) = pad_lq(x)
        self.assertEqual(x_pad.shape, (1, 1, 3, 64, 64))
        self.assertEqual((h, w), (64, 64))

    def test_padding_arbitrary_dimensions(self):
        """Input not divisible by 32 should be properly padded to multiples of 32."""
        # 70 -> 96 (pad 26), 95 -> 96 (pad 1)
        x = torch.zeros(1, 2, 3, 70, 95)
        x_pad, (h, w) = pad_lq(x)
        self.assertEqual(x_pad.shape, (1, 2, 3, 96, 96))
        self.assertEqual((h, w), (70, 95))

    def test_padding_very_small_dimensions(self):
        """Input smaller than 32 should not crash PyTorch reflection pad."""
        x = torch.zeros(1, 1, 3, 16, 16)
        x_pad, (h, w) = pad_lq(x)
        self.assertEqual(x_pad.shape, (1, 1, 3, 32, 32))
        self.assertEqual((h, w), (16, 16))

    def test_single_image_inference_shape_x4(self):
        """Verify single image (T=1) inference produces exact 4x spatial resolution."""
        if self.model is None:
            self.skipTest("best.pth not available.")

        # Test with a tiny arbitrary-size image: 48 x 52
        h, w = 48, 52
        dummy_img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)

        out = infer_rb_single(self.model, dummy_img, device=self.device)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (h * 4, w * 4, 3))

    def test_sequence_inference_shape_x4(self):
        """Verify sequence inference produces exact 4x spatial resolution across all frames."""
        if self.model is None:
            self.skipTest("best.pth not available.")

        # Test sequence of 3 frames, each 32 x 32
        h, w = 32, 32
        num_frames = 3
        frames = [(np.random.rand(h, w, 3) * 255).astype(np.uint8) for _ in range(num_frames)]

        out_frames = infer_rb_sequence(
            self.model,
            frames,
            chunk_size=3,
            overlap=1,
            device=self.device,
        )
        self.assertEqual(len(out_frames), num_frames)
        for out in out_frames:
            self.assertEqual(out.dtype, np.uint8)
            self.assertEqual(out.shape, (h * 4, w * 4, 3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available in this Python environment.")
    def test_cuda_single_image_matches_cpu_shape_and_pixels(self):
        """Exercise the real CUDA checkpoint path with a small image."""
        gpu_model, gpu_device = load_realbasicvsr(self.ckpt_path, device=torch.device("cuda:0"))
        image = np.random.default_rng(7).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        gpu_out = infer_rb_single(gpu_model, image, device=gpu_device)
        cpu_out = infer_rb_single(self.model, image, device=self.device)
        self.assertEqual(gpu_out.shape, (128, 128, 3))
        self.assertLess(float(np.abs(gpu_out.astype(np.int16) - cpu_out.astype(np.int16)).mean()), 0.1)


if __name__ == "__main__":
    unittest.main()
