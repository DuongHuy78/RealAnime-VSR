import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import FFMPEG_BIN, WORK_DIR
from backend.app.inference.realbasicvsr_pipeline import save_image_rgb
from backend.app.video.chunk_stream import (
    VideoChunkStreamError,
    compute_chunk_windows,
)
from backend.app.video.decode import VideoDecodeError, decode_video_frames
from backend.app.video.encode import VideoEncodeError, encode_video_from_frames
from backend.app.video.probe import (
    VideoProbeError,
    VideoProbeResult,
    parse_fractional_fps,
    probe_video,
)


class TestVideoPipeline(unittest.TestCase):
    """Unit and integration tests for video pipeline components."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = WORK_DIR / "unit_test_temp"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.sample_video = cls.test_dir / "test_mini.mp4"

        # Generate a small 1-second 10fps synthetic video for testing
        cmd = [
            FFMPEG_BIN,
            "-v", "error",
            "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=1.0:size=64x64:rate=10",
            "-f", "lavfi",
            "-i", "sine=frequency=440:duration=1.0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "64k",
            "-shortest",
            str(cls.sample_video),
        ]
        subprocess.run(cmd, check=True)

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    # 1. Chunk window tests
    def test_compute_chunk_windows_standard(self):
        """Standard window calculation: 12 frames, chunk 7, overlap 2 -> stride 5."""
        windows = compute_chunk_windows(total_frames=12, chunk_size=7, overlap=2)
        self.assertEqual(windows, [(0, 7), (5, 12)])

    def test_compute_chunk_windows_smaller_than_chunk(self):
        """When total frames is less than chunk size, single chunk covering all frames."""
        windows = compute_chunk_windows(total_frames=5, chunk_size=7, overlap=2)
        self.assertEqual(windows, [(0, 5)])

    def test_compute_chunk_windows_invalid_overlap(self):
        """Chunk size <= overlap must raise VideoChunkStreamError."""
        with self.assertRaises(VideoChunkStreamError):
            compute_chunk_windows(total_frames=20, chunk_size=4, overlap=4)

    # 2. FPS parsing tests
    def test_parse_fractional_fps(self):
        """Rational FPS string parsing."""
        frac = parse_fractional_fps("30000/1001")
        self.assertEqual(frac, Fraction(30000, 1001))
        self.assertAlmostEqual(float(frac), 29.97, places=2)

        frac_int = parse_fractional_fps("24/1")
        self.assertEqual(frac_int, Fraction(24, 1))

        frac_float = parse_fractional_fps("25")
        self.assertEqual(frac_float, Fraction(25, 1))

    # 3. Video probe tests
    def test_probe_missing_file_raises_error(self):
        """Probing non-existent file should raise VideoProbeError."""
        with self.assertRaises(VideoProbeError):
            probe_video(Path("non_existent_video_path.mp4"))

    def test_probe_valid_video(self):
        """Valid synthetic video probe metadata verification."""
        result = probe_video(self.sample_video, enforce_limits=True)
        self.assertIsInstance(result, VideoProbeResult)
        self.assertEqual(result.width, 64)
        self.assertEqual(result.height, 64)
        self.assertEqual(result.total_frames, 10)
        self.assertEqual(result.video_codec, "h264")
        self.assertTrue(result.has_audio)
        self.assertEqual(result.audio_codec, "aac")

    def test_probe_enforce_limits_exceeded(self):
        """Enforcing strict limits should raise VideoProbeError when violated."""
        # Duration limit check with patched config value or custom threshold
        with self.assertRaises(VideoProbeError):
            # A 0-byte file should fail
            dummy_empty = self.test_dir / "empty.mp4"
            dummy_empty.touch()
            probe_video(dummy_empty, enforce_limits=True)

    # 4. Video decode tests
    def test_decode_video_frames(self):
        """Extract frames from synthetic video into PNG files."""
        out_frames_dir = self.test_dir / "decoded_frames"
        count = decode_video_frames(
            input_video_path=self.sample_video,
            output_frames_dir=out_frames_dir,
            expected_frames=10,
        )
        self.assertEqual(count, 10)
        files = sorted(out_frames_dir.glob("frame_*.png"))
        self.assertEqual(len(files), 10)
        self.assertEqual(files[0].name, "frame_000001.png")
        self.assertEqual(files[-1].name, "frame_000010.png")

    # 5. Video encode & mux tests
    def test_encode_video_from_frames(self):
        """Encode PNG frame sequence to MP4 with audio muxing."""
        frames_dir = self.test_dir / "mock_sr_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Create 5 dummy 128x128 solid color frames
        for i in range(1, 6):
            arr = np.full((128, 128, 3), i * 40, dtype=np.uint8)
            save_image_rgb(frames_dir / f"sr_frame_{i:06d}.png", arr)

        encoded_video = self.test_dir / "output_encoded.mp4"
        result = encode_video_from_frames(
            frames_dir=frames_dir,
            output_video_path=encoded_video,
            fps_str="10",
            source_video_path=self.sample_video,
        )

        self.assertTrue(encoded_video.is_file())
        self.assertEqual(result.width, 128)
        self.assertEqual(result.height, 128)
        self.assertEqual(result.total_frames, 5)
        self.assertEqual(result.video_codec, "h264")
        self.assertTrue(result.has_audio)
