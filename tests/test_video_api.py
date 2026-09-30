import asyncio
import io
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import FFMPEG_BIN, WORK_DIR
from backend.app.main import app
from backend.app.model_registry import load_model_registry
from backend.app.video.job_manager import video_job_manager


class TestVideoAPI(unittest.TestCase):
    """Integration test suite for Video Super-Resolution FastAPI endpoints."""

    @classmethod
    def setUpClass(cls):
        load_model_registry()
        cls.client = TestClient(app)
        cls.test_dir = WORK_DIR / "api_test_temp"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.sample_video = cls.test_dir / "api_test_clip.mp4"

        # Generate a small 1s 10fps CFR test video with audio
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

    def test_health_includes_video_limits(self):
        """GET /api/health should include valid video_limits metadata."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("video_limits", data)
        vlimits = data["video_limits"]
        self.assertIsNotNone(vlimits)
        self.assertIn("max_upload_mb", vlimits)
        self.assertIn("max_duration_seconds", vlimits)
        self.assertIn("chunk_presets", vlimits)
        self.assertEqual(vlimits["default_chunk_size"], 7)
        self.assertEqual(vlimits["default_overlap"], 2)

    def test_probe_valid_video(self):
        """POST /api/video/probe with valid MP4 should return video metadata."""
        with open(self.sample_video, "rb") as f:
            resp = self.client.post(
                "/api/video/probe",
                files={"file": ("sample.mp4", f, "video/mp4")},
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["width"], 64)
        self.assertEqual(data["height"], 64)
        self.assertEqual(data["total_frames"], 10)
        self.assertEqual(data["video_codec"], "h264")
        self.assertTrue(data["has_audio"])

    def test_probe_empty_file_returns_400(self):
        """POST /api/video/probe with empty file should return 400 Bad Request."""
        empty_io = io.BytesIO(b"")
        resp = self.client.post(
            "/api/video/probe",
            files={"file": ("empty.mp4", empty_io, "video/mp4")},
        )
        self.assertEqual(resp.status_code, 400)

    def test_probe_non_video_file_returns_400(self):
        """POST /api/video/probe with text payload should return 400 Bad Request."""
        text_io = io.BytesIO(b"this is definitely not an mp4 video file")
        resp = self.client.post(
            "/api/video/probe",
            files={"file": ("fake.mp4", text_io, "video/mp4")},
        )
        self.assertEqual(resp.status_code, 400)

    def test_upscale_invalid_chunk_size_returns_400(self):
        """POST /api/video/upscale with invalid chunk_size should return 400."""
        with open(self.sample_video, "rb") as f:
            resp = self.client.post(
                "/api/video/upscale",
                files={"file": ("sample.mp4", f, "video/mp4")},
                data={"chunk_size": 100, "overlap": 2},
            )
        self.assertEqual(resp.status_code, 400)

    def test_upscale_invalid_overlap_returns_400(self):
        """POST /api/video/upscale with overlap >= chunk_size should return 400."""
        with open(self.sample_video, "rb") as f:
            resp = self.client.post(
                "/api/video/upscale",
                files={"file": ("sample.mp4", f, "video/mp4")},
                data={"chunk_size": 7, "overlap": 7},
            )
        self.assertEqual(resp.status_code, 400)

    def test_upscale_success_submission_returns_202(self):
        """POST /api/video/upscale should accept valid video and return 202 with job_id."""
        with open(self.sample_video, "rb") as f:
            resp = self.client.post(
                "/api/video/upscale",
                files={"file": ("sample.mp4", f, "video/mp4")},
                data={"chunk_size": 7, "overlap": 2},
            )
        self.assertEqual(resp.status_code, 202)
        data = resp.json()
        self.assertIn("job_id", data)
        self.assertEqual(data["status"], "pending")
        job_id = data["job_id"]

        # Poll status
        status_resp = self.client.get(f"/api/video/jobs/{job_id}")
        self.assertEqual(status_resp.status_code, 200)
        s_data = status_resp.json()
        self.assertEqual(s_data["job_id"], job_id)
        self.assertIn(s_data["status"], ("pending", "probing", "decoding", "inferring", "encoding", "completed"))

    def test_job_not_found_returns_404(self):
        """GET /api/video/jobs/{unknown_id} should return 404 Not Found."""
        resp = self.client.get("/api/video/jobs/non_existent_job_123")
        self.assertEqual(resp.status_code, 404)

    def test_result_not_found_returns_404(self):
        """GET /api/video/results/{unknown_id} should return 404 Not Found."""
        resp = self.client.get("/api/video/results/non_existent_result_123")
        self.assertEqual(resp.status_code, 404)

    def test_video_job_execution_lifecycle(self):
        """Execute complete job processing lifecycle and verify successful completion without unpack or lock errors."""
        job_id = video_job_manager.submit_job(
            input_path=self.sample_video,
            source_filename="lifecycle_test.mp4",
            chunk_size=7,
            overlap=2,
        )
        job = video_job_manager.get_job(job_id)
        self.assertIsNotNone(job)

        # Run _process_job synchronously in an event loop
        asyncio.run(video_job_manager._process_job(job))

        self.assertEqual(job.status, "completed", f"Job failed with error: {job.error_message}")
        self.assertIsNone(job.error_message)
        self.assertIsNotNone(job.output_path)
        self.assertTrue(job.output_path.is_file())
        self.assertEqual(job.output_meta["width"], 256)
        self.assertEqual(job.output_meta["height"], 256)
        self.assertEqual(job.output_meta["total_frames"], 10)
        self.assertTrue(job.output_meta["has_audio"])

