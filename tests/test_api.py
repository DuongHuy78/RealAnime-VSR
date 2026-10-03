import io
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient
from PIL import Image

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from backend.app.config import MAX_IMAGE_DIMENSION, MAX_IMAGE_PIXELS, MAX_IMAGE_PIXELS_GPU, MIN_IMAGE_DIMENSION, OUTPUTS_DIR, UPLOADS_DIR
from backend.app.main import app
from backend.app import model_registry
from backend.app.storage import cleanup_old_files, is_valid_uuid


def create_dummy_png_bytes(width: int = 48, height: int = 48, color=(200, 100, 50), mode: str = "RGB") -> bytes:
    """Helper to generate in-memory valid PNG image bytes."""
    img = Image.new(mode, (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestAPI(unittest.TestCase):
    """Integration test suite for FastAPI backend endpoints."""

    @classmethod
    def setUpClass(cls):
        # Using context manager with TestClient triggers FastAPI lifespan (model loading)
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_health_endpoint(self):
        """GET /api/health should return 200 OK and system device info."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("device", data)
        self.assertIn("cuda_available", data)
        self.assertIsInstance(data["active_models_loaded"], list)
        expected_pixels = MAX_IMAGE_PIXELS_GPU if data["device"].startswith("cuda") else MAX_IMAGE_PIXELS
        self.assertEqual(data["image_limits"]["max_image_pixels"], expected_pixels)
        self.assertEqual(data["image_limits"]["max_image_dimension"], MAX_IMAGE_DIMENSION)

    def test_models_endpoint(self):
        """GET /api/models should return available model catalog."""
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        model_ids = [m["id"] for m in data["models"]]
        self.assertIn("realbasicvsr_anime_v3", model_ids)

        # Verify model details
        rb_model = next(m for m in data["models"] if m["id"] == "realbasicvsr_anime_v3")
        self.assertEqual(rb_model["scale"], 4)
        self.assertTrue(rb_model["available"])

    def test_upscale_success_flow(self):
        """POST /api/upscale should accept valid image and return 4x result metadata and url."""
        in_w, in_h = 48, 48
        img_bytes = create_dummy_png_bytes(width=in_w, height=in_h)

        response = self.client.post(
            "/api/upscale",
            files={"image": ("test_thumb.png", img_bytes, "image/png")},
            data={"model_id": "realbasicvsr_anime_v3"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(is_valid_uuid(data["result_id"]))
        self.assertEqual(data["model_id"], "realbasicvsr_anime_v3")
        self.assertEqual(data["scale"], 4)
        self.assertEqual(data["input_size"], {"width": in_w, "height": in_h})
        self.assertEqual(data["output_size"], {"width": in_w * 4, "height": in_h * 4})
        self.assertGreater(data["inference_time_ms"], 0)
        self.assertTrue(data["result_url"].startswith("/api/results/"))

        # Verify retrieving the result file
        result_id = data["result_id"]
        res_get = self.client.get(f"/api/results/{result_id}")
        self.assertEqual(res_get.status_code, 200)
        self.assertEqual(res_get.headers["content-type"], "image/png")

        # Verify image content
        out_img = Image.open(io.BytesIO(res_get.content))
        self.assertEqual(out_img.size, (in_w * 4, in_h * 4))

    def test_upscale_consecutive_requests_cached_model(self):
        """Verify multiple consecutive upscale requests run smoothly using cached model."""
        img_bytes = create_dummy_png_bytes(width=32, height=32)

        for _ in range(2):
            res = self.client.post(
                "/api/upscale",
                files={"image": ("test_cache.png", img_bytes, "image/png")},
                data={"model_id": "realbasicvsr_anime_v3"},
            )
            self.assertEqual(res.status_code, 200)

    def test_checkpoint_switch_and_inference_use_selected_file(self):
        catalog = self.client.get("/api/models").json()["models"]
        if len(catalog) < 2:
            self.skipTest("Two checkpoints are required for switching integration test.")
        initial_id = self.client.get("/api/health").json()["active_models_loaded"][0]
        try:
            for model in catalog:
                activated = self.client.post(f"/api/models/{model['id']}/activate")
                self.assertEqual(activated.status_code, 200, activated.text)
                self.assertTrue(activated.json()["loaded"])
                self.assertEqual(activated.json()["checkpoint_filename"], model["checkpoint_filename"])
                with patch.object(model_registry, "load_realbasicvsr", side_effect=AssertionError("Cached checkpoint must not reload")):
                    result = self.client.post(
                        "/api/upscale",
                        files={"image": ("switch.png", create_dummy_png_bytes(24, 32), "image/png")},
                        data={"model_id": model["id"]},
                    )
                self.assertEqual(result.status_code, 200, result.text)
                self.assertEqual(result.json()["checkpoint_filename"], model["checkpoint_filename"])
                self.assertEqual(result.json()["output_size"], {"width": 96, "height": 128})
                loaded = [item for item in self.client.get("/api/models").json()["models"] if item["loaded"]]
                self.assertEqual([item["id"] for item in loaded], [model["id"]])
                self.assertEqual(self.client.get("/api/health").json()["device"], result.json()["device"])
        finally:
            self.client.post(f"/api/models/{initial_id}/activate")

    def test_activate_unknown_model_returns_404(self):
        self.assertEqual(self.client.post("/api/models/unknown/activate").status_code, 404)

    def test_activate_invalid_checkpoint_returns_422_and_keeps_current_model(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=model_registry.MODELS_DIR) as temp:
            path = Path(temp) / "broken.pth"
            path.write_bytes(b"invalid pytorch checkpoint")
            initial = self.client.get("/api/health").json()["active_models_loaded"]
            broken = next(model for model in self.client.get("/api/models").json()["models"] if model["checkpoint_filename"] == "broken.pth")
            response = self.client.post(f"/api/models/{broken['id']}/activate")
            self.assertEqual(response.status_code, 422)
            self.assertIn("không tương thích", response.json()["detail"])
            self.assertEqual(self.client.get("/api/health").json()["active_models_loaded"], initial)

    def test_download_query_param(self):
        """GET /api/results/{result_id}?download=true should include attachment header."""
        img_bytes = create_dummy_png_bytes(width=32, height=32)
        post_res = self.client.post(
            "/api/upscale",
            files={"image": ("test.png", img_bytes, "image/png")},
            data={"model_id": "realbasicvsr_anime_v3"},
        )
        result_id = post_res.json()["result_id"]

        get_res = self.client.get(f"/api/results/{result_id}?download=true")
        self.assertEqual(get_res.status_code, 200)
        self.assertIn("attachment", get_res.headers.get("content-disposition", ""))

    def test_invalid_file_format_returns_400(self):
        """Uploading non-image file should return 400 Bad Request."""
        fake_bytes = b"Hello, this is not an image file!"
        response = self.client.post(
            "/api/upscale",
            files={"image": ("document.txt", fake_bytes, "text/plain")},
            data={"model_id": "realbasicvsr_anime_v3"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("không được hỗ trợ", response.json()["detail"])

    def test_extension_must_match_decoded_image_format(self):
        """A JPEG payload renamed to .png must not bypass format validation."""
        image = Image.new("RGB", (32, 32), color=(200, 100, 50))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        response = self.client.post(
            "/api/upscale",
            files={"image": ("wrong.png", buffer.getvalue(), "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("không khớp", response.json()["detail"])

    def test_inference_memory_failure_returns_clear_error(self):
        """The API should report a memory failure without an unhandled 500."""
        with patch("backend.app.main.infer_rb_single", side_effect=MemoryError("simulated")):
            response = self.client.post(
                "/api/upscale",
                files={"image": ("small.png", create_dummy_png_bytes(32, 32), "image/png")},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("Không đủ bộ nhớ", response.json()["detail"])

    def test_empty_file_returns_400(self):
        """Uploading empty file should return 400 Bad Request."""
        response = self.client.post(
            "/api/upscale",
            files={"image": ("empty.png", b"", "image/png")},
            data={"model_id": "realbasicvsr_anime_v3"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("rỗng", response.json()["detail"])

    def test_invalid_model_id_returns_404(self):
        """Requesting un-registered model should return 404 Not Found."""
        img_bytes = create_dummy_png_bytes(width=32, height=32)
        response = self.client.post(
            "/api/upscale",
            files={"image": ("test.png", img_bytes, "image/png")},
            data={"model_id": "non_existent_model_id"},
        )
        self.assertEqual(response.status_code, 404)

    def test_oversized_image_returns_400(self):
        """Uploading oversized image should return 400 Bad Request."""
        # Exercise the CPU-sized pixel budget independently of the active device.
        img_bytes = create_dummy_png_bytes(width=400, height=300)
        with patch("backend.app.main.MAX_IMAGE_PIXELS", 102400), patch("backend.app.main.MAX_IMAGE_PIXELS_GPU", 102400):
            response = self.client.post(
                "/api/upscale",
                files={"image": ("large.png", img_bytes, "image/png")},
                data={"model_id": "realbasicvsr_anime_v3"},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("vượt quá ngưỡng xử lý an toàn", response.json()["detail"])

    def test_path_traversal_protection(self):
        """GET /api/results/ with path traversal strings should return 400 or 404."""
        attack_paths = [
            "../../etc/passwd",
            "..%2F..%2Fetc%2Fpasswd",
            "non-uuid-string",
            "12345",
        ]
        for bad_id in attack_paths:
            res = self.client.get(f"/api/results/{bad_id}")
            self.assertIn(res.status_code, [400, 404])

    def test_min_dimension_boundary_success(self):
        """Image exactly at minimum dimension (16x16) should upscale to 64x64."""
        img_bytes = create_dummy_png_bytes(width=16, height=16)
        res = self.client.post(
            "/api/upscale",
            files={"image": ("min16.png", img_bytes, "image/png")},
            data={"model_id": "realbasicvsr_anime_v3"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["input_size"], {"width": 16, "height": 16})
        self.assertEqual(data["output_size"], {"width": 64, "height": 64})

    def test_under_min_dimension_rejected(self):
        """Image below minimum dimension (15x15) should be rejected with 400."""
        img_bytes = create_dummy_png_bytes(width=15, height=15)
        res = self.client.post(
            "/api/upscale",
            files={"image": ("under16.png", img_bytes, "image/png")},
            data={"model_id": "realbasicvsr_anime_v3"},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("quá nhỏ", res.json()["detail"])

    def test_alpha_transparent_png_handled_cleanly(self):
        """RGBA image with transparency channel should upscale without black fringes or error."""
        # Semi-transparent red pixel
        img_bytes = create_dummy_png_bytes(width=24, height=24, color=(255, 50, 50, 128), mode="RGBA")
        res = self.client.post(
            "/api/upscale",
            files={"image": ("transparent.png", img_bytes, "image/png")},
            data={"model_id": "realbasicvsr_anime_v3"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["output_size"], {"width": 96, "height": 96})

    def test_special_characters_and_unicode_filename(self):
        """Filenames with Vietnamese accents, spaces, and brackets should be handled cleanly."""
        img_bytes = create_dummy_png_bytes(width=20, height=20)
        res = self.client.post(
            "/api/upscale",
            files={"image": ("hình_ảnh_anime #01 (thử nghiệm).png", img_bytes, "image/png")},
            data={"model_id": "realbasicvsr_anime_v3"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(is_valid_uuid(data["result_id"]))

    def test_non_square_aspect_ratio_success(self):
        """Non-square images (e.g. 16x48) should be correctly padded and cropped to 4x."""
        img_bytes = create_dummy_png_bytes(width=16, height=48)
        res = self.client.post(
            "/api/upscale",
            files={"image": ("aspect.png", img_bytes, "image/png")},
            data={"model_id": "realbasicvsr_anime_v3"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["input_size"], {"width": 16, "height": 48})
        self.assertEqual(data["output_size"], {"width": 64, "height": 192})

    def test_cleanup_old_files_logic(self):
        """Verify cleanup_old_files deletes expired UUID files and retains fresh/non-UUID files."""
        # 1. Create an expired UUID file (mtime set to 30 hours ago)
        expired_id = str(uuid.uuid4())
        expired_file = OUTPUTS_DIR / f"{expired_id}.png"
        expired_file.write_bytes(b"dummy_expired_data")
        past_time = time.time() - (30 * 3600)
        os.utime(str(expired_file), (past_time, past_time))

        # 2. Create a fresh UUID file (mtime now)
        fresh_id = str(uuid.uuid4())
        fresh_file = OUTPUTS_DIR / f"{fresh_id}.png"
        fresh_file.write_bytes(b"dummy_fresh_data")

        # 3. Create a non-UUID file that should never be deleted by cleanup
        non_uuid_file = OUTPUTS_DIR / "custom_keep.txt"
        non_uuid_file.write_bytes(b"important_data")
        os.utime(str(non_uuid_file), (past_time, past_time))

        try:
            removed = cleanup_old_files(max_age_hours=24)
            self.assertGreaterEqual(removed, 1)
            self.assertFalse(expired_file.exists(), "Expired UUID file must be deleted.")
            self.assertTrue(fresh_file.exists(), "Fresh UUID file must be preserved.")
            self.assertTrue(non_uuid_file.exists(), "Non-UUID file must be preserved.")
        finally:
            if fresh_file.exists():
                fresh_file.unlink()
            if non_uuid_file.exists():
                non_uuid_file.unlink()

    def test_concurrency_serializes_requests(self):
        """Two concurrent upscale requests should both complete successfully without race conditions."""
        img1 = create_dummy_png_bytes(width=24, height=24, color=(100, 150, 200))
        img2 = create_dummy_png_bytes(width=24, height=24, color=(200, 150, 100))

        def send_request(img_bytes, filename):
            return self.client.post(
                "/api/upscale",
                files={"image": (filename, img_bytes, "image/png")},
                data={"model_id": "realbasicvsr_anime_v3"},
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(send_request, img1, "concurrent1.png")
            future2 = executor.submit(send_request, img2, "concurrent2.png")
            res1 = future1.result()
            res2 = future2.result()

        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res2.status_code, 200)
        self.assertNotEqual(res1.json()["result_id"], res2.json()["result_id"])


if __name__ == "__main__":
    unittest.main()
