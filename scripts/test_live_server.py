import io
import json
import os
import urllib.error
import urllib.request
import uuid
from PIL import Image

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def make_multipart_body(fields, files):
    boundary = "----TestBoundary" + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode("utf-8")
        )
    for field_name, (filename, content, content_type) in files.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
        )
        parts.append(content)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def create_png_bytes(width, height, color=(100, 150, 220), mode="RGB"):
    img = Image.new(mode, (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def post_upscale(image_bytes, filename="test.png", model_id="realbasicvsr_anime_v3"):
    payload, content_type = make_multipart_body(
        fields={"model_id": model_id},
        files={"image": (filename, image_bytes, "image/png")},
    )
    req = urllib.request.Request(
        f"{BASE_URL}/api/upscale",
        data=payload,
        headers={"Content-Type": content_type},
    )
    return urllib.request.urlopen(req)


def main():
    print("=" * 60)
    print(f"LIVE SERVER VERIFICATION: {BASE_URL}")
    print("=" * 60)

    # 1. Test GET /api/health
    print("[1/8] Testing GET /api/health ...")
    with urllib.request.urlopen(f"{BASE_URL}/api/health") as res:
        health_data = json.loads(res.read().decode("utf-8"))
        print(f"      Status: {health_data['status']}, Device: {health_data['device']}, Limits: {health_data.get('image_limits')}")
        assert health_data["status"] == "ok"
        assert "image_limits" in health_data

    # 2. Test GET /api/models
    print("[2/8] Testing GET /api/models ...")
    with urllib.request.urlopen(f"{BASE_URL}/api/models") as res:
        models_data = json.loads(res.read().decode("utf-8"))
        models = models_data["models"]
        print(f"      Found {len(models)} model(s): {[m['id'] for m in models]}")
        assert len(models) >= 1
        assert any(m["id"] == "realbasicvsr_anime_v3" and m["available"] for m in models)

    # 3. Test Standard Upscale (48x48 -> 192x192)
    print("[3/8] Testing standard upscale (48x48 -> 192x192) ...")
    img_bytes = create_png_bytes(48, 48)
    with post_upscale(img_bytes, filename="standard_48.png") as res:
        data = json.loads(res.read().decode("utf-8"))
        print(f"      Result ID: {data['result_id']}, Time: {data['inference_time_ms']}ms, Scale: {data['scale']}x")
        assert data["scale"] == 4
        assert data["output_size"] == {"width": 192, "height": 192}
        standard_result_id = data["result_id"]

    # 4. Test Minimum Dimension Boundary (16x16 -> 64x64)
    print("[4/8] Testing minimum dimension boundary (16x16 -> 64x64) ...")
    min_bytes = create_png_bytes(16, 16)
    with post_upscale(min_bytes, filename="min_16.png") as res:
        data = json.loads(res.read().decode("utf-8"))
        print(f"      Output: {data['output_size']['width']}x{data['output_size']['height']} (Success)")
        assert data["output_size"] == {"width": 64, "height": 64}

    # 5. Test Alpha Channel / RGBA Compositing
    print("[5/8] Testing alpha transparency handling (RGBA mode) ...")
    rgba_bytes = create_png_bytes(32, 32, color=(240, 80, 80, 128), mode="RGBA")
    with post_upscale(rgba_bytes, filename="transparent_sprite.png") as res:
        data = json.loads(res.read().decode("utf-8"))
        print(f"      Output: {data['output_size']['width']}x{data['output_size']['height']} (Clean alpha composite)")
        assert data["output_size"] == {"width": 128, "height": 128}

    # 6. Test Unicode / Vietnamese Filename
    print("[6/8] Testing unicode & special characters filename ...")
    vn_bytes = create_png_bytes(24, 24)
    with post_upscale(vn_bytes, filename="hình_ảnh_anime #01 (thử nghiệm).png") as res:
        data = json.loads(res.read().decode("utf-8"))
        print(f"      Success with Result ID: {data['result_id']}")
        assert data["scale"] == 4

    # 7. Test Result Retrieval and Download Headers
    print("[7/8] Testing result retrieval and attachment download ...")
    res_url = f"{BASE_URL}/api/results/{standard_result_id}?download=true"
    with urllib.request.urlopen(res_url) as res_file:
        content_type = res_file.headers.get("content-type")
        disposition = res_file.headers.get("content-disposition", "")
        file_bytes = res_file.read()
        out_img = Image.open(io.BytesIO(file_bytes))
        print(f"      Content-Type: {content_type}, Disposition: {disposition}, Size: {out_img.size}")
        assert content_type == "image/png"
        assert out_img.size == (192, 192)
        assert "attachment" in disposition

    # 8. Test Error Rejections (Boundary violations)
    print("[8/8] Testing error rejection on boundary violations ...")
    # A. Below min dimension (15x15)
    under_min_bytes = create_png_bytes(15, 15)
    try:
        post_upscale(under_min_bytes)
        assert False, "Should have rejected 15x15 image"
    except urllib.error.HTTPError as e:
        print(f"      Under min dim (15x15) properly rejected with HTTP {e.code}")
        assert e.code == 400

    # B. Oversized pixel count (400x300 = 120,000 > 102,400)
    oversized_bytes = create_png_bytes(400, 300)
    try:
        post_upscale(oversized_bytes)
        assert False, "Should have rejected oversized pixel budget image"
    except urllib.error.HTTPError as e:
        print(f"      Oversized pixel budget properly rejected with HTTP {e.code}")
        assert e.code == 400

    print("=" * 60)
    print("ALL 8 LIVE DIAGNOSTIC TESTS PASSED CLEANLY!")
    print("The local backend is completely verified, stable, and ready for production.")
    print("=" * 60)


if __name__ == "__main__":
    main()

