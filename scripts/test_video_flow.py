import json
import os
import time
import urllib.request
import uuid

BASE_URL = "http://127.0.0.1:8000"


def make_multipart(fields, files):
    boundary = "----TestBoundary" + uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8"))
    for k, (filename, content, mime) in files.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n".encode(
                "utf-8"
            )
        )
        parts.append(content)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def test_flow():
    print("[1] Health check...")
    with urllib.request.urlopen(f"{BASE_URL}/api/health") as res:
        health = json.loads(res.read().decode("utf-8"))
        print("  Health:", health.get("status"), "Device:", health.get("device"))
        assert "video_limits" in health

    sample_path = "frontend/public/demo/anime_sample_2s.mp4"
    with open(sample_path, "rb") as f:
        video_bytes = f.read()

    print("[2] Probe video...")
    body, ct = make_multipart({}, {"file": ("anime_sample_2s.mp4", video_bytes, "video/mp4")})
    req = urllib.request.Request(f"{BASE_URL}/api/video/probe", data=body, headers={"Content-Type": ct})
    with urllib.request.urlopen(req) as res:
        probe = json.loads(res.read().decode("utf-8"))
        print(f"  Probe: {probe['width']}x{probe['height']}, frames: {probe['total_frames']}, dur: {probe['duration_seconds']}s")
        assert probe["width"] == 320
        assert probe["height"] == 240

    print("[3] Submit upscale job (chunk_size=12, overlap=2)...")
    body, ct = make_multipart(
        {"chunk_size": "12", "overlap": "2"},
        {"file": ("anime_sample_2s.mp4", video_bytes, "video/mp4")},
    )
    req = urllib.request.Request(f"{BASE_URL}/api/video/upscale", data=body, headers={"Content-Type": ct})
    with urllib.request.urlopen(req) as res:
        create_res = json.loads(res.read().decode("utf-8"))
        job_id = create_res["job_id"]
        print(f"  Job created: {job_id}")

    print("[4] Polling job status...")
    t0 = time.time()
    while True:
        with urllib.request.urlopen(f"{BASE_URL}/api/video/jobs/{job_id}") as res:
            job_data = json.loads(res.read().decode("utf-8"))
            status = job_data["status"]
            pct = job_data["progress_percent"]
            chunk = f"{job_data.get('current_chunk',0)}/{job_data.get('total_chunks',0)}"
            print(f"  Status: {status} ({pct}%, chunk {chunk}) ...", end="\r")
            if status == "completed":
                print(f"\n  Job {job_id} COMPLETED in {time.time() - t0:.1f}s!")
                break
            elif status == "failed":
                print(f"\n  Job FAILED: {job_data.get('error_message')}")
                raise RuntimeError(job_data.get("error_message"))
        time.sleep(1.0)

    print("[5] Verify video result stream...")
    with urllib.request.urlopen(f"{BASE_URL}/api/video/results/{job_id}") as res:
        ct = res.headers.get("Content-Type")
        cl = res.headers.get("Content-Length")
        print(f"  Result stream headers: Content-Type={ct}, Content-Length={cl}")
        assert "video/mp4" in ct
        assert int(cl) > 10000

    print("ALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_flow()
