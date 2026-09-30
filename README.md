# RealAnime-VSR — AI Anime Video & Image Super-Resolution Studio

> **High-Performance 4X Video & Image Super-Resolution Web Studio powered by RealBasicVSR, PyTorch (CUDA), and FastAPI.**
> Ứng dụng web cục bộ chuyên dụng phục chế và nâng cấp độ phân giải 4X cho hình ảnh và video Anime/Animation bằng Trí tuệ nhân tạo (RealBasicVSR Anime V3), hỗ trợ tăng tốc GPU NVIDIA CUDA và giao diện Dark Studio chuyên nghiệp.

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%20CUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v4-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Bounded%20Streaming-007808?logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)
[![Tests](https://img.shields.io/badge/Tests-45%2F45%20Passed-brightgreen)](tests/)

---

## 📌 Hiện trạng dự án

- **Dual-Mode Studio:** Hỗ trợ đầy đủ cả **📷 Image Studio (Ảnh đơn x4)** và **🎬 Video Studio (Video x4)** bằng checkpoint AI thật `best.pth` (RealBasicVSR Anime V3) trên GPU NVIDIA GeForce GTX 1650 (CUDA) và CPU.
  - **Frontend:** React 19 + Vite 8 + Tailwind CSS v4, giao diện Dark Studio cao cấp, hỗ trợ chuyển đổi mượt mà giữa Image Studio và Video Studio.
    - *Image Studio:* Kéo thả ảnh, thanh trượt so sánh Before/After chia đôi, tự động nhận diện giới hạn, tải ảnh PNG x4.
    - *Video Studio:* Kéo thả tệp MP4 CFR, tự động probe thông số (resolution, fps, audio), nút nạp nhanh video Anime mẫu (2s) chỉ với 1 click, thanh tiến trình SSE thời gian thực (frames, chunks, ETA), và **Trình phát video đồng bộ (Synchronized Dual Player)** điều khiển Play/Pause/Seek đồng thời giữa video gốc và video x4.
  - **Backend:** FastAPI + PyTorch cung cấp đầy đủ REST API ảnh và video (`/api/health`, `/api/models`, `/api/upscale`, `/api/video/probe`, `/api/video/upscale` 202, `/api/video/jobs/{id}`, `/api/video/jobs/{id}/stream`, `/api/video/results/{id}`).
  - **Lõi xử lý Video (Bounded Memory Streaming):** FFmpeg giải mã CFR passthrough ra PNG, thuật toán sliding window buffer (chunk 7 hoặc 12, overlap 2) giải phóng RAM ngay khi hoàn tất từng frame, và FFmpeg mã hóa H.264 `yuv420p` tương thích 100% mọi trình duyệt web kèm ghép âm thanh AAC gốc.
  - **Kiểm thử tự động:** **45/45 bài tests đạt 100%** (19 bài API ảnh, 8 bài model PyTorch CPU/CUDA, 9 bài video engine, 9 bài video API).

---

## 🛠️ Yêu cầu môi trường

- **Hệ điều hành:** Windows / Linux / macOS
- **Python:** phiên bản tương thích với PyTorch (đã kiểm tra trên Python 3.13 và PyTorch 2.14.0+cu126).
- **Node.js:** `^20.19.0` hoặc `>=22.12.0` (đã kiểm tra trên Node.js v22).
- **FFmpeg & FFprobe:** Đã cài đặt trong hệ thống (đã kiểm tra FFmpeg 8.0.1 hỗ trợ `libx264` và `aac`).
- **Phần cứng:** NVIDIA GPU (khuyên dùng, ví dụ GTX 1650 4 GB VRAM) hoặc CPU Fallback.

---

## 🚀 Hướng dẫn khởi chạy ứng dụng (Local Development)

### 1. Khởi động Backend (FastAPI)

1. Mở terminal và chuyển đến thư mục gốc của dự án:
   ```powershell
   cd d:\Hoclaptrinh\thi_giac_may_tinh
   ```

2. **Chạy với tăng tốc GPU NVIDIA (Khuyên dùng khi có card rời):**
   ```powershell
   .\.venv_cuda\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
   ```

3. **Chạy ở chế độ CPU Fallback (Khi không có GPU):**
   ```powershell
   python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
   ```

   *Tài liệu OpenAPI Swagger tại: `http://127.0.0.1:8000/docs`.*

### 2. Khởi động Frontend (React + Vite)

1. Mở một terminal khác và chuyển đến thư mục `frontend/`:
   ```powershell
   cd d:\Hoclaptrinh\thi_giac_may_tinh\frontend
   npm.cmd install
   npm.cmd run dev
   ```
   *Mở trình duyệt truy cập: `http://localhost:5173`.*

---

## 🎬 Trải nghiệm Video Studio trên Trình duyệt

1. Truy cập `http://localhost:5173`.
2. Nhấp vào tab **🎬 VIDEO STUDIO (4X)** ở thanh Header trên cùng.
3. Nhấp vào nút **`⚡ Dùng thử mẫu Anime (2s)`** để nạp ngay clip mẫu anime 320×240.
4. Chọn cấu hình Chunk:
   - **7 frames (Mặc định - An toàn VRAM):** Tiêu thụ chỉ ~1.0 GB VRAM trên GTX 1650.
   - **12 frames (Cân bằng hiệu năng):** Tăng tốc độ ~15.5% (1.49 FPS), tiêu thụ ~2.45 GB VRAM.
5. Nhấp **🚀 Bắt Đầu Siêu Phân Giải x4**.
6. Theo dõi tiến trình thời gian thực (Trích xuất frame → RealBasicVSR x4 → Mã hóa H.264/AAC).
7. Khi hoàn tất, sử dụng **Trình phát video đồng bộ** để Play/Pause và kéo Seekbar so sánh chi tiết giữa video gốc và video x4!

---

## 💻 CLI Video Inference & Benchmark độc lập

Bạn có thể chạy siêu phân giải video độc lập qua dòng lệnh bằng script `scripts/infer_video.py`:

1. **Chạy thử trên clip mẫu Anime (cắt 2 giây):**
   ```powershell
   .\.venv_cuda\Scripts\python.exe scripts/infer_video.py --crop-sample --duration 2.0 --chunk-size 7
   ```

2. **Chạy thử trên clip tổng hợp có âm thanh (Synthetic test pattern):**
   ```powershell
   .\.venv_cuda\Scripts\python.exe scripts/infer_video.py --synthetic --duration 2.0
   ```

3. **Chạy trên tệp MP4 bất kỳ:**
   ```powershell
   .\.venv_cuda\Scripts\python.exe scripts/infer_video.py --input "duong_dan/video.mp4" --output "outputs/videos/result_x4.mp4" --chunk-size 12
   ```

### Kết quả Benchmark thực tế trên NVIDIA GeForce GTX 1650 (4 GB VRAM)

Thử nghiệm trên clip anime gốc (320×240, 29.97 fps, 60 frames -> 1280×960):

| Cấu hình Chunk | Số Chunks | Thời gian Infer | Tốc độ | Peak VRAM | Tỷ lệ VRAM (4GB) | Ghi chú |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Chunk 7 (Overlap 2)** | 12 | 46.54s | 1.29 FPS | **1021.6 MB** | 24.9% | Mặc định an toàn tuyệt đối cho mọi GPU phổ thông |
| **Chunk 12 (Overlap 2)** | 6 | 40.35s | **1.49 FPS** | **2447.2 MB** | 59.7% | Tăng tốc ~15.5%, mượt mà và tối ưu |

---

## 🧪 Kiểm thử tự động (Test Suite)

Chạy toàn bộ **45 bài kiểm thử tự động** từ thư mục gốc của dự án:

```powershell
.\.venv_cuda\Scripts\python.exe -m unittest discover tests -v
```

- **19 Image API tests:** Upload, min/max boundaries, RGBA alpha composite, EXIF, unicode filename, timeout/memory error, cleanup logic, concurrency.
- **8 Model & Pipeline tests:** Nạp strict checkpoint `best.pth`, pad divisibility by 32, single frame inference (T=1), sequence inference x4, CUDA vs CPU pixel parity.
- **9 Video Engine tests:** ffprobe metadata parsing, fractional FPS parsing, CFR validation, frame decoding, chunk sliding window calculation, FFmpeg H.264 encoding và audio muxing.
- **9 Video API tests:** `/api/health` video limits, `/api/video/probe` validation, `/api/video/upscale` 202 queuing, parameter boundary validation, job polling và error handling.

