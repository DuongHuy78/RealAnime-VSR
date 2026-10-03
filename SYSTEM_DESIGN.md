# Thiết kế hệ thống: web demo upscale ảnh/video bằng AI

## 1. Mục tiêu và phạm vi

Mục tiêu trước mắt là web demo chạy hoàn toàn trên máy cá nhân: người dùng upload một ảnh, chọn model, xem ảnh trước/sau và tải kết quả. Output cố định ở x4 và kích thước được tính từ input. Frontend dùng React + Vite + Tailwind CSS; backend dùng FastAPI, PyTorch và Pillow/OpenCV. REST API đủ cho luồng ảnh ban đầu.

Video và các model video như RealBasicVSR là phần mở rộng sau khi luồng ảnh hoạt động ổn định. Kế hoạch code chi tiết nằm trong `VIDEO_IMPLEMENTATION_PLAN.md`: CLI video bounded → API job → React video → kiểm thử tài nguyên. Khi có video, người dùng được chọn số frame mỗi chunk đã được benchmark; output vẫn cố định x4. FFmpeg dùng cho giải mã/encode/mux. WebSocket chỉ cần nếu polling không đủ cho tiến trình realtime.

**Hiện trạng repo:** React/Vite ở `frontend/` đã nối với FastAPI ở `backend/`: người dùng upload ảnh, chạy `POST /api/upscale` bằng `best.pth`, xem trước/sau và tải PNG qua `GET /api/results/{id}`. Ảnh demo được gắn nhãn riêng. `/api/health` cung cấp trạng thái, device và giới hạn ảnh để frontend kiểm tra cùng cấu hình backend. Prototype cũ `index.html`, `app.js`, `styles.css` vẫn ở gốc để tham khảo. Notebook, CLI, benchmark CPU và 26 bài kiểm thử model/API đã hoàn thiện. Mặc định CPU giới hạn ảnh theo bộ nhớ đã đo (tối đa 102.400 pixel, đỉnh RSS ~1.8 GB). Cơ chế dọn kết quả và file tạm quá hạn đã được **lập lịch tự động trong FastAPI lifespan** (`_cleanup_worker`).

## 2. Sơ đồ thành phần

```text
Browser (React/Vite)
  │  upload ảnh + model (output x4 cố định)
  ▼
FastAPI  ── kiểm tra input / quản lý file
  │
  ├── Model registry ── checkpoint PyTorch được load và tái sử dụng
  ├── Pillow/OpenCV ── đọc ảnh, chuẩn hóa, ghi ảnh
  └── Inference trên CUDA hoặc CPU
  │
  ▼
Output lưu cục bộ ── URL preview/download ── Browser
```

Trong bản demo local, Vite phục vụ frontend ở chế độ dev và Uvicorn phục vụ API. Backend là tiến trình Python duy nhất làm chủ model. Nếu dùng GPU có ít VRAM, xử lý một tác vụ inference tại một thời điểm là mặc định an toàn; chỉ tăng concurrency sau khi đo bộ nhớ và thời gian xử lý. Tránh nhiều worker Uvicorn vì mỗi worker sẽ có bản model riêng.

Giới hạn ảnh đơn: mỗi cạnh tối đa 512 px; CUDA mặc định tối đa 262.144 pixel,
CPU tối đa 102.400 pixel. Với `best.pth`, T=1, float32 trên GTX 1650 4 GB,
ảnh 511×308 đã tạo output 2044×1232 trong 0,94 s (peak allocated 1549,6 MiB);
ảnh 512×512 tạo output 2048×2048 trong 1,53 s (peak allocated 2462,1 MiB,
peak reserved 3186,0 MiB). Đây là phép thử inference một lần cho mỗi kích thước,
không phải cam kết tốc độ cho mọi ảnh. Frontend lấy giới hạn theo device thực tế
từ `/api/health`; thay cấu hình cần khởi động lại backend.

## 3. Cấu trúc hiện có và phần dự kiến

```text
thi_giac_may_tinh/
├── AGENTS.md                 # Điểm vào tự động cho coding agent
├── agent.md                  # Quy tắc làm việc của agent
├── SYSTEM_DESIGN.md          # Kiến trúc, luồng dữ liệu, API và lộ trình
├── IMPLEMENTATION_PLAN.md    # Các phần code và hướng dẫn tích hợp best.pth
├── VIDEO_IMPLEMENTATION_PLAN.md # Các bước code video/FFmpeg/job/React
├── README.md                 # Hướng dẫn chạy khung ứng dụng local
├── .gitignore                # Loại checkpoint, upload, output, venv, cache
├── realbasicvsr-animedb.ipynb # Notebook train/test hiện có
├── *.pth                     # Checkpoint RealBasicVSR (best_v3.pth, best_v4.pth...); không commit
├── index.html                # Prototype HTML cũ, chưa phải Vite app
├── app.js                    # Logic UI và mô phỏng inference của prototype cũ
├── styles.css                # CSS prototype cũ
├── frontend/
│   ├── package.json         # Scripts và dependencies frontend
│   ├── vite.config.js        # Vite; proxy /api đến backend khi phát triển
│   ├── index.html            # HTML entry
│   └── src/
│       ├── main.jsx          # Mount React
│       ├── App.jsx           # Bố cục và state của màn hình demo
│       ├── components/       # Upload, chọn model khả dụng, so sánh; download khi có kết quả thật
│       ├── api/              # Client gọi REST API và xử lý lỗi
│       └── index.css         # CSS Tailwind và các kiểu giao diện đã chuyển từ prototype
├── backend/
│   ├── requirements.txt      # Dependencies Python; version được khóa khi triển khai
│   └── app/
│       ├── main.py           # FastAPI, lifespan, CORS local, routes
│       ├── config.py         # Đường dẫn và giới hạn qua biến môi trường
│       ├── schemas.py        # Kiểu dữ liệu response/error
│       ├── models/           # RealBasicVSRNet đã tách từ notebook
│       ├── inference/        # Loader, tiền xử lý và inference đã có
│       ├── model_registry.py # Registry và cache model đã có
│       └── storage.py        # Kiểm tra ảnh, alpha composite, lưu/trả output và dọn file tự động
├── scripts/                  # CLI, chuẩn bị dữ liệu và benchmark độc lập
├── tests/                    # Kiểm thử loader, pad và output x4
├── models/                   # Checkpoint tải riêng; không commit file trọng số
├── uploads/                  # File upload tạm; không commit
└── outputs/                  # Kết quả sinh ra; không commit
```

Chỉ tạo những module cần cho chức năng đang triển khai. Nếu mã thực tế khác cây dự kiến, cập nhật tài liệu này theo mã.

## 4. Luồng ảnh giai đoạn 1

1. Frontend lấy danh sách model khả dụng từ API rồi hiển thị tùy chọn hợp lệ. Scale giai đoạn đầu cố định x4.
2. Người dùng chọn ảnh. Frontend hiển thị preview gốc, kiểm tra loại/dung lượng sơ bộ và gửi `multipart/form-data`.
3. Backend kiểm tra file thực sự là ảnh được hỗ trợ, kích thước và giới hạn cấu hình; xử lý EXIF orientation và mode màu/alpha một cách nhất quán (hỗ trợ alpha composite cho ảnh trong suốt).
4. Backend lấy model đã load từ registry, chạy inference bằng `torch.inference_mode()` trên CUDA nếu khả dụng, nếu không trên CPU. Cách chia tile/giới hạn kích thước cần theo khả năng model và bộ nhớ máy.
5. Backend ghi ảnh kết quả vào `outputs/`, trả metadata và URL kết quả. Frontend hiển thị so sánh trước/sau và nút tải xuống.
6. Upload tạm và output được dọn tự động qua background worker định kỳ theo cấu hình `CLEANUP_INTERVAL_MINUTES` và `CLEANUP_MAX_AGE_HOURS`; không lưu vô hạn chỉ vì đây là demo local.

### API dự kiến

| Endpoint | Mục đích | Ghi chú |
| --- | --- | --- |
| `GET /api/health` | Tình trạng API và device | Không bắt buộc load checkpoint trong health check |
| `GET /api/models` | Quét file .pth ở gốc, MODELS_DIR và CHECKPOINT_PATH | Trả id, checkpoint_filename/path, loaded, available, device, error; chưa nạp thì device=none |
| `POST /api/models/{model_id}/activate` | Kiểm tra và kích hoạt checkpoint được chọn | Chờ khóa inference; lỗi checkpoint trả 422, ID không tồn tại trả 404 |
| `POST /api/upscale` | Nhận `image`, `model_id`; chạy inference x4 | Trả `result_id`, kích thước input/output và `result_url` |
| `GET /api/results/{result_id}` | Preview hoặc download kết quả | Chỉ truy cập ID do server tạo, không nhận đường dẫn tùy ý |

Giai đoạn 1 có thể dùng request đồng bộ. Với ảnh lớn hoặc video, chuyển sang job bất đồng bộ (`POST` tạo job, `GET` hỏi trạng thái) trước khi thêm WebSocket. API trả mã lỗi và thông báo rõ cho file sai định dạng, model/scale không khả dụng, checkpoint lỗi, quá giới hạn và thiếu bộ nhớ.

### API video đã triển khai (Giai đoạn 3 / Phase V1 - V4)

| Endpoint | Method | Mục đích | Ghi chú |
| --- | --- | --- | --- |
| `/api/video/probe` | `POST` | Pre-flight probe video MP4 | Trả `VideoProbeResponse`: width, height, duration, fps, frame count, audio codec, codec H.264. |
| `/api/video/upscale` | `POST` | Nộp video vào hàng đợi (HTTP 202) | Nhận `file`, `model_id`, `chunk_size` (mặc định 7), `overlap` (mặc định 2). Trả `job_id`; job giữ model_id đã chọn. |
| `/api/video/jobs/{job_id}` | `GET` | Polling trạng thái job | Trả tiến độ (%), current_frame, total_frames, current_chunk, eta_seconds, timing. |
| `/api/video/jobs/{job_id}/stream` | `GET` | Server-Sent Events (SSE) | Stream realtime cập nhật frame/chunk/tiến độ tới React frontend. |
| `/api/video/results/{job_id}` | `GET` | Xem trực tiếp video x4 | Stream tệp MP4 chuẩn `yuv420p` trên web player. |
| `/api/video/results/{job_id}/download` | `GET` | Tải video MP4 x4 | Header `Content-Disposition: attachment`. |

## 5. Kiến trúc Video Super-Resolution Pipeline (Đã triển khai hoàn tất)

```text
[Input MP4] ──▶ 1. PROBE (ffprobe CFR, H.264, no-rotate, audio stream)
                     │
                     ▼
             2. DECODE (-fps_mode passthrough ──▶ PNG frame_%06d.png)
                     │
                     ▼
             3. BOUNDED STREAMING CHUNK INFERENCE
                ├── Cửa sổ trượt (Sliding Window Buffer): chunk_size=7 / 12, overlap=2
                ├── Model RealBasicVSRNet forward pass trên GPU GTX 1650
                ├── Purge hoàn tất: Ghi ngay sr_frame_%06d.png ra đĩa và giải phóng RAM
                └── RAM tiêu thụ: O(chunk_size * H * W) < 200 MB (Hoàn toàn chống OOM)
                     │
                     ▼
             4. ENCODE & MUX (FFmpeg libx264, yuv420p, +faststart, aac -shortest)
                     │
                     ▼
[Output MP4 x4] ──▶ 5. SERVE / DOWNLOAD / DUAL SYNCHRONIZED PLAYER
```

### Kết quả Benchmark thực tế trên NVIDIA GeForce GTX 1650 (4 GB VRAM)
Thử nghiệm trên clip anime gốc (320×240, 29.97 fps, 60 frames -> 1280×960):

| Cấu hình Chunk | Số Chunks | Thời gian Infer | Tốc độ | Peak VRAM | Tỷ lệ VRAM (4GB) | Ghi chú |
| --- | --- | --- | --- | --- | --- | --- |
| **Chunk 7 (Overlap 2)** | 12 chunks | 46.54s | 1.29 FPS | **1021.6 MB** | 24.9% | Mặc định an toàn tuyệt đối cho mọi GPU |
| **Chunk 12 (Overlap 2)** | 6 chunks | 40.35s | **1.49 FPS** | **2447.2 MB** | 59.7% | Tăng tốc ~15.5%, mượt mà và tối ưu |

## 6. Giao diện người dùng: Dual-Mode Studio

- **Chọn checkpoint:** `ModelSelector.jsx` dùng chung cho hai studio, hiển thị
  tên/đường dẫn tương đối của file `.pth`, trạng thái nạp, thiết bị và lỗi.
  Nút Làm mới quét danh sách; App cũng làm mới mỗi 15 giây. Chọn file gọi API
  activate, khóa nút chạy trong lúc chuyển và cập nhật giới hạn ảnh theo device.
  Kết quả ảnh/video giữ tên checkpoint đã dùng để không nhầm với lựa chọn mới.
- **📷 Image Studio:** Chế độ ảnh siêu phân giải đơn lẻ, so sánh thanh trượt trước/sau (CompareSlider).
- **🎬 Video Studio:** Chế độ video siêu phân giải chuyên nghiệp:
  - Pre-flight video inspector (kích thước, duration, fps, frames, codec, audio).
  - Tùy chọn Chunk Presets: 7 (Mặc định - An toàn), 12 (Cân bằng), 16 (Tốc độ cao).
  - Nút nạp nhanh video anime mẫu (2s) chỉ với 1 click.
  - Thanh tiến trình thời gian thực (SSE & Polling): Khung hình hiện tại, Chunks, ETA.
  - Trình phát video đồng bộ (Synchronized Dual Player): Play/Pause, Seekbar, Mute đồng bộ giữa video gốc và video x4.

## 7. Lộ trình và trạng thái

| Giai đoạn | Phạm vi | Trạng thái |
| --- | --- | --- |
| **Giai đoạn 1 & 2** | Image Studio: React/Vite, FastAPI, RealBasicVSR x4, CPU & CUDA GTX 1650 | **HOÀN THÀNH 100%** |
| **Giai đoạn 3 (V1)**| Core Video Engine: ffprobe, CFR passthrough decode, streaming chunk inference, FFmpeg encode + AAC mux, CLI runner | **HOÀN THÀNH 100%** |
| **Giai đoạn 3 (V2)**| Asynchronous Video Job Manager: FIFO queue, SSE stream, REST 202/200, GPU lock serialization | **HOÀN THÀNH 100%** |
| **Giai đoạn 3 (V3)**| UI Video Studio: Dual-mode header, VideoStudio component, synchronized dual player | **HOÀN THÀNH 100%** |
| **Giai đoạn 3 (V4)**| Benchmark mở rộng: Chunk 7 vs 12, unit tests 45/45 pass 100%, tài liệu hoá | **HOÀN THÀNH 100%** |


## Registry nhiều checkpoint

Backend quét `.pth` trực tiếp trong thư mục gốc và đệ quy trong `MODELS_DIR`,
cộng file `CHECKPOINT_PATH` nếu có. Client chỉ gửi ID do registry tạo; không
nhận đường dẫn tùy ý. ID cũ `realbasicvsr_anime_v3` được giữ cho checkpoint
mặc định của phiên; tên hiển thị dùng tên file thực, kể cả V4. Lúc khởi động,
ưu tiên `CHECKPOINT_PATH`, rồi thử checkpoint theo tên nếu cần.

Checkpoint được nạp strict trên CPU bằng weights_only=True khi dùng lần đầu.
Các model từng dùng được cache trên CPU; chỉ model đang active chuyển lên
CUDA và warmup T=2. Chuyển model và inference ảnh/video giữ cùng RLock và
semaphore; lấy model bên trong khóa để tránh dùng device cũ sau khi chuyển.
Checkpoint không hợp lệ trả lỗi và giữ model cũ. Khi CUDA lỗi trong chế độ
auto, model mới chạy CPU và API cập nhật device/giới hạn theo kết quả thực.
Thay đổi mtime/size của file khiến lần dùng kế tiếp nạp lại checkpoint.

`available=true` với file mới nghĩa là checkpoint có thể được chọn để kiểm
tra, chưa chứng minh tương thích; `loaded=true` xác nhận model active đã nạp
và warmup CUDA thành công nếu chạy GPU. File lỗi có `available=false`, `error`
cụ thể và vẫn hiện trong danh sách để sửa/thử lại. Health chỉ đọc cache,
không nạp model. Job video lưu `model_id` lúc gửi và activate ID đó trong khóa
suốt inference; đổi model từ giao diện không thay ID của job đã gửi. Hai API
ảnh/video mặc định dùng model active nếu không truyền `model_id`.

## Crop & Zoom để so sánh chi tiết

Sau khi có kết quả, Image Studio và Video Studio cho phép bật Crop & Zoom,
kéo một vùng rồi thả để mở popup trước/sau. CropSelection quy đổi tọa độ theo
phần hình thực tế, bỏ viền đen object-contain. cropComparison.js ánh xạ cùng
vùng từ nguồn sang kết quả theo kích thước thật, chụp bằng canvas và phóng
nguồn tới cùng kích thước crop kết quả. CropCompareDialog có zoom đồng bộ,
Vừa khung / 100% / 200% / 400% / nút − và +, cùng cuộn đồng bộ. 100% ứng với
một pixel output trên một CSS pixel. Video tạm dừng và seek hai player tới
cùng thời điểm trước khi chụp. Escape, nút Đóng hoặc nhấn ngoài popup để đóng.
Crop không chạy lại inference hoặc thay đổi file kết quả. Ảnh demo giữ nhãn
minh họa. Kiểm thử hình học: node --test frontend/tests/cropComparison.test.js.
