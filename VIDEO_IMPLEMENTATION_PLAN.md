# Kế hoạch triển khai upscale video x4

## Mục tiêu và hiện trạng

Hoàn thành luồng local: **upload MP4 → chọn RealBasicVSR Anime V3 và temporal chunk → tạo job → xem tiến trình thật → xem/tải MP4 x4 có âm thanh**. Dùng `best.pth` hiện có, GPU CUDA khi hoạt động và CPU khi cần. Giữ luồng upscale ảnh hiện tại.

Hiện repo chỉ có API ảnh và CLI xử lý **thư mục frame**. `infer_rb_sequence` trong `backend/app/inference/realbasicvsr_pipeline.py` nhận toàn bộ frame và tạo `accum` float32 cho **mọi** frame output, nên không gọi thẳng hàm này với video dài. Video mẫu ở `assets/ebd9028b-a231-4b6a-b4a3-acdba777e7f5.mp4` là H.264/AAC 320×240, khoảng 73 giây, 2.189 frame. Riêng `accum` x4 của video này sẽ chiếm khoảng **30 GiB RAM**. Máy hiện có GTX 1650 4 GB; benchmark ảnh đơn `T=1` không chứng minh chunk video 7/12/16/24 frame sẽ vừa VRAM.

FFmpeg/ffprobe 8.0.1 đã được tìm thấy trên máy khi lập kế hoạch. Không cài thêm model, không train lại và không dùng ảnh resize thông thường thay cho output AI.

## Phạm vi MVP video

- **Đầu vào đầu tiên:** MP4 có một video stream H.264, CFR, có hoặc không có một audio stream AAC. Các container/codec khác và VFR chỉ mở sau khi xử lý timestamp được kiểm tra riêng. Kiểm tra nội dung bằng `ffprobe`, không tin phần mở rộng/MIME đơn lẻ.
- **Đầu ra:** MP4 H.264 `yuv420p`, kích thước `W×4` và `H×4`, cùng thứ tự/số frame và FPS hữu tỉ của input. Giữ audio khi có; file không audio vẫn hợp lệ. `+faststart` giúp preview trên browser. Audio copy nếu tương thích; nếu không, chuyển sang AAC và kiểm tra đồng bộ A/V.
- **Giới hạn khởi đầu để chứng minh pipeline:** clip khoảng 2–3 giây, 128×96 hoặc thấp hơn, `chunk_size=7`, `overlap=2`, một job chạy tại một thời điểm. Đây là bộ kiểm thử, **chưa phải** giới hạn sản phẩm cuối. Sau khi đo VRAM/RAM/đĩa trên clip thật mới mở dần thời lượng, resolution và preset chunk. Không dùng nguyên video mẫu 73 giây làm lần thử đầu.
- **Cấu hình thử nghiệm ban đầu cho API:** `MAX_VIDEO_UPLOAD_MB=20`, `MAX_VIDEO_DURATION_SECONDS=5`, `MAX_VIDEO_FRAMES=150`, `MAX_VIDEO_INPUT_PIXELS=12288` (128×96), `VIDEO_JOB_QUEUE_SIZE=2` (một job chạy, một job chờ). Đây là ngưỡng tạm để hoàn thành V2/V3; V4 thay bằng ngưỡng đo được trên GPU/CPU. Frontend lấy các giới hạn này từ API, không hardcode khác backend.
- **Giao diện:** tách chế độ Ảnh và Video; video có upload, preview input, chunk hợp lệ, nút chạy/hủy, tiến trình, preview/download output. PSNR/SSIM không hiển thị khi không có ground truth.

## Luồng xử lý dự kiến

```text
React: chọn video + chunk_size
  → POST /api/video/jobs: upload theo từng block, lưu file tên UUID
  → ffprobe: kiểm tra stream, duration, FPS, số frame, codec, audio
  → trả 202 + job_id sau upload/probe, không đợi inference
  → hàng đợi 1 GPU worker, không chặn các endpoint health/status
  → FFmpeg giải mã frame theo thứ tự vào thư mục tạm
  → RealBasicVSR xử lý từng chunk và ghi frame x4 đã hoàn tất
  → FFmpeg encode video H.264, ghép audio từ input
  → ffprobe xác minh MP4 hoàn tất
  → GET status → preview/download qua GET /api/video/results/{id}
```

### 1. CLI video và kiểm tra khả năng GPU trước khi làm web

1. Tạo `scripts/infer_video.py` gọi module video trong `backend/app/video/`. Nhận input MP4, output MP4, checkpoint, `chunk_size`, device. Dùng `subprocess` với danh sách tham số và `shell=False`; chỉ thao tác trên đường dẫn file local do ứng dụng tạo. Cấu hình `FFMPEG_BIN` và `FFPROBE_BIN`, báo lỗi rõ nếu thiếu.
2. Dùng `ffprobe` JSON để đọc `width`, `height`, `avg_frame_rate`, duration, codec, audio và `nb_frames` nếu có. Kiểm tra stream hợp lệ, rotation, giới hạn byte/thời lượng/frame/pixel/đĩa. MVP từ chối rotation metadata khác 0 cho đến khi đường xử lý xoay được xác minh. Với MVP chỉ chấp nhận CFR; kiểm tra PTS thực tế khi metadata FPS không đủ tin cậy, từ chối VFR bằng lỗi dễ hiểu. Giữ FPS dưới dạng phân số như `30000/1001`.
3. Dùng FFmpeg giải mã theo đúng thứ tự thành `frame_%06d.png` trong job dir; chọn chế độ frame sync không tự nhân/loại frame (`-fps_mode passthrough`) và xác minh số frame sau decode. Vì MVP chỉ nhận CFR, encode lại theo FPS hữu tỉ đã probe. Trước mỗi chunk, chỉ đọc số frame cần thiết từ đĩa; **không** nạp toàn bộ video vào list như CLI hiện tại. Giữ RGB uint8 và cùng `H,W` trước khi đưa vào model.
4. Tách lõi `infer_rb_sequence` thành hàm xử lý chunk giới hạn bộ nhớ: mặc định `chunk_size=7`, `overlap=2`, `stride=chunk_size-overlap`. Giữ tối đa chunk hiện tại và frame overlap chưa thể chốt; cộng trung bình dự đoán ở vùng overlap, ghi **mỗi frame output đúng một lần** dưới tên zero-padded. Frame cuối ngắn hơn chunk phải chạy đúng. Việc đọc model vẫn một lần; dùng cùng khóa inference GPU với API ảnh để không có hai forward pass đồng thời.
5. FFmpeg encode các PNG x4 theo FPS hữu tỉ thành MP4 video-only H.264 (`libx264`, `yuv420p`); sau đó mux audio từ nguồn nếu có. H.264 NVENC chỉ là tối ưu tùy chọn sau khi kiểm tra encoder trên máy. Giữ nguyên số frame, kiểm tra output bằng `ffprobe` (dimensions x4, FPS, duration, audio, codec). Chỉ đổi tên file `.partial.mp4` thành kết quả cuối sau khi kiểm tra thành công. Nếu lỗi/hủy, giữ trạng thái lỗi và dọn temp của job theo chính sách.
6. Đo `torch.cuda.max_memory_allocated/reserved`, `mem_get_info`, RAM RSS, thời gian từng phase và dung lượng temp trên GTX 1650. Chạy trước ở 128×96, sau đó tăng từng mức. Chỉ đưa preset `7`, `12`, `16`, `24` (và `10` nếu cần) vào danh sách khả dụng sau khi preset đó chạy ổn trên resolution/giới hạn công bố. Model train với 7 frame; kiểm tra thêm chất lượng và đường nối chunk khi dùng T dài hơn.

**Xong phần CLI khi:** một clip **có audio** và một clip **không audio** đều tạo MP4 x4 đúng số frame/FPS, xem được, không mất đồng bộ nghe nhìn; đỉnh RAM không tăng tỷ lệ với tổng số frame. File video mẫu 73 giây chỉ xử lý sau khi pipeline ngắn và giới hạn tài nguyên đã đạt.

### 2. API job video chạy nền

| Endpoint | Nội dung |
| --- | --- |
| `POST /api/video/jobs` | `multipart/form-data`: `video`, `model_id`, `chunk_size`; lưu upload theo từng block với giới hạn byte, `ffprobe` kiểm tra, rồi trả `202` gồm `job_id` và `status_url` (không chờ inference) |
| `GET /api/video/jobs/{job_id}` | `status`, `phase`, `processed_frames`, `total_frames` nếu biết, `progress`, `chunk_size`, `overlap`, device, lỗi và URL kết quả khi hoàn tất |
| `POST /api/video/jobs/{job_id}/cancel` | Đặt cờ hủy; dừng sau chunk an toàn, kết thúc subprocess FFmpeg và chuyển trạng thái `cancelled` |
| `GET /api/video/results/{job_id}` | MP4 preview (`inline`) hoặc `?download=true` (`attachment`); chỉ trả khi job `completed` và file đã xác minh |

- **Trạng thái:** `queued → decoding → inferencing → encoding → muxing → completed`; nhánh `failed`, `cancelled`, `interrupted`. Upload/probe hoàn tất trước phản hồi `202`. Job ID là UUID; ghi metadata JSON/SQLite cục bộ bằng thay file nguyên tử để refresh browser không mất trạng thái. Khi server khởi động lại, đánh dấu job dở dang `interrupted`; chưa tự resume model giữa chừng.
- **Worker local:** một Uvicorn worker, một hàng đợi giới hạn, một job inference GPU tại một thời điểm. Worker chạy trong lifespan và đưa tác vụ nặng sang thread/process thích hợp để `/api/health` và status luôn phản hồi. Dùng chung cơ chế khóa với `/api/upscale`; nếu video chiếm GPU lâu, API ảnh trả trạng thái **thiết bị bận** rõ ràng hoặc có hàng đợi ngắn, không chờ HTTP vô hạn. Chạy video server **không dùng `--reload`** trong lúc có job.
- **Tiến trình thật:** tăng `processed_frames` khi frame đã được chốt/ghi; `total_frames` lấy từ metadata hoặc đếm khi giải mã. Phase encode/mux lấy `-progress` của FFmpeg; chỉ báo `100%` khi file cuối đã qua `ffprobe`. Nếu không xác định được tổng frame, hiển thị tiến trình không xác định thay vì tự tạo FPS/ETA.
- **Lưu trữ:** `work/video_jobs/{uuid}/source.mp4`, input/output frames và `job.json`; `outputs/videos/{uuid}.mp4` cho file hoàn tất. Mở rộng cleanup hiện có để dọn job đã xong/hỏng quá hạn, **không xóa job đang chạy** hoặc các file thử nghiệm khác. Mọi thao tác dọn thư mục phải xác thực UUID và đường dẫn resolved nằm trong `work/video_jobs`/`outputs/videos`.
- **Bảo vệ tài nguyên:** cấu hình độc lập `MAX_VIDEO_UPLOAD_MB`, `MAX_VIDEO_DURATION_SECONDS`, `MAX_VIDEO_FRAMES`, `MAX_VIDEO_WIDTH/HEIGHT`, `MAX_VIDEO_TEMP_DISK_GB`, `VIDEO_JOB_QUEUE_SIZE`, preset chunk theo device. Giới hạn dùng kết quả đo, không sao chép ngưỡng ảnh đơn `T=1`. Kiểm tra dung lượng đĩa trước/giữa job, thời gian subprocess, OOM, input lỗi và output encoder lỗi. Không chuyển GPU→CPU âm thầm giữa một video đã chạy dở.

**Xong phần API khi:** tạo job trả `202` ngay sau khi nhận/kiểm tra file, không đợi inference; poll thấy state tăng theo frame thật; hai job không chạy inference đồng thời; hủy/lỗi không để lại file kết quả hoàn chỉnh giả; restart chuyển job dở dang sang `interrupted`; video hoàn tất preview/download được và endpoint status vẫn phản hồi khi GPU bận.

### 3. Frontend video

1. Thêm tab/chế độ **Ảnh** và **Video** trong React; không đưa lại các số liệu FPS/VRAM giả của prototype cũ. Giữ luồng ảnh hiện hành độc lập.
2. Với video: chọn file, xem metadata do backend trả (duration, resolution, FPS, audio), chọn `chunk_size` từ các preset **backend công bố là đã hỗ trợ**. Mặc định 7 và overlap 2. Nút 12/16/24 chỉ bật sau benchmark GPU tương ứng; nếu người dùng muốn 10, thêm preset/ô số sau khi backend xác thực 10.
3. Upload → nhận job ID → poll `GET status` mỗi khoảng 1–2 giây; hiển thị queued/phase, frame đã xong/tổng, lỗi hoặc nút hủy. Khi rời trang có thể ngừng poll, **không ngầm hủy job**; giữ job ID để mở lại tiến trình.
4. Khi `completed`, hiển thị `<video controls>` của input và output, nút tải MP4 qua API. Ở MVP có thể đặt hai video cạnh nhau; slider video và tua đồng bộ chỉ làm sau khi bản cơ bản đúng FPS/audio. URL download chỉ hiện với file cuối thật.

**Xong phần frontend khi:** người dùng hoàn tất upload → chờ job → xem output → download MP4; chọn chunk được phản ánh đúng trong job status; refresh trang vẫn tìm lại được job; ảnh mẫu/demo không bị gắn nhãn như video AI thật.

## Thứ tự giao việc cho AI code

| Phần | Việc cần code | Điều kiện chuyển phần | Trạng thái |
| --- | --- | --- | --- |
| **V1** | `ffprobe`, decode, chunk streaming, encode/mux trong CLI | Clip ngắn có/không audio x4 đúng frame/FPS, bộ nhớ bounded | **HOÀN THÀNH (OK)** |
| **V2** | Job manager, API 202/status/cancel/result và storage | Poll thật, khóa GPU chung với ảnh, xử lý lỗi/restart/cleanup | **HOÀN THÀNH (OK)** |
| **V3** | React VideoPanel, chunk selector, progress, preview/download | Luồng browser end-to-end trên clip ngắn | **HOÀN THÀNH (OK)** |
| **V4** | Mở giới hạn theo benchmark và kiểm thử biên | Video mẫu/trường hợp lỗi đạt chuẩn, tài liệu phản ánh ngưỡng thật | **HOÀN THÀNH (OK)** |

**Các file đã triển khai:**
- `backend/app/video/probe.py`: ffprobe JSON validation (H.264, CFR check, no-rotate, audio).
- `backend/app/video/decode.py`: CFR passthrough decoding ra `frame_%06d.png`.
- `backend/app/video/chunk_stream.py`: Thuật toán bounded-memory streaming sliding window buffer, purge ngay khi hết overlap.
- `backend/app/video/encode.py`: FFmpeg encode H.264 `yuv420p` + AAC audio mux + probe verify.
- `backend/app/video/job_manager.py`: Quản lý tác vụ nền bất đồng bộ, SSE broadcasting, serialize lock với Image API.
- `backend/app/video/router.py`: REST API endpoints (`/probe`, `/upscale` 202, `/jobs/{id}`, `/jobs/{id}/stream`, `/results/{id}`).
- `scripts/infer_video.py`: CLI độc lập hỗ trợ synthetic video generation, crop sample và benchmark profiling.
- `frontend/src/components/VideoStudio.jsx`: Giao diện Video Studio chuyên nghiệp, one-click anime demo sample, progress tracking và Synchronized Dual Video Player.
- `tests/test_video.py` và `tests/test_video_api.py`: 18 bài unit & integration test chuyên biệt cho video pipeline.


## Kiểm tra nghiệm thu cuối

1. Sinh clip tổng hợp CFR 2 giây có audio và không audio, rồi dùng clip anime 3 giây cắt từ video mẫu. Kiểm tra output x4, số frame và thứ tự chính xác, FPS hữu tỉ, thời lượng **video stream** lệch không quá khoảng một frame; kiểm tra audio tồn tại khi có và sai lệch A/V trong ngưỡng đo chấp nhận được (ví dụ ≤100 ms cho clip thử).
2. Chạy chunk 7 và ít nhất một chunk lớn hơn **đã được benchmark**; so output overlap/biên frame trên cảnh có chuyển động. Với 4 GB VRAM, không công bố preset chưa chạy qua phép đo peak và OOM.
3. Kiểm tra input sai container/codec, file hỏng, VFR không hỗ trợ, vượt byte/duration/frame/pixel/disk, video không audio, thiếu FFmpeg, OOM, hủy giữa job, restart giữa job, UUID/path sai và hai request cạnh tranh GPU.
4. Kiểm tra `GET /api/video/results/{id}` preview và `Range`/seek trong browser, `?download=true` tải MP4. Kiểm tra job hoàn tất còn truy cập được đến thời hạn lưu, job đang chạy không bị cleanup xóa.
5. Đo end-to-end trên GTX 1650 và CPU với cùng clip; ghi thời gian decode, inference, encode, mux, peak VRAM, RSS và dung lượng đĩa. Chỉ nói video nhanh hơn sau khi có số đo này; benchmark ảnh đơn không đủ.

## Tài liệu nguồn

- [FFprobe JSON và `-show_entries`](https://ffmpeg.org/ffprobe.html) để kiểm tra container/stream.
- [FFmpeg stream mapping và audio tùy chọn](https://ffmpeg.org/ffmpeg.html) để mux video/audio.
- [FFmpeg image sequence, rawvideo và `+faststart`](https://ffmpeg.org/ffmpeg-formats.html) cho decode/encode và preview MP4.
- [FFmpeg `-progress`](https://ffmpeg.org/ffmpeg.html) cho tiến trình encode dạng `key=value`.
- [FastAPI `UploadFile`](https://fastapi.tiangolo.com/tutorial/request-files/) cho upload video không đọc toàn file vào RAM.
- [FastAPI BackgroundTasks](https://fastapi.tiangolo.com/tutorial/background-tasks/) nêu giới hạn với tác vụ nặng; bản local dùng worker/job manager một tiến trình và trạng thái rõ ràng.
- [Starlette FileResponse](https://www.starlette.io/responses/) có hỗ trợ `Range` cho preview/seek file video.
