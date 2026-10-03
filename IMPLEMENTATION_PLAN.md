# Kế hoạch hoàn thành web demo upscale AI

## Đích đầu tiên: MVP ảnh x4 chạy local

Người dùng chọn một ảnh, thấy kích thước input và output tự tính `rộng × 4`, `cao × 4`, chạy một checkpoint super resolution x4 thật, so sánh trước/sau và tải ảnh kết quả. Backend báo rõ model và device thực sự dùng. Một ảnh upload không có ground truth nên không hiển thị PSNR/SSIM.

Repo đã có [notebook train/test](realbasicvsr-animedb.ipynb) và checkpoint [best.pth](best.pth). Đây là **RealBasicVSR Anime V3 cho chuỗi frame**, không phải checkpoint Real-ESRGAN. Phần 2 đã chứng minh checkpoint chạy độc lập theo kiến trúc notebook, kể cả tensor ảnh đơn `T=1`. Notebook huấn luyện trên cửa sổ 7 frame; đánh giá chất lượng ảnh đơn mới là thử nghiệm trên một số crop anime, nên chỉ công bố model đúng phạm vi anime và tiếp tục xem kết quả thực trước khi mở rộng sang ảnh tổng quát.

**Trạng thái hiện tại:** MVP ảnh local Phần 1–5 và nhánh CUDA 5A đã triển khai. Frontend React/Vite gửi ảnh tới FastAPI; RealBasicVSRNet nạp `best.pth` strict trên `cuda:0` trong `.venv_cuda` hoặc CPU ở môi trường gốc. API có giới hạn input, alpha/EXIF, khóa inference và dọn kết quả UUID quá hạn. Sau rà soát, bộ kiểm thử CUDA có 27 bài đạt; bản CPU có 26 bài đạt và bỏ qua bài dành riêng cho CUDA. Live server CUDA đạt 8/8 kịch bản. Kết quả tăng tốc và chất lượng là số đo trên máy/ảnh thử, không phải cam kết cho mọi input. Phần mở rộng tiếp theo là video hoặc đánh giá model trên nhiều ảnh hơn.

**Giới hạn của bằng chứng Phần 2:** ảnh đơn `T=1` chạy thật và một crop khuôn mặt anime nhìn ổn khi đối chiếu Bicubic; chưa có ground truth hoặc tập ảnh đa dạng để chứng minh chất lượng tổng quát. Phép so sánh `0.17/255` trong báo cáo ban đầu dùng vùng nền gần đồng nhất. Khi kiểm tra lại vùng có mắt/đường viền với 5 frame dịch nhẹ, chênh lệch T=1/T=5 là `0.52/255` (một ví dụ, không phải bảo đảm cho mọi video). Benchmark cũ dùng `tracemalloc` chỉ đo Python heap, bỏ sót phần lớn bộ nhớ PyTorch. Benchmark RSS trên CPU local đạt đỉnh khoảng **1.7–1.9 GB** ở input 320×320; không lấy số 38.7 MB hoặc giới hạn 512 px/timeout 60 giây trong báo cáo làm thông số an toàn mặc định.

## Thứ tự các phần để code

### Phần 1 — Chuẩn bị repo và chốt giao diện MVP

- Tạo `frontend/` với React + Vite + Tailwind CSS và `backend/` với FastAPI; thêm `.gitignore`, cấu hình môi trường mẫu và lệnh chạy local.
- Chuyển các asset và phong cách giao diện còn dùng từ prototype sang frontend. Giữ một màn hình ảnh với upload, model khả dụng, input/output x4, trước/sau và download. Ẩn hoặc loại các màn hình video, batch, model zoo, thông số GPU giả và chỉ số chất lượng giả khỏi luồng MVP.
- Sửa `README.md` theo trạng thái thực; không mô tả demo là đã inference được trước khi model hoạt động.

**Xong khi:** frontend mở được qua Vite, backend trả `GET /api/health`, và giao diện không đưa dữ liệu mô phỏng ra như kết quả AI thật.

### Phần 2 — Chạy `best.pth` độc lập trước khi làm API

- Tách định nghĩa `RealBasicVSRNet` từ cell kiến trúc trong notebook thành module Python; nạp `best.pth` theo mục **Hướng dẫn tích hợp checkpoint** bên dưới. Không dùng loader Real-ESRGAN hay đổi tên model trong UI.
- Làm CLI inference cho một chuỗi frame anime hoặc clip ngắn theo đúng notebook: RGB → float `[0,1]` → tensor `[B,T,C,H,W]` → pad → model → crop x4 → PNG. Dùng CUDA khi khả dụng, còn lại CPU; model load một lần, `eval()` + `torch.inference_mode()`.
- Thử ảnh đơn với `T=1` và so sánh chất lượng trực quan. Nếu không phù hợp, ghi rõ kết quả và thêm một image SR checkpoint riêng trước khi nối API ảnh. Giữ chế độ video của `best.pth` theo chuỗi frame, không xử lý từng frame độc lập.
- Giới hạn resolution/độ dài theo RAM và VRAM thực tế; không sao chép cách notebook giữ toàn bộ output video trong RAM cho file dài.

**Xong khi:** CLI nạp `best.pth` với `strict=True`, xử lý được một chuỗi frame ngắn bằng weight thật và tạo output đúng `4 × chiều rộng`, `4 × chiều cao`; có kết luận rõ về chất lượng chế độ ảnh đơn. Nếu checkpoint thiếu/lỗi, chương trình báo lỗi rõ, không tạo ảnh resize giả.

#### Hướng dẫn tích hợp checkpoint `best.pth` cho AI code

**Nguồn chính xác:** [realbasicvsr-animedb.ipynb](realbasicvsr-animedb.ipynb), phần test: cell **3** (định nghĩa kiến trúc), **5** (`load_rb_finetuned`), **6** (`read_rgb`, `pad_lq`, `infer_rb_sequence`). Cell kiến trúc ở phần train và test giống nhau. Notebook dùng đường dẫn Kaggle `/kaggle/working/runs_realbasicvsr_anime_v3/best.pth`; bản local hiện nằm ở `./best.pth`, nên đường dẫn phải lấy từ cấu hình backend, không hardcode đường dẫn Kaggle. Không cần dataset train hoặc optimizer để chạy inference.

**Checkpoint đã đối chiếu:** `best.pth` là dict có `epoch`, `net`, `optimizer`, `scheduler`, `val_psnr`, `best_val_psnr`, `config`. `epoch=19`, `config.num_frame=7`, `config.num_block=20`, `net` có 320 tensor và không có tiền tố `module.`. `best_val_psnr≈32.499 dB` là chỉ số validation lúc train, **không phải** PSNR của ảnh người dùng upload. SHA-256 của file local: `A2EFFFD0171C5620A3E69CCDC58C1A0EA24BEAC0B991BAC78B63BDC53B892173`.

1. Tách nguyên các lớp `ResidualBlockNoBN`, `ResidualBlocksWithInputConv`, `PixelShufflePack`, `ConvModule`, `SPyNetBasicModule`, `SPyNet`, `RealBasicVSRNet` từ cell kiến trúc vào một module, giữ tên layer để key trong checkpoint khớp. Tạo `RealBasicVSRNet(mid_channels=64, num_blocks=20)`.
2. Nạp **`ckpt["net"]`**, không đưa cả checkpoint vào `load_state_dict`. Có thể dùng `torch.load(path, map_location="cpu", weights_only=True)` với file local này; cách đọc này đã mở được checkpoint. `net.load_state_dict(state, strict=True)`, rồi `net.to(device).eval()`. Nếu checkpoint khác về sau có tiền tố `module.`, chỉ bỏ tiền tố đó sau khi kiểm tra key thực tế. Không nạp `optimizer`/`scheduler` khi phục vụ API.

   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
   net = RealBasicVSRNet(mid_channels=64, num_blocks=20)
   net.load_state_dict(ckpt["net"], strict=True)
   net = net.to(device).eval()
   ```

3. Đọc frame dưới dạng RGB `uint8` cùng kích thước. Notebook dùng OpenCV `BGR → RGB`; nếu backend dùng Pillow thì xử lý EXIF orientation và chuyển RGB tương đương. Ghép frame thành `float32 / 255.0`, thứ tự `[1,T,3,H,W]`, chuyển sang device. **Không** chạy hàm tạo suy giảm/downgrade GT, không resize LQ xuống 160/256, không tự chuẩn hóa ImageNet bên ngoài; SPyNet đã có `mean/std` bên trong model.
4. Pad cạnh phải/dưới để `H,W` chia hết cho 32 bằng `pad_lq` (notebook dùng `reflect`; ảnh quá nhỏ cần giới hạn tối thiểu hoặc kiểu pad phù hợp). Chạy model trong `torch.inference_mode()`, `clamp(0,1)`, crop bỏ padding về `H*4 × W*4`, chuyển `[T,H*4,W*4,3]` RGB `uint8` và lưu PNG/BGR theo thư viện ghi file. Model đã tự upscale x4 bên trong, không resize thêm bên ngoài.
5. Với video/chuỗi ảnh, làm theo `infer_rb_sequence`: mặc định `chunk_size=7`, `overlap=2`, bước tiến `chunk_size - overlap`; các đoạn chồng lấp được gộp để giảm đường biên thời gian. Adapter phải nhận `chunk_size` từ job để người dùng có thể chọn 7/12/16/24 trên giao diện (và 10 nếu thêm lựa chọn), thay vì hardcode 7. Giữ thứ tự frame và FPS. Ví dụ notebook giữ toàn bộ output trong RAM, chỉ phù hợp clip ngắn; bản web cần giới hạn clip ban đầu rồi chuyển sang ghi tạm theo luồng khi mở rộng.
6. Với ảnh đơn, thử `T=1` và lấy frame output duy nhất. Đây là suy luận hợp lệ theo chữ ký `forward(lqs)` nhưng không dùng ngữ cảnh thời gian 7 frame như khi train. Chỉ công bố chức năng ảnh bằng `best.pth` sau khi xem chất lượng trên ảnh anime thật; không quảng bá checkpoint này là model ảnh tổng quát. Nếu chất lượng kém, tích hợp model image SR riêng cho MVP ảnh và dùng `best.pth` ở phần video.

API phải tái sử dụng một model đã load, xếp hàng/giới hạn inference đồng thời theo VRAM và trả lỗi rõ khi checkpoint không khớp. Phần inference chỉ cần PyTorch, NumPy và thư viện đọc/ghi ảnh phù hợp; các import dùng riêng cho train/biểu đồ không cần ở server. Không sao chép các cell cài package, tải model gốc, resume train hoặc đường dẫn `/kaggle/*` vào tiến trình server.

### Phần 3 — API ảnh và quản lý file

- Hoàn thiện `GET /api/health`, `GET /api/models`, `POST /api/upscale` và `GET /api/results/{result_id}` theo `SYSTEM_DESIGN.md`. `POST /api/upscale` nhận ảnh và `model_id`; scale cố định x4 ở MVP. Chỉ công bố `best.pth` trong API ảnh nếu thử nghiệm `T=1` ở phần 2 đạt yêu cầu.
- Kiểm tra định dạng và nội dung ảnh, dung lượng, số pixel; quản lý tên file/ID an toàn, thư mục upload/output, chính sách dọn file tạm và lỗi thiếu checkpoint/thiếu bộ nhớ.
- Đặt giới hạn inference theo **pixel và device** dựa trên RSS/VRAM đo thực tế; `MAX_IMAGE_DIMENSION=2048` ở khung Phần 1 chỉ là giới hạn chọn file, chưa phải giới hạn an toàn để chạy model. Trên CPU hiện tại, bắt đầu với mức nhỏ và thử tăng dần; xử lý một tác vụ mỗi lần. Không dùng `tracemalloc` để ước lượng tổng RAM PyTorch.
- Trả metadata thật: kích thước input/output, model, device, thời gian xử lý và URL kết quả.

**Xong khi:** gọi API bằng một ảnh thật trả về file upscale có thể mở/tải; input lỗi nhận thông báo dễ hiểu; hai request liên tiếp không load lại checkpoint.

### Phần 4 — Nối frontend với API
 
- Gọi `/api/models` để chỉ hiển thị model x4 thật đang có; nếu chỉ có một model thì hiển thị tên thay vì dropdown giả. Đặt tên rõ `RealBasicVSR Anime V3` nếu dùng `best.pth`.
- Upload ảnh, lấy kích thước thật để hiện output x4, gửi request, hiển thị trạng thái chờ/lỗi và kết quả thật. Dùng thanh so sánh trước/sau hiện có nếu phù hợp; nút Download lấy file từ API.
- Bỏ logic `setInterval` giả lập inference, FPS/VRAM/PSNR/SSIM mẫu và các lệnh export video trong luồng ảnh.

**Xong khi:** thao tác từ browser hoàn tất `upload → inference → preview → download`; không còn đường đi nào trong MVP báo “thành công” khi API chưa trả kết quả. *(Đã hoàn thành và xác thực qua browser subagent)*

### Phần 5 — Ổn định bản local và bàn giao

- Kiểm tra luồng trên một ảnh nhỏ, một ảnh lớn trong giới hạn, file sai định dạng và trường hợp thiếu checkpoint; kiểm tra CPU, CUDA nếu máy có GPU. Nếu có cùng bộ `testLQ`/clip của notebook thì đối chiếu thêm output chuỗi frame.
- Đo thời gian/bộ nhớ cơ bản để chọn giới hạn upload hợp lý. Sửa lỗi xoay ảnh, màu/alpha, tên file, request lặp và dọn file tạm nếu gặp.
- Cập nhật `README.md` với cài đặt, checkpoint, lệnh chạy frontend/backend và cách kiểm tra kết quả; đồng bộ `SYSTEM_DESIGN.md` với mã thực tế.

**Xong khi:** người khác có thể làm theo README để chạy demo local bằng checkpoint hợp lệ, và các lỗi thường gặp đều có thông báo rõ. *(Đã hoàn thành toàn diện: automated cleanup trong lifespan, alpha compositing, EXIF orientation, boundary validation, 26 bài unit/integration tests đạt 100%, 8 bài chẩn đoán live server đạt 100%, và đồng bộ tài liệu)*

#### Phần 5A — Ưu tiên GPU rời cho inference khi khả dụng

**Hiện trạng máy lúc lập kế hoạch (29/09/2026):** `nvidia-smi` thấy **NVIDIA GeForce GTX 1650, VRAM 4 GB, driver 572.16**, và báo CUDA Version 12.8. Python hiện cài `torch 2.10.0+cpu`; `torch.version.cuda is None`, `torch.cuda.is_available() == False`. Do đó API hiện **đang chạy CPU**. Con số CUDA trong `nvidia-smi` cho biết mức driver hỗ trợ, không chứng minh PyTorch hiện tại đã dùng được GPU. Kiểm tra lại các thông số này trên máy triển khai trước khi cài đặt.

**Phạm vi phần này:** hỗ trợ NVIDIA qua CUDA trên Windows trước. GPU rời AMD/Intel không dùng đường CUDA hiện tại; chỉ thêm ROCm/XPU/DirectML sau khi có môi trường PyTorch và phép toán của RealBasicVSR được kiểm thử riêng. Máy không có backend GPU tương thích tiếp tục dùng CPU.

1. **Chuẩn bị môi trường CUDA riêng:** giữ môi trường CPU đang chạy được; tạo virtual environment riêng cho CUDA trên Windows. Dùng [bộ chọn cài đặt chính thức của PyTorch](https://pytorch.org/get-started/locally/) với Windows + Pip + phiên bản CUDA phù hợp GPU/driver để lấy lệnh cài hiện hành, không ghi cứng wheel/version CUDA vào script dự án. Cài các phụ thuộc trong `backend/requirements.txt` và kiểm tra lại rằng chúng không thay bản torch CUDA vừa chọn. Không tự động cài hoặc đổi driver từ ứng dụng web.
2. **Xác nhận PyTorch thực sự dùng GPU:** chạy `nvidia-smi` và lệnh dưới đây trong **cùng Python/venv dùng để khởi chạy Uvicorn**. Chỉ coi CUDA sẵn sàng khi `torch.cuda.is_available()` là `True`, tên GPU đúng và tạo tensor trên `cuda:0` thành công; việc có card rời trong Windows chưa đủ.

   ```powershell
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print(torch.ones(1, device='cuda:0') if torch.cuda.is_available() else '')"
   ```

3. **Chọn device ở backend:** thêm cấu hình `AI_DEVICE=auto|cuda|cpu` (mặc định `auto`) và `CUDA_DEVICE_INDEX=0`. Ở chế độ `auto`, thử CUDA đã được PyTorch xác nhận, nạp `best.pth` **một lần** vào GPU rồi chạy một ảnh nhỏ để xác thực cả model và các phép toán như `grid_sample`. Nếu bước này thất bại, ghi nguyên nhân và nạp model trên CPU để dịch vụ vẫn hoạt động. Với `AI_DEVICE=cuda` do người vận hành chọn rõ, báo lỗi/model không khả dụng nếu CUDA thất bại thay vì âm thầm chuyển CPU. `/api/health`, `/api/models` và response `/api/upscale` phải báo **device thực tế của model đã nạp**, không suy từ tên GPU hoặc chỉ từ `torch.cuda.is_available()`.
4. **Giới hạn VRAM trên GTX 1650 4 GB:** giữ một inference mỗi lần và một worker Uvicorn; không tạo nhiều bản sao model trên GPU. Đo VRAM còn trống bằng `torch.cuda.mem_get_info()`, đỉnh tensor bằng `max_memory_allocated()` và bộ nhớ cache bằng `max_memory_reserved()` trên các ảnh 64, 128, 224, 320 px. Thiết lập ngân sách pixel GPU riêng từ kết quả đo; không áp dụng mặc định giới hạn CPU 102.400 pixel cho GPU hoặc tự tăng lên 512 px khi chưa thử. Nếu OOM trong request, trả lỗi rõ và gợi ý ảnh nhỏ hơn; không coi `empty_cache()` là cách tăng VRAM cho tensor đang dùng.
5. **Đo tốc độ và chất lượng đúng cách:** chạy cùng checkpoint, ảnh đầu vào và output PNG trên CPU/GPU; warmup trước khi đo. Đồng bộ CUDA trước/sau phép đo (`torch.cuda.synchronize()` hoặc CUDA events) vì kernel chạy bất đồng bộ. Ghi thời gian model, thời gian tổng request, VRAM peak và kích thước output x4; so sánh màu/chi tiết với FP32 CPU. Chỉ ghi “nhanh hơn” sau khi có số liệu thực trên máy này; với ảnh rất nhỏ, thời gian chuyển CPU↔GPU có thể ảnh hưởng kết quả.
6. **Tối ưu sau khi FP32 CUDA ổn định:** có thể thử `torch.autocast(device_type='cuda', dtype=torch.float16)` để giảm VRAM/tăng tốc, nhưng phải kiểm tra `grid_sample`/SPyNet, NaN, sọc/đổi màu và độ lệch so với FP32. Nếu không ổn hoặc không nhanh hơn, giữ FP32. Video/chunk 7–24 frame cần đo VRAM riêng; kết quả ảnh đơn `T=1` không đủ để chọn chunk video an toàn.

**File đã cập nhật:** `backend/app/config.py`, `backend/app/model_registry.py`, `backend/app/inference/realbasicvsr_pipeline.py`, `backend/app/main.py`, `backend/app/schemas.py`, `scripts/benchmark_cuda_vs_cpu.py`, `.env.example`, `README.md`, `SYSTEM_DESIGN.md`.

**Kết quả kiểm tra Phần 5A:**
- Môi trường ảo `.venv_cuda` cài đặt `torch 2.14.0+cu126` nhận diện chính xác `NVIDIA GeForce GTX 1650 (4 GB VRAM)`.
- Backend warmup **hai frame 32×32** để chạy cả optical flow/`grid_sample`, nạp model vào `cuda:0`; live API báo đúng `device=cuda:0` và tên GTX 1650. Môi trường CPU gốc vẫn chạy được.
- Report ban đầu đo 320×320 là CPU 16.21 s / GPU 0.656 s (24.7×). Trong phép đo độc lập với cùng input và warmup cùng kích thước: **CPU 6.12 s / GPU 0.582 s (~10.5×)**, PyTorch peak allocated **978.4 MB**, peak reserved **1280 MB**. Ảnh 128×128 đạt khoảng 9.9× trong lần đo này. Tốc độ CPU thay đổi theo điều kiện chạy; không dùng 24.7× như mức tăng tốc bảo đảm.
- Kết quả uint8 CPU/GPU rất gần nhau trên ảnh thử: MAE khoảng `0.000026/255` ở 320×320, sai khác lớn nhất 1 mức màu; `0.0000` là số đã làm tròn, không chứng minh đồng nhất số học tuyệt đối.
- Môi trường CUDA chạy **27 bài kiểm thử đạt** (gồm một bài CUDA thật); môi trường CPU chạy **26 bài đạt và một bài CUDA được bỏ qua**. Live server CUDA trên cổng kiểm tra riêng đạt 8/8 kịch bản.

Nguồn kỹ thuật: [PyTorch cài đặt trên Windows](https://pytorch.org/get-started/locally/), [PyTorch CUDA và bộ nhớ](https://docs.pytorch.org/docs/stable/notes/cuda.html), [đo VRAM trống](https://docs.pytorch.org/docs/stable/generated/torch.cuda.mem_get_info.html), [NVIDIA giải thích CUDA Version trong `nvidia-smi`](https://docs.nvidia.com/deploy/nvidia-smi/).

## Phần mở rộng sau MVP

1. **Nhiều model ảnh/anime:** thêm checkpoint và metadata vào model registry; chỉ công bố x4 nếu checkpoint đó hỗ trợ x4. Nếu sau này muốn mở x2, thiết kế lại UI và API theo khả năng từng model.
   **Đã triển khai lựa chọn checkpoint RealBasicVSR:** registry quét `.pth`
   ở thư mục gốc và `MODELS_DIR`, API activate kiểm tra strict trước khi đổi,
   `ModelSelector` dùng chung cho ảnh/video, job giữ ID đã chọn, cache model
   từng dùng trên CPU và chỉ giữ một model trên GPU. Hỗ trợ kiến trúc x4
   hiện có (64 channels, 20 blocks); checkpoint kiến trúc khác báo lỗi khi nạp.
2. **Video:** triển khai theo [VIDEO_IMPLEMENTATION_PLAN.md](VIDEO_IMPLEMENTATION_PLAN.md): CLI xử lý chunk bounded → API job bất đồng bộ → React video → đo và mở giới hạn. Người dùng chọn `chunk_size` đã được benchmark; FFmpeg xử lý frame/FPS/audio/MP4. Dùng polling trước, WebSocket chỉ khi cần cập nhật realtime.
3. **Đánh giá chất lượng:** chế độ riêng nhận ảnh ground truth đã căn chỉnh mới tính PSNR/SSIM; không dùng ảnh input làm ground truth cho ảnh output x4.

## Điểm bắt đầu khi code

**Toàn bộ các giai đoạn chính và mở rộng đã hoàn thành:**
- **Phần 1 đến 5:** MVP ảnh x4 local, FastAPI, model registry, alpha composite, auto cleanup, 26 unit tests đạt 100%.
- **Phần 5A:** CUDA GPU Acceleration cho card GTX 1650 (4 GB VRAM) đạt tốc độ gấp ~10.5× so với CPU, MAE ~0.0000.
- **Phần Video (V1 - V4):** Xem [VIDEO_IMPLEMENTATION_PLAN.md](VIDEO_IMPLEMENTATION_PLAN.md):
  - **Phase V1:** Core Video Engine (ffprobe CFR probe, passthrough frame decoding, bounded-memory sliding chunk buffer stream, FFmpeg H.264 `yuv420p` + AAC audio mux, CLI runner `scripts/infer_video.py`).
  - **Phase V2:** Asynchronous Video Job Manager với hàng đợi FIFO, SSE broadcasting `/api/video/jobs/{id}/stream`, polling, serialize GPU lock và REST endpoints.
  - **Phase V3:** UI Video Studio trên React/Vite với Mode Switcher Header, one-click anime sample loader, progress tracking bar và Synchronized Dual Video Player.
  - **Phase V4:** Benchmark mở rộng (Chunk 7 vs Chunk 12 trên GTX 1650) và bộ test suite 45 bài kiểm thử đạt 100%.


