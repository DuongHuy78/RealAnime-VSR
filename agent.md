# Hướng dẫn chung cho agent

## Mục tiêu dự án

Xây dựng web demo upscale ảnh/video chạy trên máy cá nhân. Giai đoạn đầu dùng React + Vite + Tailwind CSS ở frontend và FastAPI + PyTorch ở backend. Repo đã có `best.pth` của RealBasicVSR Anime V3; xem `IMPLEMENTATION_PLAN.md` để nạp đúng kiến trúc, thử inference chuỗi frame và đánh giá riêng chế độ ảnh đơn `T=1`. Video web và AnimeSR thuộc giai đoạn sau.

## Hiện trạng

- Repo có prototype cũ `index.html`, `app.js`, `styles.css` ở gốc và ứng dụng React/Vite trong `frontend/`, FastAPI trong `backend/`. Model `best.pth` chạy qua CLI/API ảnh và frontend đã gửi request inference thật. `.venv_cuda` trên máy này dùng GTX 1650 qua CUDA; Python hệ thống vẫn là môi trường CPU. Ảnh demo được gắn nhãn riêng. Xem kết quả rà soát GPU Phần 5A và phần video còn lại trong `IMPLEMENTATION_PLAN.md`.
- `SYSTEM_DESIGN.md` là nguồn tham khảo cho kiến trúc, luồng dữ liệu và mục đích của các thư mục dự kiến. Cây thư mục trong tài liệu đó chưa có nghĩa là các file đã tồn tại.

## Nguyên tắc khi triển khai

1. Ưu tiên một luồng ảnh hoàn chỉnh: upload → chọn model x4 khả dụng → inference → xem trước → tải xuống. Với `best.pth`, phải xác thực chất lượng ảnh đơn `T=1` trước khi cho phép dùng trong luồng ảnh; giữ khả năng suy luận chuỗi frame để triển khai video sau.
2. Backend gọi model Python trực tiếp. Chỉ thêm dịch vụ khác khi có nhu cầu thực tế.
3. Load checkpoint một lần khi khởi động hoặc lần đầu dùng model, rồi tái sử dụng. Tránh load lại ở từng request. Giới hạn số tác vụ inference đồng thời phù hợp với bộ nhớ GPU/CPU.
4. Chỉ hiển thị model, scale và device thực sự khả dụng. Scale x2/x4 phải dựa trên checkpoint/cấu hình model; không gắn nhãn x2 cho một kết quả x4 rồi thu nhỏ mà không mô tả rõ.
5. Chỉ chọn CUDA khi bản PyTorch/driver và inference model đã chạy thử trên GPU; với `AI_DEVICE=auto`, dùng CPU nếu CUDA không khả dụng. Báo device thực tế, lỗi checkpoint/OOM rõ ràng; không trả ảnh resize thông thường dưới tên kết quả AI. Xem kế hoạch GPU Phần 5A trong `IMPLEMENTATION_PLAN.md`.
6. PSNR/SSIM chỉ tính khi có ảnh ground truth độc lập, cùng kích thước và được căn chỉnh. Màn hình demo upload một ảnh không được tự tính các chỉ số này từ ảnh gốc và ảnh upscale.
7. Kiểm tra định dạng, kích thước và nội dung file upload; giới hạn dung lượng/cạnh ảnh theo cấu hình. Tạo tên file kết quả an toàn, không dùng trực tiếp đường dẫn do client cung cấp. Dọn file tạm theo chính sách rõ ràng.
8. Không commit checkpoint, file upload, output, môi trường ảo hoặc bí mật vào repo. Ghi nguồn, giấy phép và cách tải checkpoint được dùng.
9. Giữ ranh giới rõ: `frontend` phụ trách giao diện; `backend` phụ trách API, tiền xử lý, inference và lưu kết quả. Không nhúng logic model vào frontend.
10. Khi thay đổi API hoặc cấu trúc thư mục, cập nhật `SYSTEM_DESIGN.md` và hướng dẫn chạy tương ứng. Kiểm tra luồng thực với một ảnh nhỏ trước khi coi tính năng hoàn tất.
11. Khi làm video RealBasicVSR, nối lựa chọn temporal chunk từ UI tới job và inference; đây là số frame mỗi chunk, không phải batch size. Xem preset, mặc định và cách kiểm tra giá trị trong `SYSTEM_DESIGN.md`.

## Cách làm việc trong repo

- Đọc `SYSTEM_DESIGN.md` và kiểm tra file thực tế trước khi sửa. Không giả định các thư mục dự kiến đã được tạo.
- Giữ thay đổi nhỏ, có thể chạy được; ưu tiên lỗi có thông báo hữu ích trên cả API và giao diện.
- Khi đưa giao diện hiện có vào Vite, xem `index.html`, `app.js`, `styles.css` là tài sản giao diện tham khảo. Không trình bày dữ liệu mẫu, chỉ số hiệu năng hay tiến trình mô phỏng trong prototype như kết quả AI thật.
- Ghi rõ trong báo cáo những gì đã chạy thử, checkpoint/device đã dùng và những phần vẫn chỉ là thiết kế.
