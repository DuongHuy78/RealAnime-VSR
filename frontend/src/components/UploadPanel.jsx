import React, { useRef } from 'react';
import ModelSelector from './ModelSelector';

export default function UploadPanel({
  imageMeta,
  modelSelectorProps,
  onImageSelected,
  onSelectSample,
  samples,
  isUpscaling = false,
  upscaleError = null,
  resultMeta = null,
  onRunUpscale,
  onDismissError,
  health,
}) {
  const fileInputRef = useRef(null);
  const { models, selectedModel, isSwitchingModel } = modelSelectorProps;
  const limits = health?.image_limits;
  const maxUploadBytes = limits?.max_upload_size_bytes ?? 15 * 1024 * 1024;
  const maxDimension = limits?.max_image_dimension ?? 512;
  const maxPixels = limits?.max_image_pixels ?? 102400;
  const minDimension = limits?.min_image_dimension ?? 16;
  const maxUploadMb = Math.floor(maxUploadBytes / (1024 * 1024));

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (file) {
      processSelectedFile(file);
    }
  };

  const processSelectedFile = (file) => {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      alert('Vui lòng chọn file ảnh định dạng PNG, JPG, hoặc WEBP.');
      return;
    }
    if (file.size > maxUploadBytes) {
      alert(`Ảnh vượt giới hạn ${maxUploadMb} MB.`);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      if (img.naturalWidth < minDimension || img.naturalHeight < minDimension) {
        URL.revokeObjectURL(objectUrl);
        alert(`Mỗi cạnh ảnh cần ít nhất ${minDimension} px.`);
        return;
      }
      if (img.naturalWidth > maxDimension || img.naturalHeight > maxDimension || img.naturalWidth * img.naturalHeight > maxPixels) {
        URL.revokeObjectURL(objectUrl);
        alert(`Ảnh cần có mỗi cạnh không quá ${maxDimension} px và tổng không quá ${maxPixels.toLocaleString('vi-VN')} pixel.`);
        return;
      }
      onImageSelected({
        file,
        url: objectUrl,
        name: file.name,
        size: (file.size / 1024).toFixed(1) + ' KB',
        width: img.naturalWidth,
        height: img.naturalHeight,
        outWidth: img.naturalWidth * 4,
        outHeight: img.naturalHeight * 4,
      });
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      alert('Không thể đọc file ảnh đã chọn.');
    };
    img.src = objectUrl;
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      processSelectedFile(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const isBackendOnline = Boolean(health && health.status === 'ok');
  const hasSelectedModel = models.some((model) => model.id === selectedModel && model.available);
  const canUpscale = Boolean(imageMeta?.file && isBackendOnline && hasSelectedModel && !isUpscaling && !isSwitchingModel);

  return (
    <div className="w-full lg:w-80 flex-shrink-0 flex flex-col space-y-4 bg-[#17191d] border-r border-[#262930] p-4 text-xs font-mono overflow-y-auto">
      {/* Error Alert Banner */}
      {upscaleError && (
        <div className="p-3 rounded bg-red-950/40 border border-red-500/40 text-red-200 text-[11px] leading-relaxed relative flex items-start justify-between">
          <div className="flex items-start space-x-2">
            <span className="text-red-400 font-bold">⚠️</span>
            <div>
              <span className="font-semibold text-red-300 block mb-0.5">Lỗi suy luận AI:</span>
              <span>{upscaleError}</span>
            </div>
          </div>
          {onDismissError && (
            <button
              onClick={onDismissError}
              className="text-red-400 hover:text-red-200 ml-2 p-1 text-sm font-bold leading-none cursor-pointer"
              title="Đóng thông báo"
            >
              &times;
            </button>
          )}
        </div>
      )}

      {/* 1. Upload Dropzone */}
      <div>
        <span className="text-[11px] font-semibold text-[#7d8494] uppercase tracking-wider block mb-2">
          01 // Source Media
        </span>

        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => fileInputRef.current?.click()}
          className="border border-dashed border-[#343842] hover:border-[#b9f73a] bg-[#1d1f25] hover:bg-[#22252c] rounded-lg p-5 flex flex-col items-center justify-center cursor-pointer transition-colors group text-center"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            className="hidden"
            onChange={handleFileChange}
          />
          <div className="w-10 h-10 rounded-full bg-[#262930] group-hover:bg-[#2e343d] flex items-center justify-center text-[#7d8494] group-hover:text-[#b9f73a] mb-2.5 transition-colors">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
          </div>
          <p className="text-white font-medium mb-1">Upload Source Image</p>
          <p className="text-[11px] text-[#7d8494]">Drag & drop or browse</p>
          <p className="text-[10px] text-[#7d8494] mt-2">PNG, JPG, WEBP · ≤{maxUploadMb} MB</p>
          <p className="text-[10px] text-[#7d8494] mt-1">Mỗi cạnh ≤{maxDimension} px · Tổng ≤{maxPixels.toLocaleString('vi-VN')} pixel</p>
        </div>
      </div>

      {/* 2. Quick Test Samples */}
      {samples && samples.length > 0 && (
        <div>
          <span className="text-[11px] font-semibold text-[#7d8494] uppercase tracking-wider block mb-2">
            Quick Test Samples
          </span>
          <div className="grid grid-cols-2 gap-2">
            {samples.map((sample) => (
              <button
                key={sample.id}
                onClick={() => onSelectSample(sample)}
                className="px-2.5 py-1.5 rounded bg-[#1d1f25] hover:bg-[#22252c] border border-[#262930] hover:border-[#343842] text-left transition-colors flex items-center justify-between cursor-pointer"
              >
                <span className="text-[#d1d5db] truncate">{sample.label}</span>
                <span className="text-[10px] text-amber-300">DEMO</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 3. Image Specs (Input -> Output x4) */}
      <div className="bg-[#1d1f25] rounded-lg p-3.5 border border-[#262930] space-y-2.5">
        <span className="text-[11px] font-semibold text-[#7d8494] uppercase tracking-wider block">
          02 // Resolution Specs (Fixed 4X)
        </span>

        {imageMeta ? (
          <div className="space-y-2 text-[11px]">
            <div className="flex justify-between items-center py-1 border-b border-[#262930]">
              <span className="text-[#7d8494]">Input Dimension:</span>
              <span className="text-white font-semibold">{imageMeta.width ? `${imageMeta.width} × ${imageMeta.height} px` : 'Đang đọc ảnh...'}</span>
            </div>
            <div className="flex justify-between items-center py-1 border-b border-[#262930]">
              <span className="text-[#7d8494]">Output mục tiêu (4X):</span>
              <span className="text-[#b9f73a] font-semibold">{imageMeta.outWidth ? `${imageMeta.outWidth} × ${imageMeta.outHeight} px` : 'Đang tính...'}</span>
            </div>
            <div className="flex justify-between items-center py-1 border-b border-[#262930]">
              <span className="text-[#7d8494]">File Size:</span>
              <span className="text-[#d1d5db]">{imageMeta.size}</span>
            </div>
            <div className="flex justify-between items-center py-1">
              <span className="text-[#7d8494]">Scale Factor:</span>
              <span className="px-1.5 py-0.5 rounded bg-[#262930] text-[#b9f73a] font-semibold">4.0&times; cố định</span>
            </div>
          </div>
        ) : (
          <p className="text-[11px] text-[#7d8494] italic py-2">
            Chọn hoặc upload ảnh để tính kích thước mục tiêu 4X.
          </p>
        )}
      </div>

      {/* 4. Model Selection */}
      <ModelSelector {...modelSelectorProps} disabled={isUpscaling} />
      {resultMeta?.checkpoint_filename && (
        <p className="text-[11px] text-[#9aa2b1]">Mô hình của kết quả: {resultMeta.checkpoint_filename}</p>
      )}

      {/* 5. Primary Action Button */}
      <div className="pt-1">
        {isUpscaling ? (
          <button
            disabled
            className="w-full py-3 px-4 rounded-lg bg-[#22252c] border border-[#343842] text-[#b9f73a] font-bold uppercase tracking-wider flex items-center justify-center space-x-2 cursor-wait"
          >
            <svg className="animate-spin h-4 w-4 text-[#b9f73a]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
            </svg>
            <span>ĐANG SUY LUẬN AI...</span>
          </button>
        ) : (
          <button
            onClick={onRunUpscale}
            disabled={!canUpscale}
            className={`w-full py-3 px-4 rounded-lg font-bold uppercase tracking-wider transition-all flex items-center justify-center space-x-2 ${
              canUpscale
                ? 'bg-[#b9f73a] hover:bg-[#c6ff47] text-black shadow-[0_0_15px_rgba(185,247,58,0.25)] hover:shadow-[0_0_20px_rgba(185,247,58,0.4)] cursor-pointer active:scale-[0.99]'
                : 'bg-[#22252c] border border-[#262930] text-[#555a66] cursor-not-allowed'
            }`}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <span>CHẠY SIÊU PHÂN GIẢI (4X)</span>
          </button>
        )}

        {!imageMeta?.file && (
          <p className="text-[10px] text-[#7d8494] text-center mt-2 italic">
            Tải ảnh từ máy để kích hoạt nút suy luận AI.
          </p>
        )}
        {imageMeta?.file && !isBackendOnline && (
          <p className="text-[10px] text-red-400 text-center mt-2">
            Backend API đang ngoại tuyến. Vui lòng kiểm tra server.
          </p>
        )}
        {imageMeta?.file && isBackendOnline && !hasSelectedModel && (
          <p className="text-[10px] text-amber-300 text-center mt-2">
            Chưa có model ảnh khả dụng trên backend.
          </p>
        )}
      </div>

      {/* 6. Execution Stats (After successful AI upscale) */}
      {resultMeta && (
        <div className="bg-[#1d1f25] rounded-lg p-3.5 border border-[#b9f73a]/30 space-y-2 text-[11px] animate-fadeIn">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-[#b9f73a] uppercase tracking-wider">
              04 // Kết quả suy luận
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#b9f73a]/20 text-[#b9f73a] font-semibold">
              REAL AI
            </span>
          </div>
          <div className="space-y-1.5 pt-1 border-t border-[#262930]">
            <div className="flex justify-between items-center">
              <span className="text-[#7d8494]">Thời gian suy luận:</span>
              <span className="text-white font-semibold">
                {resultMeta.inference_time_ms ? `${resultMeta.inference_time_ms.toLocaleString()} ms` : '--'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#7d8494]">Thiết bị thực thi:</span>
              <span className="text-[#b9f73a] font-semibold">{resultMeta.device?.toUpperCase() || 'CPU'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#7d8494]">Kích thước đầu ra:</span>
              <span className="text-white font-semibold">
                {resultMeta.output_size?.width} × {resultMeta.output_size?.height} px
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#7d8494]">Hệ số phóng đại:</span>
              <span className="text-[#b9f73a] font-semibold">4.0&times; Checkpoint thật</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

