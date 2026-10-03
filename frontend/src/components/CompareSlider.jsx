import React, { useState, useRef, useEffect, useCallback } from 'react';
import CropSelection from './CropSelection';
import CropCompareDialog from './CropCompareDialog';
import { captureComparison } from '../utils/cropComparison';

export default function CompareSlider({
  originalUrl,
  comparisonUrl,
  isDemoSample,
  isUpscaling = false,
  resultMeta = null,
}) {
  const [sliderPos, setSliderPos] = useState(50); // percentage (0 - 100)
  const [viewMode, setViewMode] = useState('split'); // 'split', 'original', 'upscaled'
  const hasComparison = Boolean(comparisonUrl);
  const activeMode = hasComparison ? viewMode : 'original';
  const containerRef = useRef(null);
  const isDragging = useRef(false);
  const originalMediaRef = useRef(null);
  const comparisonMediaRef = useRef(null);
  const [cropMode, setCropMode] = useState(false);
  const [cropComparison, setCropComparison] = useState(null);
  const [cropError, setCropError] = useState(null);

  function compareCrop(selection) {
    try {
      setCropError(null);
      setCropComparison(captureComparison(originalMediaRef.current, comparisonMediaRef.current, selection));
    } catch (error) {
      setCropError(error.message || 'Không thể mở vùng crop.');
    }
  }

  const handlePointerDown = () => {
    isDragging.current = true;
  };

  const handlePointerUp = useCallback(() => {
    isDragging.current = false;
  }, []);

  const handlePointerMove = useCallback((e) => {
    if (!isDragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pos = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(pos);
  }, []);

  useEffect(() => {
    window.addEventListener('pointerup', handlePointerUp);
    window.addEventListener('pointermove', handlePointerMove);
    return () => {
      window.removeEventListener('pointerup', handlePointerUp);
      window.removeEventListener('pointermove', handlePointerMove);
    };
  }, [handlePointerUp, handlePointerMove]);

  if (!originalUrl) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 text-center select-none bg-[#121316]">
        <div className="w-16 h-16 rounded-2xl bg-[#17191d] border border-[#262930] flex items-center justify-center text-[#7d8494] mb-4">
          <svg className="w-8 h-8" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" />
          </svg>
        </div>
        <h3 className="text-white text-base font-semibold font-mono mb-1">
          Chưa có hình ảnh
        </h3>
        <p className="text-xs text-[#7d8494] font-mono max-w-sm">
          Tải ảnh lên từ bảng bên trái hoặc chọn một trong các ảnh mẫu (Quick Test Samples) để xem trước và so sánh.
        </p>
      </div>
    );
  }

  const downloadUrl = resultMeta?.result_id
    ? `/api/results/${resultMeta.result_id}?download=true`
    : null;

  return (
    <div className="flex-1 min-w-0 flex flex-col h-full bg-[#121316] overflow-hidden select-none">
      <img ref={originalMediaRef} src={originalUrl} alt="" className="hidden" />
      {/* Viewport Toolbar */}
      <div className="px-6 py-2.5 border-b border-[#262930] bg-[#17191d] flex flex-wrap gap-2 items-center justify-between font-mono text-xs">
        <div className="flex items-center space-x-2">
          <span className="text-[#7d8494]">CHẾ ĐỘ XEM:</span>
          <div className="inline-flex rounded-md bg-[#121316] p-0.5 border border-[#262930]">
            <button
              onClick={() => setViewMode('split')}
              disabled={!hasComparison}
              className={`px-3 py-1 rounded text-xs transition-colors cursor-pointer ${
                activeMode === 'split' ? 'bg-[#22252c] text-[#b9f73a] font-semibold' : 'text-[#7d8494] hover:text-white'
              } disabled:opacity-40 disabled:cursor-not-allowed`}
            >
              Thanh trượt chia đôi
            </button>
            <button
              onClick={() => setViewMode('original')}
              className={`px-3 py-1 rounded text-xs transition-colors cursor-pointer ${
                activeMode === 'original' ? 'bg-[#22252c] text-white font-semibold' : 'text-[#7d8494] hover:text-white'
              }`}
            >
              Ảnh gốc (1X)
            </button>
            <button
              onClick={() => setViewMode('upscaled')}
              disabled={!hasComparison}
              className={`px-3 py-1 rounded text-xs transition-colors cursor-pointer ${
                activeMode === 'upscaled' ? 'bg-[#22252c] text-[#b9f73a] font-semibold' : 'text-[#7d8494] hover:text-white'
              } disabled:opacity-40 disabled:cursor-not-allowed`}
            >
              {isDemoSample ? 'Ảnh mẫu Sau (4X)' : 'Siêu phân giải (4X)'}
            </button>
          </div>
        </div>

        <button
          type="button"
          disabled={!hasComparison || isUpscaling}
          aria-pressed={cropMode}
          onClick={() => { setCropMode((value) => !value); setCropError(null); }}
          className={`rounded border px-3 py-1.5 disabled:opacity-40 disabled:cursor-not-allowed ${cropMode ? 'border-[#b9f73a] text-[#b9f73a]' : 'border-[#343842] text-white'}`}
        >
          {cropMode ? 'Hủy chọn vùng' : 'Crop & Zoom'}
        </button>

        {/* Download Button */}
        {hasComparison && !isDemoSample && downloadUrl && (
          <a
            href={downloadUrl}
            className="px-3.5 py-1.5 rounded bg-[#b9f73a] hover:bg-[#c6ff47] text-black font-semibold flex items-center space-x-1.5 transition-colors shadow-[0_0_10px_rgba(185,247,58,0.2)] cursor-pointer"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
            </svg>
            <span>Tải về 4X</span>
          </a>
        )}
      </div>

      {cropMode && <p className="bg-[#17191d] px-6 py-2 text-xs text-[#b9f73a]">Kéo một vùng trên ảnh rồi thả để mở popup so sánh và zoom.</p>}
      {cropError && <p role="alert" className="px-6 py-2 text-xs text-red-300">{cropError}</p>}

      {/* Main Viewport Container */}
      <div className="flex-1 relative flex items-center justify-center p-4 overflow-hidden bg-[#0d0e11]">
        <div
          ref={containerRef}
          className="relative max-w-full max-h-full aspect-auto rounded-lg overflow-hidden border border-[#262930] shadow-2xl bg-[#17191d]"
          style={{ maxHeight: 'calc(100vh - 140px)' }}
        >
          {/* Base: comparison image or original when inference is not available */}
          <img
            ref={comparisonMediaRef}
            src={comparisonUrl || originalUrl}
            alt={isDemoSample ? 'Ảnh minh họa' : 'Xem trước hình ảnh'}
            className="block max-w-full max-h-[calc(100vh-140px)] w-auto object-contain pointer-events-none"
          />

          {/* Split Mode Overlay: Original image clipped */}
          {activeMode === 'split' && (
            <div
              className="absolute inset-0"
              style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
            >
              <img
                src={originalUrl}
                alt="Ảnh gốc"
                className="absolute inset-0 w-full h-full object-contain pointer-events-none"
              />
            </div>
          )}

          {/* Single Views */}
          {activeMode === 'original' && hasComparison && (
            <img
              src={originalUrl}
              alt="Ảnh gốc độc lập"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            />
          )}

          {/* Split Divider & Handle */}
          {activeMode === 'split' && !cropMode && (
            <div
              onPointerDown={handlePointerDown}
              className="absolute top-0 bottom-0 w-1 cursor-ew-resize z-20 slider-divider flex items-center justify-center"
              style={{ left: `calc(${sliderPos}% - 2px)` }}
            >
              <div className="w-7 h-7 rounded-full bg-[#17191d] border-2 border-[#b9f73a] shadow-[0_0_12px_rgba(185,247,58,0.5)] flex items-center justify-center text-[#b9f73a]">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M8 9l-3 3 3 3m8-6l3 3-3 3" />
                </svg>
              </div>
            </div>
          )}

          {/* Floating Badges */}
          <div className="absolute top-3 left-3 px-2.5 py-1 rounded bg-[#17191d]/80 backdrop-blur border border-[#262930] text-[10px] font-mono text-[#d1d5db]">
            ORIGINAL (1X)
          </div>

          {cropMode && hasComparison && !isUpscaling && <CropSelection mediaRef={comparisonMediaRef} onComplete={compareCrop} />}
          <div className="absolute top-3 right-3 px-2.5 py-1 rounded bg-[#17191d]/80 backdrop-blur border border-[#262930] text-[10px] font-mono text-[#b9f73a]">
            {isDemoSample
              ? 'DEMO — ẢNH MINH HỌA'
              : hasComparison
              ? `AI UPSCALE (4X) • ${(resultMeta?.device || 'AI').toUpperCase()}`
              : 'CHƯA XỬ LÝ AI'}
          </div>

          {/* In-Flight Inference Overlay */}
          {isUpscaling && (
            <div className="absolute inset-0 bg-[#0d0e11]/75 backdrop-blur-sm z-30 flex flex-col items-center justify-center space-y-3">
              <div className="relative flex items-center justify-center">
                <div className="w-12 h-12 rounded-full border-2 border-[#262930] border-t-[#b9f73a] animate-spin"></div>
                <div className="absolute w-4 h-4 rounded-full bg-[#b9f73a]/30 animate-pulse"></div>
              </div>
              <div className="text-center font-mono">
                <p className="text-white text-xs font-semibold tracking-wider">
                  ĐANG SUY LUẬN SIÊU PHÂN GIẢI 4X...
                </p>
                <p className="text-[#7d8494] text-[11px] mt-0.5">
                  RealBasicVSR Anime V3 (Single Frame)
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
      {cropComparison && <CropCompareDialog comparison={cropComparison} isDemo={isDemoSample} onClose={() => setCropComparison(null)} />}
    </div>
  );
}

