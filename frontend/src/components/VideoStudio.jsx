import React, { useState, useEffect, useRef } from 'react';
import CropSelection from './CropSelection';
import CropCompareDialog from './CropCompareDialog';
import ModelSelector from './ModelSelector';
import { captureComparison, seekVideoFrame } from '../utils/cropComparison';
import {
  probeVideo,
  submitVideoUpscale,
  fetchVideoJobStatus,
  getVideoResultUrl,
  getVideoDownloadUrl,
} from '../api/client';

const CHUNK_PRESETS = [
  { size: 7, label: '7 frames' },
  { size: 12, label: '12 frames' },
  { size: 16, label: '16 frames' },
];

export default function VideoStudio({ health, modelSelectorProps }) {
  const { models, selectedModel, isSwitchingModel } = modelSelectorProps;
  const [selectedFile, setSelectedFile] = useState(null);
  const [sourceUrl, setSourceUrl] = useState(null);
  const [probeData, setProbeData] = useState(null);
  const [isProbing, setIsProbing] = useState(false);
  const [probeError, setProbeError] = useState(null);

  // Settings
  const [chunkSize, setChunkSize] = useState(7);
  const [overlap] = useState(2);

  // Job status
  const [activeJobId, setActiveJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Synchronized playback refs & states
  const originalVideoRef = useRef(null);
  const srVideoRef = useRef(null);
  const playbackPositionRef = useRef(0);
  const shouldResumeRef = useRef(false);
  const isSeekingRef = useRef(false);
  const lastSyncTimeRef = useRef(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [viewMode, setViewMode] = useState('split'); // 'split' | 'single_sr' | 'single_orig'
  const [cropMode, setCropMode] = useState(false);
  const [cropComparison, setCropComparison] = useState(null);
  const [cropError, setCropError] = useState(null);
  const [cropBusy, setCropBusy] = useState(false);

  const fileInputRef = useRef(null);
  const sourceUrlRef = useRef(null);
  const pollTimerRef = useRef(null);
  const eventSourceRef = useRef(null);

  const videoLimits = health?.video_limits || {
    max_upload_mb: 25,
    max_duration_seconds: 10.0,
    max_frames: 300,
    max_input_pixels: 76800,
  };

  const isCompleted = jobStatus?.status === 'completed';
  const isProcessing = Boolean(jobStatus && ['pending', 'probing', 'decoding', 'inferring', 'encoding'].includes(jobStatus.status));
  const isLimitExceeded = Boolean(probeData && (
    probeData.duration_seconds > videoLimits.max_duration_seconds ||
    probeData.total_frames > videoLimits.max_frames ||
    (probeData.width * probeData.height) > videoLimits.max_input_pixels
  ));
  const hasSelectedModel = models.some((model) => model.id === selectedModel && model.available);
  const canStart = Boolean(selectedFile && probeData && hasSelectedModel && health?.status === 'ok' && !isSwitchingModel && !isLimitExceeded && !isProbing && !isProcessing && !isSubmitting);


  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
      if (eventSourceRef.current) eventSourceRef.current.close();
      if (sourceUrlRef.current) URL.revokeObjectURL(sourceUrlRef.current);
    };
  }, []);

  const setSourceFile = (file) => {
    if (sourceUrlRef.current) URL.revokeObjectURL(sourceUrlRef.current);
    const url = file ? URL.createObjectURL(file) : null;
    sourceUrlRef.current = url;
    setSourceUrl(url);
    setSelectedFile(file);
  };

  // Handle file select & automatic probe
  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;

    resetState();
    setProbeData(null);
    setSourceFile(file);
    setIsProbing(true);
    setProbeError(null);

    try {
      const probeRes = await probeVideo(file);
      setProbeData(probeRes);
    } catch (err) {
      setProbeError(err.message || 'Lỗi kiểm tra tệp video.');
      setProbeData(null);
    } finally {
      setIsProbing(false);
    }
  };

  const resetState = () => {
    setCropMode(false);
    setCropComparison(null);
    setCropError(null);
    setCropBusy(false);
    originalVideoRef.current?.pause();
    srVideoRef.current?.pause();
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    if (eventSourceRef.current) eventSourceRef.current.close();
    setActiveJobId(null);
    setJobStatus(null);
    setErrorMessage(null);
    setIsPlaying(false);
    setCurrentTime(0);
    playbackPositionRef.current = 0;
    shouldResumeRef.current = false;
  };

  const handleLoadAnimeSample = async () => {
    resetState();
    setProbeData(null);
    setSourceFile(null);
    setIsProbing(true);
    setProbeError(null);

    try {
      const response = await fetch('/demo/anime_sample_2s.mp4');
      if (!response.ok) throw new Error('Không thể tải video mẫu từ server.');
      const blob = await response.blob();
      const file = new File([blob], 'anime_sample_2s.mp4', { type: 'video/mp4' });

      setSourceFile(file);
      const probeRes = await probeVideo(file);
      setProbeData(probeRes);
    } catch (err) {
      setProbeError(err.message || 'Lỗi tải video mẫu.');
      setProbeData(null);
    } finally {
      setIsProbing(false);
    }
  };

  // Submit upscale job
  const handleStartUpscale = async () => {
    if (!canStart) return;
    resetState();
    setIsSubmitting(true);
    setErrorMessage(null);

    try {
      const createRes = await submitVideoUpscale(selectedFile, chunkSize, overlap, null, selectedModel);
      const jobId = createRes.job_id;
      setActiveJobId(jobId);

      // Start SSE listener with polling fallback
      setupProgressListener(jobId);
    } catch (err) {
      setErrorMessage(err.message || 'Không thể khởi tạo tác vụ siêu phân giải video.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const setupProgressListener = (jobId) => {
    // Try Server-Sent Events
    try {
      const sse = new EventSource(`/api/video/jobs/${jobId}/stream`);
      eventSourceRef.current = sse;

      sse.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setJobStatus(data);
          if (data.status === 'completed' || data.status === 'failed') {
            sse.close();
            if (data.status === 'failed') {
              setErrorMessage(data.error_message || 'Tác vụ xử lý video thất bại.');
            }
          }
        } catch (e) {
          console.warn('SSE JSON parse error:', e);
        }
      };

      sse.onerror = () => {
        // Fallback to polling if SSE connection drops
        sse.close();
        startPolling(jobId);
      };
    } catch {
      startPolling(jobId);
    }
  };

  const startPolling = (jobId) => {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    pollTimerRef.current = setInterval(async () => {
      try {
        const data = await fetchVideoJobStatus(jobId);
        setJobStatus(data);
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(pollTimerRef.current);
          if (data.status === 'failed') {
            setErrorMessage(data.error_message || 'Tác vụ xử lý video thất bại.');
          }
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 1500);
  };

  // Switch view mode preserving time & playing state
  const handleViewModeChange = (mode) => {
    if (mode === 'single_sr' && !isCompleted) return;
    const currentActive = viewMode === 'single_sr' ? srVideoRef.current : (originalVideoRef.current || srVideoRef.current);
    const wasPlaying = currentActive ? !currentActive.paused : isPlaying;
    const pos = currentActive ? currentActive.currentTime : currentTime;

    playbackPositionRef.current = pos;
    shouldResumeRef.current = wasPlaying;
    setCurrentTime(pos);
    setIsPlaying(wasPlaying);
    setViewMode(mode);
  };

  const handleLoadedMetadata = (event) => {
    const video = event.currentTarget;
    if (video.duration && (!duration || video === originalVideoRef.current)) {
      setDuration(video.duration);
    }
    const seekTo = Math.min(playbackPositionRef.current, Math.max(0, (video.duration || 1) - 0.05));
    if (seekTo > 0 && Math.abs(video.currentTime - seekTo) > 0.1) {
      video.currentTime = seekTo;
    }
    video.muted = video === srVideoRef.current && viewMode === 'split' ? true : isMuted;
    if (shouldResumeRef.current) {
      video.play().catch(() => {
        shouldResumeRef.current = false;
        setIsPlaying(false);
      });
    }
  };

  // Synchronized playback controls
  const pauseForCrop = () => {
    originalVideoRef.current?.pause();
    srVideoRef.current?.pause();
    shouldResumeRef.current = false;
    setIsPlaying(false);
  };

  const compareVideoCrop = async (selection) => {
    if (cropBusy) return;
    const original = originalVideoRef.current;
    const result = srVideoRef.current;
    if (!original || !result) return;
    pauseForCrop();
    setCropError(null);
    setCropBusy(true);
    try {
      const time = original.currentTime;
      await Promise.all([seekVideoFrame(original, time), seekVideoFrame(result, time)]);
      if (originalVideoRef.current !== original || srVideoRef.current !== result) return;
      setCropComparison({ ...captureComparison(original, result, selection), time });
    } catch (error) {
      setCropError(error.message || 'Không thể crop khung hình video.');
    } finally {
      setCropBusy(false);
    }
  };

  const togglePlay = () => {
    if (cropMode || cropBusy || cropComparison) return;
    const v1 = originalVideoRef.current;
    const v2 = srVideoRef.current;
    const videos = [v1, v2].filter(Boolean);
    if (videos.length === 0) return;

    const isAnyPlaying = videos.some((video) => !video.paused && !video.ended);
    if (isAnyPlaying) {
      videos.forEach((video) => video.pause());
      shouldResumeRef.current = false;
      setIsPlaying(false);
    } else {
      // Auto-rewind if at end
      const maxDur = Math.max(...videos.map((v) => v.duration || 0));
      const currentPos = videos[0]?.currentTime || 0;
      if (maxDur > 0 && currentPos >= maxDur - 0.08) {
        videos.forEach((v) => { v.currentTime = 0; });
        playbackPositionRef.current = 0;
        setCurrentTime(0);
      }

      shouldResumeRef.current = true;
      Promise.all(videos.map((video) => video.play()))
        .then(() => {
          setIsPlaying(true);
        })
        .catch((err) => {
          console.warn('Playback gesture or audio restricted, falling back to muted:', err);
          videos.forEach((video) => { video.muted = true; });
          setIsMuted(true);
          Promise.all(videos.map((video) => video.play()))
            .then(() => setIsPlaying(true))
            .catch(() => {
              videos.forEach((video) => video.pause());
              shouldResumeRef.current = false;
              setIsPlaying(false);
            });
        });
    }
  };

  // Smooth seamless loop when playback reaches end
  const handleEnded = () => {
    const v1 = originalVideoRef.current;
    const v2 = srVideoRef.current;
    const videos = [v1, v2].filter(Boolean);
    if (videos.length === 0) return;

    videos.forEach((v) => { v.currentTime = 0; });
    playbackPositionRef.current = 0;
    setCurrentTime(0);
    Promise.all(videos.map((v) => v.play())).catch(() => {
      setIsPlaying(false);
      shouldResumeRef.current = false;
    });
  };

  const handleTimeUpdate = (e) => {
    const master = viewMode === 'single_sr' ? srVideoRef.current : (originalVideoRef.current || srVideoRef.current);
    if (!master || e.currentTarget !== master) return;

    const current = master.currentTime;
    playbackPositionRef.current = current;
    setCurrentTime(current);

    // In split mode, only synchronize slave to master with throttling and seeking guard
    if (viewMode === 'split') {
      const slave = srVideoRef.current;
      if (slave && slave.readyState >= 2 && !slave.seeking && !isSeekingRef.current) {
        const diff = Math.abs(slave.currentTime - current);
        const now = Date.now();
        if (diff > 0.35 && now - lastSyncTimeRef.current > 500) {
          lastSyncTimeRef.current = now;
          slave.currentTime = current;
        }
      }
    }
  };

  const handleSeek = (e) => {
    const seekTime = parseFloat(e.target.value);
    isSeekingRef.current = true;
    playbackPositionRef.current = seekTime;
    setCurrentTime(seekTime);
    if (originalVideoRef.current) originalVideoRef.current.currentTime = seekTime;
    if (srVideoRef.current) srVideoRef.current.currentTime = seekTime;
    setTimeout(() => {
      isSeekingRef.current = false;
    }, 150);
  };

  const toggleMute = () => {
    const nextMuted = !isMuted;
    setIsMuted(nextMuted);
    if (originalVideoRef.current) originalVideoRef.current.muted = nextMuted;
    if (srVideoRef.current) srVideoRef.current.muted = viewMode === 'split' ? true : nextMuted;
  };

  return (
    <div className="flex-1 flex flex-col lg:flex-row overflow-hidden bg-[#121316]">
      {/* Left Column: Upload, Probe & Controls */}
      <div className="w-full lg:w-[420px] xl:w-[460px] flex flex-col border-r border-[#262930] bg-[#16181d] overflow-y-auto custom-scrollbar">
        <div className="p-6 space-y-6">
          {/* Section Header */}
          <div>
            <div className="flex items-center space-x-2">
              <span className="h-2 w-2 rounded-full bg-[#b9f73a]" />
              <h2 className="text-xs font-mono font-semibold tracking-wider text-[#b9f73a] uppercase">
                Video Studio Engine
              </h2>
            </div>
            <h3 className="text-lg font-bold text-white mt-1">Siêu Phân Giải Video x4</h3>
            <p className="text-xs text-[#9aa2b1] mt-0.5">
              Tái tạo chuyển động và làm nét chi tiết video Anime bằng RealBasicVSR.
            </p>
          </div>

          {/* Upload Drop Zone */}
          <div
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all duration-200 ${
              selectedFile
                ? 'border-[#3a404d] bg-[#1b1e25]'
                : 'border-[#2d313b] hover:border-[#b9f73a] hover:bg-[#1a1d24] bg-[#171920]'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="flex flex-col items-center">
              <div className="w-10 h-10 rounded-lg bg-[#22252e] flex items-center justify-center text-[#b9f73a] mb-2 border border-[#2b2f3a]">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
                </svg>
              </div>
              <p className="text-sm font-medium text-white">
                {selectedFile ? selectedFile.name : 'Tải lên video MP4'}
              </p>
              <p className="text-[11px] text-[#7d8494] mt-1 font-mono">
                Chuẩn H.264 / CFR • Tối đa {videoLimits.max_duration_seconds}s • Max {videoLimits.max_upload_mb} MB
              </p>
            </div>
          </div>

          {/* Quick Demo Sample Option */}
          <div className="flex items-center justify-between text-xs font-mono px-1">
            <span className="text-[#6c7382]">CHƯA CÓ VIDEO SẴN?</span>
            <button
              type="button"
              onClick={handleLoadAnimeSample}
              disabled={isProbing || isProcessing}
              className="text-[#b9f73a] hover:text-[#c6fa52] underline underline-offset-4 flex items-center space-x-1 font-semibold transition-all disabled:opacity-50"
            >
              <span>⚡ Dùng thử mẫu Anime (2s)</span>
            </button>
          </div>

          {/* Probe Loading or Error */}
          {isProbing && (
            <div className="p-3.5 rounded-lg bg-[#1a1d24] border border-[#2a2e38] flex items-center space-x-3 text-xs text-[#9aa2b1]">
              <span className="w-4 h-4 border-2 border-[#b9f73a] border-t-transparent rounded-full animate-spin" />
              <span>Đang phân tích thông số khung hình & âm thanh...</span>
            </div>
          )}

          {probeError && (
            <div className="p-3.5 rounded-lg bg-red-950/40 border border-red-800/60 text-xs text-red-300">
              <p className="font-semibold">Video không hợp lệ:</p>
              <p className="mt-0.5 text-red-400">{probeError}</p>
            </div>
          )}

          {/* Video Metadata Inspection Card */}
          {probeData && (
            <div className="p-4 rounded-xl bg-[#1a1d24] border border-[#272b35] space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-mono text-[#7d8494]">THÔNG SỐ NGUỒN</span>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-[#242832] text-[#b9f73a] border border-[#343946]">
                  {probeData.video_codec.toUpperCase()} / {probeData.has_audio ? 'AAC' : 'NO AUDIO'}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="p-2.5 rounded bg-[#15171c] border border-[#22252e]">
                  <span className="text-[#6c7382] block text-[10px]">ĐỘ PHÂN GIẢI GỐC</span>
                  <span className="text-white font-semibold">{probeData.width} × {probeData.height}</span>
                </div>
                <div className="p-2.5 rounded bg-[#15171c] border border-[#22252e]">
                  <span className="text-[#6c7382] block text-[10px]">ĐỘ PHÂN GIẢI X4 (AI)</span>
                  <span className="text-[#b9f73a] font-semibold">{probeData.width * 4} × {probeData.height * 4}</span>
                </div>
                <div className="p-2.5 rounded bg-[#15171c] border border-[#22252e]">
                  <span className="text-[#6c7382] block text-[10px]">TỐC ĐỘ KHUNG HÌNH</span>
                  <span className="text-white font-semibold">{probeData.fps_float} FPS</span>
                </div>
                <div className="p-2.5 rounded bg-[#15171c] border border-[#22252e]">
                  <span className="text-[#6c7382] block text-[10px]">THỜI LƯỢNG / FRAMES</span>
                  <span className="text-white font-semibold">{probeData.duration_seconds}s ({probeData.total_frames}f)</span>
                </div>
              </div>

              {isLimitExceeded && (
                <div className="p-2.5 rounded bg-amber-950/40 border border-amber-800/60 text-[11px] text-amber-300">
                  ⚠️ Video vượt giới hạn: tối đa {videoLimits.max_duration_seconds}s, {videoLimits.max_frames} frame và {videoLimits.max_input_pixels.toLocaleString('vi-VN')} pixel/frame. Hãy chọn đoạn ngắn hơn hoặc giảm độ phân giải trước khi tải lên.
                </div>
              )}
            </div>
          )}

          {/* Configuration Settings */}
          <ModelSelector {...modelSelectorProps} disabled={isProcessing || isSubmitting} />
          {jobStatus?.checkpoint_filename && (
            <p className="text-[11px] font-mono text-[#9aa2b1]">
              Mô hình của tác vụ: {jobStatus.checkpoint_filename}
            </p>
          )}
          <div className="space-y-3">
            <label className="text-xs font-mono text-[#7d8494] block uppercase">
              Cấu hình Chunk Temporal Window
            </label>
            <div className="space-y-2">
              {CHUNK_PRESETS.map((preset) => (
                <button
                  key={preset.size}
                  type="button"
                  onClick={() => setChunkSize(preset.size)}
                  className={`w-full text-left px-3.5 py-2.5 rounded-lg text-xs font-mono transition-all border ${
                    chunkSize === preset.size
                      ? 'bg-[#222630] border-[#b9f73a] text-white'
                      : 'bg-[#181a20] border-[#252831] text-[#8e96a5] hover:border-[#383d4a]'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span>{preset.label}</span>
                    {chunkSize === preset.size && (
                      <span className="text-[#b9f73a] text-sm">✓</span>
                    )}
                  </div>
                </button>
              ))}
            </div>
            <p className="text-[11px] text-[#6b7280]">
              * Thuật toán Bounded Streaming tự động xóa frame trung gian khỏi RAM để tránh tràn bộ nhớ.
            </p>
          </div>

          {/* Action Button */}
          <button
            type="button"
            disabled={!canStart}
            onClick={handleStartUpscale}
            className={`w-full py-3.5 px-4 rounded-xl text-xs font-mono font-bold tracking-wider uppercase transition-all duration-200 shadow-lg ${
              !canStart
                ? 'bg-[#22252c] text-[#555b68] cursor-not-allowed border border-[#2a2e37]'
                : 'bg-[#b9f73a] hover:bg-[#c6fa52] text-[#121316] shadow-[#b9f73a]/15 hover:shadow-[#b9f73a]/25'
            }`}
          >
            {isProcessing ? 'Đang Xử Lý Video...' : '🚀 Bắt Đầu Siêu Phân Giải x4'}
          </button>

          {!canStart && !isProcessing && !isSubmitting && (
            <p className="text-[11px] text-[#9aa2b1] text-center">
              {isProbing
                ? 'Đang kiểm tra video trước khi chạy...'
                : probeError
                ? `Chưa thể chạy: ${probeError}`
                : !selectedFile
                ? 'Chọn video MP4 hoặc dùng mẫu Anime 2 giây để bắt đầu.'
                : isLimitExceeded
                ? 'Video vượt giới hạn xử lý hiện tại; xem cảnh báo ở mục Thông số nguồn.'
                : 'Chưa có thông số video hợp lệ để chạy.'}
            </p>
          )}

          {/* Progress Tracker Card */}
          {jobStatus && (
            <div className="p-4 rounded-xl bg-[#181b22] border border-[#272b35] space-y-3 font-mono">
              <div className="flex items-center justify-between text-xs">
                <span className="text-[#7d8494]">TRẠNG THÁI:</span>
                <span className="font-semibold text-[#b9f73a] uppercase">
                  {jobStatus.status === 'pending' && 'Đang xếp hàng...'}
                  {jobStatus.status === 'probing' && 'Đang phân tích...'}
                  {jobStatus.status === 'decoding' && 'Trích xuất khung hình...'}
                  {jobStatus.status === 'inferring' && 'RealBasicVSR x4...'}
                  {jobStatus.status === 'encoding' && 'Mã hóa H.264 & AAC...'}
                  {jobStatus.status === 'completed' && 'Hoàn tất!'}
                  {jobStatus.status === 'failed' && 'Thất bại'}
                </span>
              </div>

              {/* Progress bar */}
              <div className="w-full bg-[#121316] h-2.5 rounded-full overflow-hidden border border-[#242730]">
                <div
                  className="bg-gradient-to-r from-[#86d918] to-[#b9f73a] h-full transition-all duration-300 rounded-full"
                  style={{ width: `${jobStatus.progress_percent}%` }}
                />
              </div>

              <div className="flex justify-between text-[11px] text-[#8e96a5]">
                <span>Tiến độ: {jobStatus.progress_percent}%</span>
                {jobStatus.eta_seconds !== null && jobStatus.status === 'inferring' && (
                  <span>Còn lại: ~{jobStatus.eta_seconds}s</span>
                )}
              </div>

              {jobStatus.status === 'inferring' && (
                <div className="text-[11px] text-[#6c7382] border-t border-[#22252e] pt-2 flex justify-between">
                  <span>Khung hình: {jobStatus.current_frame} / {jobStatus.total_frames}</span>
                  <span>Chunks: {jobStatus.total_chunks}</span>
                </div>
              )}

              {jobStatus.status === 'completed' && jobStatus.timing && (
                <div className="text-[11px] text-[#8e96a5] border-t border-[#22252e] pt-2 space-y-1">
                  <div className="flex justify-between">
                    <span>Tổng thời gian:</span>
                    <span className="text-[#b9f73a] font-semibold">{jobStatus.timing.total_seconds}s</span>
                  </div>
                  <div className="flex justify-between text-[#6c7382]">
                    <span>AI Inference: {jobStatus.timing.infer_seconds}s</span>
                    <span>Encode: {jobStatus.timing.encode_seconds}s</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {errorMessage && (
            <div className="p-3.5 rounded-lg bg-red-950/40 border border-red-800/60 text-xs text-red-300">
              <p className="font-semibold">Lỗi xử lý:</p>
              <p className="mt-0.5 text-red-400">{errorMessage}</p>
            </div>
          )}
        </div>
      </div>

      {/* Right Column: Viewport & Synchronized Player */}
      <div className="flex-1 flex flex-col bg-[#0e0f12] overflow-hidden">
        {/* Top Control Bar */}
        <div className="min-h-12 border-b border-[#20232b] px-6 py-2 flex flex-wrap gap-2 items-center justify-between bg-[#14161b]">
          <div className="flex items-center space-x-3 text-xs font-mono">
            <span className="text-[#7d8494]">CHẾ ĐỘ XEM:</span>
            <div className="flex rounded bg-[#1c1f27] border border-[#272b35] p-0.5">
              <button
                type="button"
                onClick={() => handleViewModeChange('split')}
                className={`px-3 py-1 rounded text-[11px] transition-all ${
                  viewMode === 'split' ? 'bg-[#2b303c] text-white font-semibold' : 'text-[#7d8494] hover:text-white'
                }`}
              >
                Song Song (Split)
              </button>
              <button
                type="button"
                onClick={() => handleViewModeChange('single_sr')}
                disabled={!isCompleted}
                className={`px-3 py-1 rounded text-[11px] transition-all ${
                  viewMode === 'single_sr' ? 'bg-[#2b303c] text-white font-semibold' : 'text-[#7d8494] hover:text-white'
                } disabled:opacity-40 disabled:cursor-not-allowed`}
              >
                Chỉ Kết Quả (x4)
              </button>
              <button
                type="button"
                onClick={() => handleViewModeChange('single_orig')}
                className={`px-3 py-1 rounded text-[11px] transition-all ${
                  viewMode === 'single_orig' ? 'bg-[#2b303c] text-white font-semibold' : 'text-[#7d8494] hover:text-white'
                }`}
              >
                Chỉ Nguồn Gốc
              </button>
            </div>
          </div>

          <button
            type="button"
            disabled={!isCompleted || cropBusy}
            aria-pressed={cropMode}
            onClick={() => {
              pauseForCrop();
              handleViewModeChange('split');
              setCropMode((value) => !value);
              setCropError(null);
            }}
            className={`rounded border px-3 py-1.5 text-xs disabled:opacity-40 disabled:cursor-not-allowed ${cropMode ? 'border-[#b9f73a] text-[#b9f73a]' : 'border-[#343842] text-white'}`}
          >
            {cropMode ? 'Hủy chọn vùng' : 'Crop & Zoom'}
          </button>

          {/* Download button if completed */}
          {isCompleted && (
            <a
              href={getVideoDownloadUrl(activeJobId)}
              download
              className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-[#b9f73a] hover:bg-[#c6fa52] text-[#121316] text-xs font-mono font-bold transition-all shadow-md"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              <span>Tải Video x4 MP4</span>
            </a>
          )}
        </div>

        {cropMode && <p className="bg-[#14161b] px-6 py-2 text-xs text-[#b9f73a]">Video đã tạm dừng. Kéo một vùng trên video gốc để so sánh cùng frame trước/sau upscale.</p>}
        {cropBusy && <p role="status" className="px-6 py-2 text-xs text-[#9aa2b1]">Đang đồng bộ frame để mở vùng crop...</p>}
        {cropError && <p role="alert" className="px-6 py-2 text-xs text-red-300">{cropError}</p>}

        {/* Video Viewport Stage */}
        <div className="flex-1 flex items-center justify-center p-6 overflow-hidden relative">
          {!selectedFile ? (
            <div className="text-center max-w-sm text-[#5d6474]">
              <div className="w-16 h-16 rounded-2xl bg-[#171920] border border-[#22252e] flex items-center justify-center mx-auto mb-4 text-[#3e4452]">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <p className="text-sm font-medium text-white">Chưa có video nào được chọn</p>
              <p className="text-xs mt-1 text-[#7d8494]">
                Vui lòng chọn hoặc kéo thả tệp video MP4 vào khung bên trái để bắt đầu.
              </p>
            </div>
          ) : (
            <div className="w-full h-full flex flex-col justify-center items-center">
              {/* Dual / Single Player Layout */}
              <div className="w-full flex-1 flex items-center justify-center gap-4 overflow-hidden">
                {/* Original Video Frame */}
                {(viewMode === 'split' || viewMode === 'single_orig') && (
                  <div
                    onClick={togglePlay}
                    className="flex-1 min-w-0 h-full flex flex-col items-center justify-center relative bg-[#13151b] rounded-xl border border-[#22252e] overflow-hidden cursor-pointer"
                  >
                    <span className="absolute top-3 left-3 z-10 px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-black/60 backdrop-blur-md text-white border border-white/10 pointer-events-none">
                      GỐC: {probeData?.width}×{probeData?.height}
                    </span>
                    <video
                      ref={originalVideoRef}
                      src={sourceUrl || undefined}
                      onTimeUpdate={handleTimeUpdate}
                      onLoadedMetadata={handleLoadedMetadata}
                      onEnded={handleEnded}
                      playsInline
                      muted={isMuted}
                      className="absolute inset-0 block w-full h-full object-contain pointer-events-none"
                    />
                    {cropMode && isCompleted && !cropBusy && <CropSelection mediaRef={originalVideoRef} onStart={pauseForCrop} onComplete={compareVideoCrop} />}
                  </div>
                )}

                {/* Super-Resolved Video Frame */}
                {(viewMode === 'split' || viewMode === 'single_sr') && (
                  <div
                    onClick={isCompleted ? togglePlay : undefined}
                    className={`flex-1 min-w-0 h-full flex flex-col items-center justify-center relative bg-[#13151b] rounded-xl border border-[#22252e] overflow-hidden ${
                      isCompleted ? 'cursor-pointer' : ''
                    }`}
                  >
                    <span className="absolute top-3 left-3 z-10 px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-[#b9f73a]/20 backdrop-blur-md text-[#b9f73a] border border-[#b9f73a]/30 pointer-events-none">
                      SIÊU PHÂN GIẢI X4: {probeData ? `${probeData.width * 4}×${probeData.height * 4}` : '4X'}
                    </span>

                    {isCompleted && (
                      <video
                        ref={srVideoRef}
                        src={getVideoResultUrl(activeJobId)}
                        onTimeUpdate={handleTimeUpdate}
                        onLoadedMetadata={handleLoadedMetadata}
                        onEnded={handleEnded}
                        playsInline
                        muted={viewMode === 'split' ? true : isMuted}
                        className="absolute inset-0 block w-full h-full object-contain pointer-events-none"
                      />
                    )}

                    {!isCompleted && isProcessing && (
                      <div className="text-center p-6 space-y-3">
                        <span className="inline-block w-8 h-8 border-2 border-[#b9f73a] border-t-transparent rounded-full animate-spin" />
                        <p className="text-xs font-mono text-[#b9f73a]">Đang suy luận AI và xử lý khung hình...</p>
                        <p className="text-[11px] text-[#7d8494] font-mono">{jobStatus?.progress_percent}% hoàn tất</p>
                      </div>
                    )}

                    {!isCompleted && !isProcessing && (
                      <div className="text-center p-6 text-[#5d6474]">
                        <p className="text-xs font-mono">Video x4 sẽ hiển thị tại đây sau khi hoàn tất.</p>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Synchronized Player Bar */}
              <div className="w-full mt-4 p-3 rounded-xl bg-[#16181e] border border-[#242831] flex items-center space-x-4">
                <button
                  type="button"
                  onClick={togglePlay}
                  disabled={cropMode || cropBusy}
                  className="w-9 h-9 rounded-lg bg-[#222630] hover:bg-[#2b303c] text-white flex items-center justify-center transition-all border border-[#303542]"
                >
                  {isPlaying ? (
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M8 5v14l11-7z" />
                    </svg>
                  )}
                </button>

                {/* Timeline Seekbar */}
                <input
                  type="range"
                  min="0"
                  max={duration || 100}
                  step="0.01"
                  value={currentTime}
                  onChange={handleSeek}
                  disabled={cropBusy}
                  className="flex-1 accent-[#b9f73a] cursor-pointer"
                />

                <span className="text-xs font-mono text-[#8e96a5] min-w-[70px] text-right">
                  {currentTime.toFixed(1)}s / {duration.toFixed(1)}s
                </span>

                <button
                  type="button"
                  onClick={toggleMute}
                  className="p-2 rounded text-[#8e96a5] hover:text-white"
                >
                  {isMuted ? '🔇' : '🔊'}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
      {cropComparison && <CropCompareDialog comparison={cropComparison} onClose={() => setCropComparison(null)} />}
    </div>
  );
}
