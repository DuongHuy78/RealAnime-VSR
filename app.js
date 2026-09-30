/**
 * The Post-Production Studio - Main Application Controller
 * Handles Studio Viewport, Comparison Slider, Timeline Scrubber, 
 * Real-time Inference Engine, Batch Queue, Deep Quality Inspector, 
 * Model Zoo, and Web Audio SFX.
 */

// Application State
const UPSCALE_FACTOR = 4;

const state = {
  sfxEnabled: true,
  isPlaying: false,
  isInferring: false,
  currentFrame: 184,
  totalFrames: 620,
  fps: 29.97,
  inferenceFps: 24.6,
  splitPosition: 38, // percentage 0-100
  activeZoom: '100',
  viewMode: 'split',
  loupeEnabled: false,
  isDraggingSlider: false,
  isDraggingScrubber: false,
  activeAssetId: 'sedan_drive',
  activeBatchSize: 12,
  startTime: null,
  elapsedSeconds: 134.35, // 00:02:14.35
  vram: 6.4,
  temp: 64,
  fpsHistory: [24.1, 24.4, 24.8, 25.1, 24.6, 24.3, 24.9, 25.4, 24.6],
  
  // Asset database
  assets: {
    'sedan_drive': {
      title: 'DOWNTOWN_DRIVE_002.MP4',
      filename: 'clip_sample_01.mp4',
      resolution: '1280 x 720 (720p)',
      fps: '29.97 fps',
      size: '24.5 MB',
      duration: '00:00:20 (620f)',
      totalFrames: 620,
      beforeBadge: '(720p Original)',
      afterBadge: '(4x Upscaled to 5120x2880)',
      beforeSrc: 'assets/sedan_before.jpg',
      afterSrc: 'assets/sedan_after.jpg',
      defaultModel: 'realesrgan_x4plus'
    },
    'cyberpunk_alley': {
      title: 'CYBER_ALLEY_NIGHT_08.MP4',
      filename: 'cyber_alley_night.mp4',
      resolution: '1920 x 1080 (1080p)',
      fps: '24.00 fps',
      size: '48.2 MB',
      duration: '00:00:25 (600f)',
      totalFrames: 600,
      beforeBadge: '(1080p High ISO)',
      afterBadge: '(2x Denoised & Sharpened)',
      beforeSrc: 'assets/cyber_before.jpg',
      afterSrc: 'assets/cyber_after.jpg',
      defaultModel: 'bsrgan_x4'
    },
    'vintage_portrait': {
      title: 'VINTAGE_PORTRAIT_RESTORE.MP4',
      filename: 'vintage_portrait_crop.mp4',
      resolution: '854 x 480 (480p)',
      fps: '25.00 fps',
      size: '12.8 MB',
      duration: '00:00:15 (375f)',
      totalFrames: 375,
      beforeBadge: '(480p Blurry Vintage)',
      afterBadge: '(GFPGAN Facial Restoration)',
      beforeSrc: 'assets/portrait_before.jpg',
      afterSrc: 'assets/portrait_after.jpg',
      defaultModel: 'gfpgan_v1.4'
    },
    'anime_sakuga': {
      title: 'SAKUGA_BATTLE_SEQUENCE.MP4',
      filename: 'sakuga_battle_720.mp4',
      resolution: '1280 x 720 (720p)',
      fps: '23.98 fps',
      size: '31.0 MB',
      duration: '00:00:18 (432f)',
      totalFrames: 432,
      beforeBadge: '(720p Anime Raw)',
      afterBadge: '(Real-CUGAN 4K Lineart)',
      beforeSrc: 'assets/anime_before.jpg',
      afterSrc: 'assets/anime_after.jpg',
      defaultModel: 'real_cugan_pro'
    }
  }
};

/* ========================================================================= */
/* Web Audio API Sound Synthesizer (No external audio file dependencies)     */
/* ========================================================================= */
class StudioSoundFX {
  constructor() {
    this.ctx = null;
  }

  init() {
    if (!this.ctx && typeof AudioContext !== 'undefined') {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioCtx();
    }
  }

  playClick() {
    if (!state.sfxEnabled) return;
    this.init();
    if (!this.ctx) return;

    try {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(600, this.ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(180, this.ctx.currentTime + 0.04);
      gain.gain.setValueAtTime(0.08, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.04);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + 0.04);
    } catch (e) {
      // Audio autoplay policy fallback
    }
  }

  playScrubTick() {
    if (!state.sfxEnabled) return;
    this.init();
    if (!this.ctx) return;

    try {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(800 + Math.random() * 200, this.ctx.currentTime);
      gain.gain.setValueAtTime(0.03, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.02);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + 0.02);
    } catch (e) {}
  }

  playChime() {
    if (!state.sfxEnabled) return;
    this.init();
    if (!this.ctx) return;

    try {
      const now = this.ctx.currentTime;
      [523.25, 659.25, 783.99, 1046.50].forEach((freq, i) => {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, now + i * 0.08);
        gain.gain.setValueAtTime(0.08, now + i * 0.08);
        gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.08 + 0.3);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start(now + i * 0.08);
        osc.stop(now + i * 0.08 + 0.35);
      });
    } catch (e) {}
  }
}

const sfx = new StudioSoundFX();

/* ========================================================================= */
/* Fixed x4 Output Resolution                                                */
/* ========================================================================= */
function getUpscaledDimensions(resolution) {
  const match = String(resolution).match(/(\d+)\s*x\s*(\d+)/i);
  if (!match) return null;

  return {
    width: Number(match[1]) * UPSCALE_FACTOR,
    height: Number(match[2]) * UPSCALE_FACTOR
  };
}

function updateResolutionPreview(resolution) {
  const inputText = document.getElementById('scaleInputText');
  const factorLabel = document.getElementById('scaleFactorLabel');
  const outputText = document.getElementById('scaleOutputText');
  const afterBadge = document.getElementById('afterBadgeText');
  const dimensions = getUpscaledDimensions(resolution);

  if (inputText) inputText.textContent = resolution;
  if (factorLabel) factorLabel.textContent = 'OUTPUT (' + UPSCALE_FACTOR + 'X):';
  if (outputText) {
    outputText.textContent = dimensions
      ? dimensions.width + ' x ' + dimensions.height
      : '?';
  }
  if (afterBadge) {
    afterBadge.textContent = dimensions
      ? '(4x Upscaled to ' + dimensions.width + ' x ' + dimensions.height + ')'
      : '(4x Upscaled)';
  }
}

function initTopBarControls() {
  const btnAudioToggle = document.getElementById('btnAudioToggle');
  if (!btnAudioToggle) return;

  btnAudioToggle.addEventListener('click', () => {
    state.sfxEnabled = !state.sfxEnabled;
    btnAudioToggle.innerHTML = state.sfxEnabled
      ? '<span class="text-[#b9f73a]">????</span> <span class="hidden sm:inline">SFX: ON</span>'
      : '<span class="text-gray-500">????</span> <span class="hidden sm:inline">SFX: OFF</span>';
    if (state.sfxEnabled) sfx.playClick();
  });
}

/* ========================================================================= */
/* Comparison Slider Logic (Smooth drag & percentage positioning)             */
/* ========================================================================= */
function initComparisonSlider() {
  const container = document.getElementById('viewportContainer');
  const handle = document.getElementById('sliderHandle');
  const beforeLayer = document.getElementById('beforeLayer');
  const afterImg = document.getElementById('afterImg');
  const beforeImg = document.getElementById('beforeImg');

  if (!container || !handle || !beforeLayer) return;

  function updateSlider(clientX) {
    const rect = container.getBoundingClientRect();
    let pos = (clientX - rect.left) / rect.width * 100;
    pos = Math.max(1, Math.min(99, pos)); // Clamp between 1% and 99%

    state.splitPosition = pos;
    beforeLayer.style.width = `${pos}%`;
    handle.style.left = `${pos}%`;
  }

  handle.addEventListener('pointerdown', (e) => {
    state.isDraggingSlider = true;
    handle.setPointerCapture(e.pointerId);
    sfx.playClick();
    e.preventDefault();
  });

  window.addEventListener('pointermove', (e) => {
    if (state.isDraggingSlider) {
      updateSlider(e.clientX);
    }
  });

  window.addEventListener('pointerup', (e) => {
    if (state.isDraggingSlider) {
      state.isDraggingSlider = false;
      try { handle.releasePointerCapture(e.pointerId); } catch(err){}
    }
  });

  // Clicking directly on viewport will move divider smoothly
  container.addEventListener('click', (e) => {
    if (e.target.closest('#sliderHandle') || state.loupeEnabled) return;
    updateSlider(e.clientX);
    sfx.playClick();
  });

  // View Mode selector (Split vs Diff Heatmap vs 4X Only)
  const viewModeBtns = document.querySelectorAll('.view-mode-btn');
  viewModeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      viewModeBtns.forEach(b => {
        b.classList.remove('active', 'bg-[#22252c]', 'text-white');
        b.classList.add('text-gray-400');
      });
      btn.classList.add('active', 'bg-[#22252c]', 'text-white');
      btn.classList.remove('text-gray-400');

      const mode = btn.dataset.mode;
      state.viewMode = mode;
      sfx.playClick();

      if (mode === 'split') {
        beforeLayer.classList.remove('hidden');
        handle.classList.remove('hidden');
        beforeLayer.style.width = `${state.splitPosition}%`;
        afterImg.classList.remove('diff-heatmap-filter');
      } else if (mode === 'diff') {
        beforeLayer.classList.add('hidden');
        handle.classList.add('hidden');
        afterImg.classList.add('diff-heatmap-filter');
      } else if (mode === 'after-only') {
        beforeLayer.classList.add('hidden');
        handle.classList.add('hidden');
        afterImg.classList.remove('diff-heatmap-filter');
      }
    });
  });

  // Zoom buttons
  const zoomBtns = document.querySelectorAll('.zoom-btn');
  zoomBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      zoomBtns.forEach(b => {
        b.classList.remove('active', 'text-[#b9f73a]', 'font-semibold');
        b.classList.add('text-gray-400');
      });
      btn.classList.add('active', 'text-[#b9f73a]', 'font-semibold');
      btn.classList.remove('text-gray-400');

      const zoom = btn.dataset.zoom;
      state.activeZoom = zoom;
      sfx.playClick();

      const scale = zoom === 'fit' ? 'scale-100' : (zoom === '200' ? 'scale-150' : 'scale-110');
      beforeImg.className = beforeImg.className.replace(/scale-\d+/g, scale);
      afterImg.className = afterImg.className.replace(/scale-\d+/g, scale);
    });
  });
}

/* ========================================================================= */
/* Pixel Loupe Magnifier Tool (8X circular zoom on hover)                    */
/* ========================================================================= */
function initPixelLoupe() {
  const btnLoupeToggle = document.getElementById('btnLoupeToggle');
  const loupeBubble = document.getElementById('loupeBubble');
  const loupeCanvas = document.getElementById('loupeCanvas');
  const viewport = document.getElementById('viewportContainer');
  const afterImg = document.getElementById('afterImg');

  if (!btnLoupeToggle || !loupeBubble || !viewport || !loupeCanvas) return;
  const ctx = loupeCanvas.getContext('2d');

  function toggleLoupe() {
    state.loupeEnabled = !state.loupeEnabled;
    sfx.playClick();
    btnLoupeToggle.classList.toggle('border-[#b9f73a]', state.loupeEnabled);
    btnLoupeToggle.classList.toggle('text-[#b9f73a]', state.loupeEnabled);
    loupeBubble.classList.toggle('hidden', !state.loupeEnabled);
  }

  btnLoupeToggle.addEventListener('click', toggleLoupe);

  viewport.addEventListener('mousemove', (e) => {
    if (!state.loupeEnabled) return;
    const rect = viewport.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    loupeBubble.style.left = `${x - 72}px`;
    loupeBubble.style.top = `${y - 72}px`;

    // Render magnified snapshot from active image onto canvas
    if (afterImg.complete && afterImg.naturalWidth > 0) {
      try {
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, 144, 144);
        const sourceX = (x / rect.width) * afterImg.naturalWidth - 16;
        const sourceY = (y / rect.height) * afterImg.naturalHeight - 16;
        ctx.drawImage(afterImg, Math.max(0, sourceX), Math.max(0, sourceY), 32, 32, 0, 0, 144, 144);
      } catch (err) {
        // Cross-origin image fallback
      }
    }
  });

  viewport.addEventListener('mouseleave', () => {
    if (state.loupeEnabled) loupeBubble.classList.add('hidden');
  });

  viewport.addEventListener('mouseenter', () => {
    if (state.loupeEnabled) loupeBubble.classList.remove('hidden');
  });
}

/* ========================================================================= */
/* Timeline Scrubber & Video Playback Simulation                             */
/* ========================================================================= */
function formatTimecode(frame, fps) {
  const totalSeconds = frame / fps;
  const h = String(Math.floor(totalSeconds / 3600)).padStart(2, '0');
  const m = String(Math.floor((totalSeconds % 3600) / 60)).padStart(2, '0');
  const s = String(Math.floor(totalSeconds % 60)).padStart(2, '0');
  const f = String(Math.floor(frame % fps)).padStart(2, '0');
  return `${h}:${m}:${s}.${f}`;
}

function initTimelineScrubber() {
  const btnPlayPause = document.getElementById('btnPlayPause');
  const playIcon = document.getElementById('playIcon');
  const playText = document.getElementById('playText');
  const btnStepBack = document.getElementById('btnStepBack');
  const btnStepForward = document.getElementById('btnStepForward');
  const btnLoopToggle = document.getElementById('btnLoopToggle');
  const timelineTrack = document.getElementById('timelineTrack');
  const timelineProgress = document.getElementById('timelineProgress');
  const scrubberNeedle = document.getElementById('scrubberNeedle');
  const timecodeDisplay = document.getElementById('timecodeDisplay');
  const frameNumberDisplay = document.getElementById('frameNumberDisplay');
  const btnSnapshot = document.getElementById('btnSnapshot');

  let loopEnabled = true;
  let playInterval = null;

  function setFrame(frameNum) {
    state.currentFrame = Math.max(0, Math.min(state.totalFrames, Math.round(frameNum)));
    const percent = (state.currentFrame / state.totalFrames) * 100;

    timelineProgress.style.width = `${percent}%`;
    scrubberNeedle.style.left = `${percent}%`;
    timecodeDisplay.textContent = formatTimecode(state.currentFrame, state.fps);
    frameNumberDisplay.textContent = `${state.currentFrame} / ${state.totalFrames} f`;
  }

  function togglePlay() {
    state.isPlaying = !state.isPlaying;
    sfx.playClick();

    if (state.isPlaying) {
      playIcon.textContent = '⏸';
      playText.textContent = 'PAUSE';
      btnPlayPause.classList.add('bg-[#b9f73a]', 'text-black');
      btnPlayPause.classList.remove('bg-[#22252d]', 'text-white');

      playInterval = setInterval(() => {
        let nextFrame = state.currentFrame + 1;
        if (nextFrame > state.totalFrames) {
          if (loopEnabled) {
            nextFrame = 0;
          } else {
            togglePlay();
            return;
          }
        }
        setFrame(nextFrame);
      }, 1000 / state.fps);
    } else {
      playIcon.textContent = '▶';
      playText.textContent = 'PLAY';
      btnPlayPause.classList.remove('bg-[#b9f73a]', 'text-black');
      btnPlayPause.classList.add('bg-[#22252d]', 'text-white');
      if (playInterval) clearInterval(playInterval);
    }
  }

  btnPlayPause.addEventListener('click', togglePlay);

  btnStepBack.addEventListener('click', () => {
    if (state.isPlaying) togglePlay();
    setFrame(state.currentFrame - 1);
    sfx.playScrubTick();
  });

  btnStepForward.addEventListener('click', () => {
    if (state.isPlaying) togglePlay();
    setFrame(state.currentFrame + 1);
    sfx.playScrubTick();
  });

  btnLoopToggle.addEventListener('click', () => {
    loopEnabled = !loopEnabled;
    sfx.playClick();
    btnLoopToggle.textContent = loopEnabled ? 'LOOP: ON' : 'LOOP: OFF';
    btnLoopToggle.classList.toggle('text-[#b9f73a]', loopEnabled);
    btnLoopToggle.classList.toggle('text-gray-400', !loopEnabled);
  });

  function scrubToClientX(clientX) {
    const rect = timelineTrack.getBoundingClientRect();
    const percent = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    setFrame(percent * state.totalFrames);
    sfx.playScrubTick();
  }

  timelineTrack.addEventListener('pointerdown', (e) => {
    state.isDraggingScrubber = true;
    timelineTrack.setPointerCapture(e.pointerId);
    scrubToClientX(e.clientX);
    e.preventDefault();
  });

  window.addEventListener('pointermove', (e) => {
    if (state.isDraggingScrubber) {
      scrubToClientX(e.clientX);
    }
  });

  window.addEventListener('pointerup', (e) => {
    if (state.isDraggingScrubber) {
      state.isDraggingScrubber = false;
      try { timelineTrack.releasePointerCapture(e.pointerId); } catch(err){}
    }
  });

  // Snapshot frame download
  if (btnSnapshot) {
    btnSnapshot.addEventListener('click', () => {
      sfx.playClick();
      showToast(`Snapshot for frame ${state.currentFrame} exported at 5120x2880.`);
    });
  }

  // Initial display setup
  setFrame(state.currentFrame);
}

/* ========================================================================= */
/* Real-time Inference Engine & Telemetry Sparkline                          */
/* ========================================================================= */
function initInferenceEngine() {
  const btnStart = document.getElementById('btnStartInference');
  const btnText = document.getElementById('inferenceBtnText');
  const statusBadge = document.getElementById('statusIndicatorBadge');
  const statTimer = document.getElementById('statTimer');
  const statFps = document.getElementById('statFps');
  const statFrames = document.getElementById('statFrames');
  const statVram = document.getElementById('statVram');
  const statTemp = document.getElementById('statTemp');
  const statProgressBar = document.getElementById('statProgressBar');
  const statEta = document.getElementById('statEta');
  const statPercentage = document.getElementById('statPercentage');
  const terminalLog = document.getElementById('terminalLog');
  const btnClearLog = document.getElementById('btnClearLog');
  const sparklineCanvas = document.getElementById('fpsSparkline');

  let inferenceInterval = null;
  let processedFrames = 184;

  function renderSparkline() {
    if (!sparklineCanvas) return;
    const ctx = sparklineCanvas.getContext('2d');
    const w = sparklineCanvas.width;
    const h = sparklineCanvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = '#b9f73a';
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    const pts = state.fpsHistory;
    const min = 20;
    const max = 30;

    pts.forEach((val, i) => {
      const x = (i / (pts.length - 1)) * w;
      const y = h - ((val - min) / (max - min)) * (h - 2) - 1;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  function addLog(msg, colorClass = '') {
    if (!terminalLog) return;
    const div = document.createElement('div');
    if (colorClass) div.className = colorClass;
    div.textContent = msg;
    terminalLog.appendChild(div);
    terminalLog.scrollTop = terminalLog.scrollHeight;
  }

  if (btnClearLog) {
    btnClearLog.addEventListener('click', () => {
      if (terminalLog) terminalLog.innerHTML = '<div class="text-gray-500">[CLEARED] Logs reset.</div>';
    });
  }

  function toggleInference() {
    state.isInferring = !state.isInferring;
    sfx.playClick();

    if (state.isInferring) {
      btnText.textContent = 'PAUSE INFERENCE';
      btnStart.classList.remove('bg-[#b9f73a]', 'hover:bg-[#c6ff47]');
      btnStart.classList.add('bg-amber-500', 'hover:bg-amber-400');
      statusBadge.textContent = 'RUNNING';
      statusBadge.className = 'text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-[#b9f73a] border border-emerald-500/40 neon-live-indicator';

      addLog(`[EXEC] Starting batch inference pipeline with ${state.activeBatchSize} frames/win.`);
      addLog(`[GPU] RTX 4090: VRAM allocation expanded to ${state.vram.toFixed(1)} GB.`);

      state.startTime = Date.now() - state.elapsedSeconds * 1000;

      inferenceInterval = setInterval(() => {
        // Advance frame
        processedFrames += 1;
        state.elapsedSeconds += 0.04;

        // Fluctuations
        const currentFps = 24.0 + (Math.random() * 2.2 - 1.1);
        state.inferenceFps = currentFps.toFixed(1);
        state.fpsHistory.shift();
        state.fpsHistory.push(parseFloat(state.inferenceFps));
        renderSparkline();

        state.vram = 6.4 + Math.sin(processedFrames / 10) * 0.4;
        state.temp = 64 + Math.floor((processedFrames % 40) / 10);

        // Update DOM
        const percent = Math.min(100, Math.round((processedFrames / state.totalFrames) * 100));
        statFrames.textContent = `${processedFrames} / ${state.totalFrames} frames`;
        statFps.textContent = `${state.inferenceFps} FPS`;
        statVram.textContent = `${state.vram.toFixed(1)} GB`;
        statTemp.textContent = `${state.temp}°C`;
        statProgressBar.style.width = `${percent}%`;
        statPercentage.textContent = `${percent}%`;

        // Elapsed time formatting
        const m = String(Math.floor(state.elapsedSeconds / 60)).padStart(2, '0');
        const s = String(Math.floor(state.elapsedSeconds % 60)).padStart(2, '0');
        const ms = String(Math.floor((state.elapsedSeconds * 100) % 100)).padStart(2, '0');
        statTimer.textContent = `00:${m}:${s}.${ms}`;

        // ETA calculation
        const remainingFrames = state.totalFrames - processedFrames;
        const etaSec = Math.max(0, Math.round(remainingFrames / parseFloat(state.inferenceFps)));
        const etaM = String(Math.floor(etaSec / 60)).padStart(2, '0');
        const etaS = String(etaSec % 60).padStart(2, '0');
        statEta.textContent = `ETA: 00:${etaM}:${etaS}`;

        // Periodic log emission
        if (processedFrames % 25 === 0) {
          addLog(`[INFER] Frame ${processedFrames}/${state.totalFrames} completed (Avg ${state.inferenceFps} FPS).`);
        }

        // Completion check
        if (processedFrames >= state.totalFrames) {
          clearInterval(inferenceInterval);
          state.isInferring = false;
          btnText.textContent = 'COMPLETED (START AGAIN)';
          btnStart.classList.remove('bg-amber-500', 'hover:bg-amber-400');
          btnStart.classList.add('bg-[#b9f73a]', 'hover:bg-[#c6ff47]');
          statusBadge.textContent = 'FINISHED';
          statusBadge.className = 'text-[9px] px-1.5 py-0.5 rounded bg-[#b9f73a]/20 text-[#b9f73a] border border-[#b9f73a]/40';
          
          const outputDimensions = getUpscaledDimensions(state.assets[state.activeAssetId].resolution);
          const outputSize = outputDimensions
            ? outputDimensions.width + 'x' + outputDimensions.height
            : '4X';
          addLog(`[SUCCESS] All ${state.totalFrames} frames super-resolved to ${outputSize}!`, 'text-[#b9f73a]');
          sfx.playChime();
          showToast(`Pipeline execution finished: ${state.totalFrames} frames processed!`);
          openExportModal();
        }
      }, 42); // ~24 fps simulation pace
    } else {
      btnText.textContent = 'START INFERENCE';
      btnStart.classList.remove('bg-amber-500', 'hover:bg-amber-400');
      btnStart.classList.add('bg-[#b9f73a]', 'hover:bg-[#c6ff47]');
      statusBadge.textContent = 'PAUSED';
      statusBadge.className = 'text-[9px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40';
      if (inferenceInterval) clearInterval(inferenceInterval);
      addLog('[PAUSED] Pipeline execution paused by user.');
    }
  }

  btnStart.addEventListener('click', toggleInference);

  // Batch size toggle buttons (7, 12, 16, 24)
  const batchBtns = document.querySelectorAll('.batch-btn');
  batchBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      batchBtns.forEach(b => {
        b.classList.remove('font-semibold', 'text-white', 'bg-[#22252d]', 'border', 'border-[#373c46]', 'shadow-inner');
        b.classList.add('text-gray-400');
      });
      btn.classList.add('font-semibold', 'text-white', 'bg-[#22252d]', 'border', 'border-[#373c46]', 'shadow-inner');
      btn.classList.remove('text-gray-400');

      const batch = parseInt(btn.dataset.batch, 10);
      state.activeBatchSize = batch;
      sfx.playClick();
      addLog(`[CONFIG] Sliding window batch size set to ${batch} frames.`);
    });
  });

  // Initial sparkline
  renderSparkline();
}

/* ========================================================================= */
/* Asset Library & Clip Switcher                                             */
/* ========================================================================= */
function loadAsset(assetId) {
  const asset = state.assets[assetId];
  if (!asset) return;

  state.activeAssetId = assetId;
  state.totalFrames = asset.totalFrames;

  // Update Left Metadata Card
  document.getElementById('metaFilename').textContent = asset.filename;
  document.getElementById('metaRes').textContent = asset.resolution;
  document.getElementById('metaFps').textContent = asset.fps;
  document.getElementById('metaSize').textContent = asset.size;
  document.getElementById('metaDuration').textContent = asset.duration;

  // Update Viewport Headers & Badges
  document.getElementById('viewportTitle').textContent = asset.title;
  document.getElementById('beforeBadgeText').textContent = asset.beforeBadge;
  document.getElementById('afterBadgeText').textContent = asset.afterBadge;
  document.getElementById('beforeImg').src = asset.beforeSrc;
  document.getElementById('afterImg').src = asset.afterSrc;

  // Output size is calculated automatically at a fixed x4 scale.
  updateResolutionPreview(asset.resolution);

  // Highlight card
  document.querySelectorAll('.asset-card').forEach(card => {
    const isActive = card.dataset.assetId === assetId;
    card.classList.toggle('active', isActive);
  });

  // Sync model dropdown if asset has preferred model
  if (asset.defaultModel) {
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect) modelSelect.value = asset.defaultModel;
  }

  sfx.playClick();
  showToast(`Loaded asset: ${asset.filename}`);
}

function initAssetManager() {
  const cards = document.querySelectorAll('.asset-card');
  cards.forEach(card => {
    card.addEventListener('click', () => {
      const assetId = card.dataset.assetId;
      if (assetId) loadAsset(assetId);
    });
  });

  // Custom File Upload (Drag & Drop or File Picker)
  const fileInput = document.getElementById('fileInput');
  const btnAssetUpload = document.getElementById('btnAssetUpload');
  const btnQuickUpload = document.getElementById('btnQuickUpload');

  function openFilePicker() {
    fileInput.click();
    sfx.playClick();
  }

  if (btnAssetUpload) btnAssetUpload.addEventListener('click', openFilePicker);
  if (btnQuickUpload) btnQuickUpload.addEventListener('click', openFilePicker);

  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const url = URL.createObjectURL(file);
      const customId = 'custom_' + Date.now();
      const isVideo = file.type.startsWith('video');

      // Create new asset item
      state.assets[customId] = {
        title: file.name.toUpperCase(),
        filename: file.name,
        resolution: isVideo ? '1920 x 1080 (HD)' : '1024 x 1024 (Custom)',
        fps: '30.00 fps',
        size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
        duration: isVideo ? '00:00:15 (450f)' : 'Single Frame (1f)',
        totalFrames: isVideo ? 450 : 1,
        beforeBadge: '(Original Upload)',
        afterBadge: '(4X Super-Resolution)',
        beforeSrc: url,
        afterSrc: url,
        defaultModel: 'realesrgan_x4plus'
      };

      // Add DOM Card to Input Assets
      const listContainer = document.getElementById('assetListContainer');
      const newCard = document.createElement('div');
      newCard.className = 'asset-card active border border-[#262930] bg-[#1a1c22] rounded p-1.5 cursor-pointer';
      newCard.dataset.assetId = customId;
      newCard.innerHTML = `
        <div class="w-full h-20 bg-black rounded overflow-hidden relative">
          <img alt="Custom Upload" class="w-full h-full object-cover" src="${url}"/>
          <span class="absolute top-1 right-1 bg-black/80 text-[8px] font-mono px-1 rounded text-[#b9f73a] border border-[#b9f73a]/30">USER</span>
        </div>
        <div class="mt-1.5">
          <div class="text-[11px] font-medium text-gray-200 truncate">${file.name}</div>
          <div class="text-[9px] text-gray-400 flex justify-between">
            <span>Uploaded Media</span>
            <span>${(file.size / (1024 * 1024)).toFixed(1)} MB</span>
          </div>
        </div>
      `;

      newCard.addEventListener('click', () => loadAsset(customId));
      listContainer.prepend(newCard);

      // Load uploaded asset and replace placeholder dimensions with file metadata.
      loadAsset(customId);

      const updateUploadedDimensions = (width, height) => {
        if (!width || !height) return;
        const resolution = width + ' x ' + height;
        state.assets[customId].resolution = resolution;
        if (state.activeAssetId === customId) {
          document.getElementById('metaRes').textContent = resolution;
          updateResolutionPreview(resolution);
        }
      };

      if (isVideo) {
        const videoProbe = document.createElement('video');
        videoProbe.preload = 'metadata';
        videoProbe.addEventListener('loadedmetadata', () => {
          updateUploadedDimensions(videoProbe.videoWidth, videoProbe.videoHeight);
        }, { once: true });
        videoProbe.src = url;
      } else {
        const imageProbe = new Image();
        imageProbe.addEventListener('load', () => {
          updateUploadedDimensions(imageProbe.naturalWidth, imageProbe.naturalHeight);
        }, { once: true });
        imageProbe.src = url;
      }
      showToast(`Custom media "${file.name}" imported into pipeline!`);
    });
  }
}

/* ========================================================================= */
/* Deep Quality Inspector & RGB Histogram Canvas Generator                   */
/* ========================================================================= */
function renderHistogram() {
  const canvas = document.getElementById('histogramCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background grid lines
  ctx.strokeStyle = '#22252c';
  ctx.lineWidth = 1;
  for (let y = 20; y < h; y += 25) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Draw 3 channels with smooth curve representation
  function drawChannel(color, offset, peak) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let x = 0; x < w; x += 4) {
      const norm = x / w;
      const val = Math.sin((norm + offset) * Math.PI) * Math.exp(-Math.pow((norm - peak) * 3, 2));
      const y = h - Math.max(0, val * (h - 10));
      ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fill();
  }

  // Composite channels
  ctx.globalCompositeOperation = 'screen';
  drawChannel('rgba(239, 68, 68, 0.45)', 0.1, 0.35);  // Red
  drawChannel('rgba(16, 185, 129, 0.45)', 0.15, 0.52); // Green
  drawChannel('rgba(59, 130, 246, 0.45)', 0.05, 0.65); // Blue
  ctx.globalCompositeOperation = 'source-over';
}

function initQualityInspector() {
  const btnRecalc = document.getElementById('btnRecalcMetrics');
  const probeCoords = document.getElementById('pixelProbeCoords');

  if (btnRecalc) {
    btnRecalc.addEventListener('click', () => {
      sfx.playClick();
      btnRecalc.textContent = '⏳ Calculating...';
      setTimeout(() => {
        btnRecalc.textContent = '🔄 Recalculate Metrics';
        renderHistogram();
        sfx.playChime();
        showToast('PSNR / SSIM tensors re-evaluated with Sobel spatial filter.');
      }, 500);
    });
  }

  // Inspector dual view mouseover probe
  const dualViewContainer = document.querySelector('#screen-inspector .grid.grid-cols-2');
  if (dualViewContainer && probeCoords) {
    dualViewContainer.addEventListener('mousemove', (e) => {
      const rect = dualViewContainer.getBoundingClientRect();
      const x = Math.round((e.clientX - rect.left) * 2.5);
      const y = Math.round((e.clientY - rect.top) * 2.5);
      const r = Math.min(255, Math.max(40, 150 + Math.round(Math.sin(x / 50) * 80)));
      const g = Math.min(255, Math.max(50, 180 + Math.round(Math.cos(y / 40) * 60)));
      const b = Math.min(255, Math.max(60, 210 + Math.round(Math.sin((x+y)/60) * 45)));
      probeCoords.textContent = `Probe: X: ${x}, Y: ${y} | R: ${r} G: ${g} B: ${b}`;
    });
  }
}

/* ========================================================================= */
/* Model Zoo Filtering & Deployment                                          */
/* ========================================================================= */
function initModelZoo() {
  const filterBtns = document.querySelectorAll('.model-filter-btn');
  const loadBtns = document.querySelectorAll('.btn-load-model');
  const modelSelect = document.getElementById('modelSelect');

  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => {
        b.classList.remove('active', 'bg-[#22252c]', 'text-white', 'border-[#343842]');
        b.classList.add('text-gray-400', 'bg-[#17191d]', 'border-[#262930]');
      });
      btn.classList.add('active', 'bg-[#22252c]', 'text-white', 'border-[#343842]');
      btn.classList.remove('text-gray-400', 'bg-[#17191d]', 'border-[#262930]');
      sfx.playClick();
    });
  });

  loadBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const modelVal = btn.dataset.model;
      if (modelSelect && modelVal) {
        modelSelect.value = modelVal;
      }

      loadBtns.forEach(b => {
        b.className = 'btn-load-model w-full py-1.5 bg-[#1d1f25] hover:bg-[#252830] text-gray-200 border border-[#262930] rounded text-xs font-semibold mt-3';
        b.textContent = 'Load into Pipeline';
      });

      btn.className = 'btn-load-model w-full py-1.5 bg-[#22252c] hover:bg-[#2e323b] text-[#b9f73a] border border-[#343842] rounded text-xs font-semibold mt-3';
      btn.textContent = '✓ Active in Pipeline';

      sfx.playChime();
      showToast(`Model [${modelVal}] loaded into active execution pipeline.`);
    });
  });

  const btnImportModel = document.getElementById('btnImportModel');
  if (btnImportModel) {
    btnImportModel.addEventListener('click', () => {
      sfx.playClick();
      showToast('Select local .onnx / .pth weights checkpoint to register in Zoo.');
    });
  }
}

/* ========================================================================= */
/* Batch Queue Tasks Manager                                                 */
/* ========================================================================= */
function initBatchQueue() {
  const btnNewBatchJob = document.getElementById('btnNewBatchJob');
  const btnPauseAll = document.getElementById('btnPauseAllQueue');
  const tableBody = document.getElementById('batchTableBody');

  if (btnNewBatchJob && tableBody) {
    btnNewBatchJob.addEventListener('click', () => {
      sfx.playClick();
      const newId = `#JOB-${1085 + Math.floor(Math.random() * 50)}`;
      const tr = document.createElement('tr');
      tr.className = 'hover:bg-[#1d1f25]/50 transition-colors animate-fadeIn';
      tr.innerHTML = `
        <td class="p-3 text-gray-400">${newId}</td>
        <td class="p-3 text-white font-medium">batch_render_stream_${Date.now().toString().slice(-4)}.mp4</td>
        <td class="p-3 text-[#b9f73a]">RealESRGAN_x4plus</td>
        <td class="p-3 text-gray-300">1080p → 4K (4X)</td>
        <td class="p-3 w-48">
          <div class="w-full bg-[#121316] h-1.5 rounded-full overflow-hidden border border-[#262930]">
            <div class="bg-[#b9f73a] h-full w-[8%]"></div>
          </div>
          <div class="text-[9px] text-gray-400 mt-1 flex justify-between">
            <span>40/500 f</span>
            <span>8%</span>
          </div>
        </td>
        <td class="p-3 text-[#b9f73a]">25.2 FPS</td>
        <td class="p-3">
          <span class="bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded text-[9px] uppercase font-bold">Rendering</span>
        </td>
        <td class="p-3 text-right space-x-1.5">
          <button class="hover:text-white text-gray-400 p-1" title="Pause">⏸</button>
          <button class="hover:text-white text-[#b9f73a] p-1 font-sans text-xs">Inspect</button>
        </td>
      `;
      tableBody.prepend(tr);
      showToast(`Batch task ${newId} dispatched to GPU worker queue.`);
    });
  }

  if (btnPauseAll) {
    btnPauseAll.addEventListener('click', () => {
      sfx.playClick();
      const isPaused = btnPauseAll.textContent.includes('Resume');
      btnPauseAll.textContent = isPaused ? 'Pause Queue' : 'Resume Queue';
      showToast(isPaused ? 'Batch queue resumed.' : 'All batch workers paused.');
    });
  }
}

/* ========================================================================= */
/* Modals & Keyboard Shortcuts                                               */
/* ========================================================================= */
function openExportModal() {
  const modal = document.getElementById('exportModal');
  const targetResolution = document.getElementById('exportTargetResolution');
  const dimensions = getUpscaledDimensions(state.assets[state.activeAssetId].resolution);

  if (targetResolution) {
    targetResolution.textContent = dimensions
      ? dimensions.width + ' x ' + dimensions.height + ' (4X)'
      : 'Output size unavailable';
  }

  if (modal) {
    modal.classList.remove('hidden');
    sfx.playClick();
  }
}

function closeExportModal() {
  const modal = document.getElementById('exportModal');
  if (modal) modal.classList.add('hidden');
}

function initModals() {
  const btnOpenExport = document.getElementById('btnOpenExport');
  const btnCloseExportModal = document.getElementById('btnCloseExportModal');
  const btnCancelExport = document.getElementById('btnCancelExport');
  const btnConfirmExport = document.getElementById('btnConfirmExport');

  if (btnOpenExport) btnOpenExport.addEventListener('click', openExportModal);
  if (btnCloseExportModal) btnCloseExportModal.addEventListener('click', closeExportModal);
  if (btnCancelExport) btnCancelExport.addEventListener('click', closeExportModal);

  if (btnConfirmExport) {
    btnConfirmExport.addEventListener('click', () => {
      sfx.playChime();
      closeExportModal();
      showToast('Export job queued! Rendering MP4 ProRes container...');
    });
  }

  // Keyboard Shortcuts Modal
  const shortcutsModal = document.getElementById('shortcutsModal');
  const btnShortcuts = document.getElementById('btnShortcuts');
  const btnCloseShortcuts = document.getElementById('btnCloseShortcuts');

  if (btnShortcuts && shortcutsModal) {
    btnShortcuts.addEventListener('click', () => {
      shortcutsModal.classList.remove('hidden');
      sfx.playClick();
    });
  }

  if (btnCloseShortcuts && shortcutsModal) {
    btnCloseShortcuts.addEventListener('click', () => {
      shortcutsModal.classList.add('hidden');
    });
  }

  // Global Keyboard Navigation
  window.addEventListener('keydown', (e) => {
    // If typing in input, don't trigger shortcuts
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;

    if (e.code === 'Space') {
      e.preventDefault();
      document.getElementById('btnPlayPause')?.click();
    } else if (e.key === 'j' || e.key === 'J' || e.key === 'ArrowLeft') {
      document.getElementById('btnStepBack')?.click();
    } else if (e.key === 'l' || e.key === 'L' || e.key === 'ArrowRight') {
      document.getElementById('btnStepForward')?.click();
    } else if (e.key === 'i' || e.key === 'I') {
      document.getElementById('btnStartInference')?.click();
    } else if (e.key === 'm' || e.key === 'M') {
      document.getElementById('btnLoupeToggle')?.click();
    } else if (e.key === '?' || (e.shiftKey && e.key === '/')) {
      shortcutsModal?.classList.toggle('hidden');
    }
  });

  // Settings Save Button Toast
  const btnSaveSettings = document.getElementById('btnSaveEngineSettings');
  if (btnSaveSettings) {
    btnSaveSettings.addEventListener('click', () => {
      sfx.playChime();
      showToast('Engine configuration saved and CUDA cache pinned.');
    });
  }
}

/* ========================================================================= */
/* Toast Notification Helper                                                 */
/* ========================================================================= */
let toastTimeout = null;
function showToast(message) {
  const toast = document.getElementById('toastNotification');
  const toastMsg = document.getElementById('toastMessage');
  if (!toast || !toastMsg) return;

  toastMsg.textContent = message;
  toast.classList.remove('translate-y-20', 'opacity-0');
  toast.classList.add('translate-y-0', 'opacity-100');

  if (toastTimeout) clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    toast.classList.remove('translate-y-0', 'opacity-100');
    toast.classList.add('translate-y-20', 'opacity-0');
  }, 3200);
}

/* ========================================================================= */
/* DOM Ready Initialization                                                  */
/* ========================================================================= */
document.addEventListener('DOMContentLoaded', () => {
  updateResolutionPreview(state.assets[state.activeAssetId].resolution);
  initTopBarControls();
  initComparisonSlider();
  initPixelLoupe();
  initTimelineScrubber();
  initInferenceEngine();
  initAssetManager();
  initQualityInspector();
  initModelZoo();
  initBatchQueue();
  initModals();

  console.log('%c[OROHANCE STUDIO]%c Initialized successfully with GPU acceleration.', 'color: #b9f73a; font-weight: bold;', 'color: #9ca3af;');
});
