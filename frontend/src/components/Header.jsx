import React from 'react';

export default function Header({ health, isCheckingHealth, activeMode = 'image', onModeChange }) {
  const isOnline = Boolean(health && health.status === 'ok');
  const device = health?.device ? health.device.toUpperCase() : 'UNKNOWN';
  const torchVer = health?.torch_version || '';

  return (
    <header className="px-6 py-3 flex items-center justify-between border-b border-[#262930] bg-[#17191d] select-none">
      <div className="flex items-center space-x-4">
        {/* Studio Icon */}
        <div className="p-2 rounded bg-[#22252c] border border-[#262930] text-[#b9f73a]">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        </div>

        <div>
          <div className="flex items-center space-x-2.5">
            <h1 className="text-sm font-semibold tracking-wide text-white uppercase font-mono">
              RealAnime-VSR
            </h1>
            <span className="px-2 py-0.5 text-[10px] uppercase font-mono tracking-wider font-semibold rounded bg-[#22252c] text-[#b9f73a] border border-[#343842]">
              AI Super-Resolution
            </span>
          </div>
          <p className="text-[11px] text-[#7d8494] font-mono">
            RealBasicVSR Anime V3 (x4 Enhanced)
          </p>
        </div>
      </div>

      {/* Mode Switcher Tabs */}
      <div className="flex items-center rounded-xl bg-[#121316] border border-[#262930] p-1 font-mono text-xs">
        <button
          type="button"
          onClick={() => onModeChange?.('image')}
          className={`flex items-center space-x-2 px-4 py-1.5 rounded-lg transition-all ${
            activeMode === 'image'
              ? 'bg-[#222630] text-[#b9f73a] font-bold shadow-md border border-[#343946]'
              : 'text-[#7d8494] hover:text-white'
          }`}
        >
          <span>📷</span>
          <span>IMAGE STUDIO</span>
        </button>
        <button
          type="button"
          onClick={() => onModeChange?.('video')}
          className={`flex items-center space-x-2 px-4 py-1.5 rounded-lg transition-all relative ${
            activeMode === 'video'
              ? 'bg-[#222630] text-[#b9f73a] font-bold shadow-md border border-[#343946]'
              : 'text-[#7d8494] hover:text-white'
          }`}
        >
          <span>🎬</span>
          <span>VIDEO STUDIO</span>
          <span className="px-1.5 py-0.2 rounded text-[9px] font-bold bg-[#b9f73a] text-black">
            4X
          </span>
        </button>
      </div>

      {/* Backend & Hardware Status Badges */}
      <div className="flex items-center space-x-3 text-xs font-mono">
        {/* Device Badge */}
        <div className="flex items-center space-x-1.5 px-3 py-1 rounded bg-[#1d1f25] border border-[#262930]">
          <span className="text-[#7d8494]">DEVICE:</span>
          <span className={`font-semibold ${device === 'CUDA' ? 'text-[#b9f73a]' : 'text-amber-400'}`}>
            {device}
          </span>
          {torchVer && (
            <span className="text-[10px] text-[#7d8494]">({torchVer})</span>
          )}
        </div>

        {/* Backend Online Status */}
        <div className="flex items-center space-x-2 px-3 py-1 rounded bg-[#1d1f25] border border-[#262930]">
          <span className="relative flex h-2 w-2">
            {isOnline ? (
              <>
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#b9f73a] opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-[#b9f73a]"></span>
              </>
            ) : (
              <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
            )}
          </span>
          <span className="text-[#7d8494]">API:</span>
          <span className={isOnline ? 'text-[#b9f73a] font-semibold' : 'text-red-400 font-semibold'}>
            {isCheckingHealth ? 'CHECKING...' : isOnline ? 'ONLINE' : 'OFFLINE'}
          </span>
        </div>
      </div>
    </header>
  );
}
