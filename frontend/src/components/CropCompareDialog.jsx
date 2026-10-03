import { useEffect, useRef, useState } from 'react';

export default function CropCompareDialog({ comparison, onClose, isDemo = false }) {
  const dialogRef = useRef(null);
  const beforeRef = useRef(null);
  const afterRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const syncingRef = useRef(false);

  useEffect(() => {
    dialogRef.current.showModal();
  }, []);

  function fit() {
    const pane = beforeRef.current;
    setZoom(Math.max(0.05, Math.min(1, (pane.clientWidth - 32) / comparison.width, (pane.clientHeight - 32) / comparison.height)));
  }

  function syncScroll(event, other) {
    if (syncingRef.current || !other.current) return;
    syncingRef.current = true;
    other.current.scrollLeft = event.currentTarget.scrollLeft;
    other.current.scrollTop = event.currentTarget.scrollTop;
    requestAnimationFrame(() => { syncingRef.current = false; });
  }

  const buttonClass = 'rounded border border-[#343842] bg-[#22252c] px-3 py-1.5 text-xs text-white hover:border-[#b9f73a] disabled:opacity-40';
  const panels = [
    { label: 'Trước upscale • Gốc phóng cùng kích thước', image: comparison.before, ref: beforeRef, other: afterRef },
    { label: isDemo ? 'Sau • Ảnh minh họa DEMO' : 'Sau upscale • Kết quả x4', image: comparison.after, ref: afterRef, other: beforeRef },
  ];

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby="crop-dialog-title"
      onCancel={(event) => { event.preventDefault(); onClose(); }}
      onClick={(event) => {
        if (event.target !== event.currentTarget) return;
        const rect = event.currentTarget.getBoundingClientRect();
        if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) onClose();
      }}
      className="crop-compare-dialog"
    >
      <div className="flex h-full min-h-0 flex-col">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#343842] p-4">
          <div>
            <h2 id="crop-dialog-title" className="text-base font-semibold text-white">So sánh vùng crop</h2>
            <p className="mt-1 text-xs text-[#9aa2b1]">Vùng gốc: {comparison.crop.width}×{comparison.crop.height} px tại ({comparison.crop.x}, {comparison.crop.y}) → {comparison.width}×{comparison.height} px{comparison.time !== undefined ? ` • Frame tại ${comparison.time.toFixed(3)}s` : ''}</p>
          </div>
          <button type="button" onClick={onClose} className={buttonClass} aria-label="Đóng so sánh crop" autoFocus>Đóng ×</button>
        </div>
        <div className="flex flex-wrap items-center gap-2 border-b border-[#343842] px-4 py-3">
          <span className="text-xs text-[#9aa2b1]">Zoom hai ảnh:</span>
          <button type="button" className={buttonClass} onClick={() => setZoom((value) => Math.max(0.05, value / 1.5))} aria-label="Thu nhỏ">−</button>
          <span className="min-w-14 text-center text-sm text-[#b9f73a]">{Math.round(zoom * 100)}%</span>
          <button type="button" className={buttonClass} onClick={() => setZoom((value) => Math.min(8, value * 1.5))} aria-label="Phóng to">+</button>
          <button type="button" className={buttonClass} onClick={fit}>Vừa khung</button>
          {[1, 2, 4].map((value) => <button key={value} type="button" className={buttonClass} onClick={() => setZoom(value)}>{value * 100}%</button>)}
          <span className="text-[11px] text-[#9aa2b1]">100% = 1 pixel kết quả / 1 pixel hiển thị. Cuộn để xem cùng vị trí.</span>
        </div>
        <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 p-4 sm:grid-cols-2">
          {panels.map((panel) => (
            <section key={panel.label} className="flex min-h-0 min-w-0 flex-col overflow-hidden rounded-lg border border-[#343842]">
              <h3 className="shrink-0 truncate bg-[#22252c] px-3 py-2 text-xs text-white" title={panel.label}>{panel.label}</h3>
              <div ref={panel.ref} onScroll={(event) => syncScroll(event, panel.other)} className="min-h-0 flex-1 overflow-auto bg-[#0d0e11]">
                <div className="flex min-h-full min-w-full w-max items-center justify-center p-4">
                  <img src={panel.image} alt={panel.label} draggable={false} style={{ width: comparison.width * zoom, height: comparison.height * zoom, maxWidth: 'none', flexShrink: 0, imageRendering: zoom > 1 ? 'pixelated' : 'auto' }} />
                </div>
              </div>
            </section>
          ))}
        </div>
      </div>
    </dialog>
  );
}
