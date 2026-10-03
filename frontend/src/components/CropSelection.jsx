import { useRef, useState } from 'react';
import { containedMediaRect } from '../utils/cropComparison';

export default function CropSelection({ mediaRef, onStart, onComplete }) {
  const dragRef = useRef(null);
  const [box, setBox] = useState(null);

  function point(event, rect) {
    return {
      x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
    };
  }

  function move(event) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return null;
    const end = point(event, drag.rect);
    const selection = { x: Math.min(drag.start.x, end.x), y: Math.min(drag.start.y, end.y), width: Math.abs(drag.start.x - end.x), height: Math.abs(drag.start.y - end.y) };
    setBox({ left: drag.rect.left - drag.layer.left + selection.x * drag.rect.width, top: drag.rect.top - drag.layer.top + selection.y * drag.rect.height, width: selection.width * drag.rect.width, height: selection.height * drag.rect.height });
    return selection;
  }

  return (
    <div
      aria-label="Kéo để chọn vùng so sánh"
      className="absolute inset-0 z-40 cursor-crosshair touch-none"
      onClick={(event) => event.stopPropagation()}
      onPointerDown={(event) => {
        if (!event.isPrimary || event.button !== 0) return;
        const rect = containedMediaRect(mediaRef.current);
        if (!rect || event.clientX < rect.left || event.clientX > rect.left + rect.width || event.clientY < rect.top || event.clientY > rect.top + rect.height) return;
        event.preventDefault();
        event.stopPropagation();
        onStart?.();
        dragRef.current = { rect, layer: event.currentTarget.getBoundingClientRect(), start: point(event, rect), pointerId: event.pointerId };
        event.currentTarget.setPointerCapture(event.pointerId);
        move(event);
      }}
      onPointerMove={move}
      onPointerUp={(event) => {
        const selection = move(event);
        const rect = dragRef.current?.rect;
        dragRef.current = null;
        setBox(null);
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        if (selection && rect && selection.width * rect.width >= 8 && selection.height * rect.height >= 8) onComplete(selection);
      }}
      onPointerCancel={() => { dragRef.current = null; setBox(null); }}
      onLostPointerCapture={() => { dragRef.current = null; setBox(null); }}
    >
      {box && <div className="absolute border-2 border-[#b9f73a] bg-[#b9f73a]/10 pointer-events-none" style={{ ...box, boxShadow: '0 0 0 9999px rgba(0,0,0,0.35)' }} />}
    </div>
  );
}
