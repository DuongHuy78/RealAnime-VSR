export function mediaSize(media) {
  return { width: media?.videoWidth || media?.naturalWidth || 0, height: media?.videoHeight || media?.naturalHeight || 0 };
}

// object-contain can leave black bars: selection coordinates belong to the image,
// not to the surrounding player card.
export function containedMediaRect(media) {
  const size = mediaSize(media);
  if (!size.width || !size.height) return null;
  const box = media.getBoundingClientRect();
  const scale = Math.min(box.width / size.width, box.height / size.height);
  if (!scale) return null;
  const width = size.width * scale;
  const height = size.height * scale;
  return { left: box.left + (box.width - width) / 2, top: box.top + (box.height - height) / 2, width, height };
}

export function sourceCrop(selection, width, height) {
  const x = Math.max(0, Math.min(width - 1, Math.floor(selection.x * width)));
  const y = Math.max(0, Math.min(height - 1, Math.floor(selection.y * height)));
  const right = Math.max(x + 1, Math.min(width, Math.ceil((selection.x + selection.width) * width)));
  const bottom = Math.max(y + 1, Math.min(height, Math.ceil((selection.y + selection.height) * height)));
  return { x, y, width: right - x, height: bottom - y };
}

export function captureComparison(original, result, selection) {
  const beforeSize = mediaSize(original);
  const afterSize = mediaSize(result);
  if (!beforeSize.width || !afterSize.width || (original.readyState !== undefined && original.readyState < 2) || (result.readyState !== undefined && result.readyState < 2)) {
    throw new Error('Ảnh hoặc khung hình chưa tải xong. Vui lòng thử lại.');
  }
  if (Math.abs(beforeSize.width / beforeSize.height - afterSize.width / afterSize.height) > 0.01) {
    throw new Error('Nguồn và kết quả khác tỷ lệ khung hình nên chưa thể crop cùng vùng.');
  }
  const crop = sourceCrop(selection, beforeSize.width, beforeSize.height);
  const sx = afterSize.width / beforeSize.width;
  const sy = afterSize.height / beforeSize.height;
  const resultX = Math.round(crop.x * sx);
  const resultY = Math.round(crop.y * sy);
  const width = Math.round((crop.x + crop.width) * sx) - resultX;
  const height = Math.round((crop.y + crop.height) * sy) - resultY;
  function draw(media, region) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Trình duyệt không hỗ trợ xem vùng crop.');
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';
    context.drawImage(media, region.x, region.y, region.width, region.height, 0, 0, width, height);
    return canvas.toDataURL('image/png');
  }
  return {
    before: draw(original, crop),
    after: draw(result, { x: resultX, y: resultY, width, height }),
    width, height, crop,
  };
}

export function seekVideoFrame(video, time) {
  video.pause();
  if (!video.seeking && video.readyState >= 2 && Math.abs(video.currentTime - time) < 0.001) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => finish(new Error('Không thể đồng bộ frame để crop. Vui lòng thử lại.')), 5000);
    function finish(error) {
      clearTimeout(timer);
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('error', onError);
      if (error) reject(error); else resolve();
    }
    function onSeeked() { finish(); }
    function onError() { finish(new Error('Không thể đọc khung hình video.')); }
    video.addEventListener('seeked', onSeeked);
    video.addEventListener('error', onError);
    video.currentTime = time;
  });
}
