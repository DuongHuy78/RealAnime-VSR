/**
 * API client for interacting with the FastAPI backend.
 */

export async function fetchHealth() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    console.warn('Backend /api/health unavailable:', err.message);
    return null;
  }
}

export async function fetchModels() {
  try {
    const res = await fetch('/api/models');
    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    const data = await res.json();
    return data.models || [];
  } catch (err) {
    console.warn('Backend /api/models unavailable:', err.message);
    return [];
  }
}

export async function activateModel(modelId) {
  let res;
  try {
    res = await fetch(`/api/models/${encodeURIComponent(modelId)}/activate`, { method: 'POST' });
  } catch {
    throw new Error('Không thể kết nối backend để đổi mô hình. Kiểm tra FastAPI và thử lại.');
  }
  if (!res.ok) {
    const error = await res.json().catch(() => null);
    throw new Error(error?.detail || `Không thể đổi mô hình (${res.status}).`);
  }
  return res.json();
}

export async function upscaleImage(file, modelId = 'realbasicvsr_anime_v3', signal = null) {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('model_id', modelId);

  let res;
  try {
    res = await fetch('/api/upscale', {
      method: 'POST',
      body: formData,
      signal,
    });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw new Error('Không thể kết nối backend. Kiểm tra FastAPI và thử lại.');
  }

  if (!res.ok) {
    let errorDetail = `Lỗi hệ thống (${res.status})`;
    try {
      const errJson = await res.json();
      if (errJson && errJson.detail) {
        errorDetail = Array.isArray(errJson.detail)
          ? errJson.detail.map((item) => item.msg || String(item)).join('; ')
          : String(errJson.detail);
      }
    } catch {
      // Giữ thông báo mặc định nếu không đọc được JSON
    }
    throw new Error(errorDetail);
  }

  return await res.json();
}

/**
 * Probe a video file to inspect dimensions, duration, FPS, and compliance.
 */
export async function probeVideo(file, signal = null) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('/api/video/probe', {
    method: 'POST',
    body: formData,
    signal,
  });

  if (!res.ok) {
    let errorDetail = `Lỗi phân tích video (${res.status})`;
    try {
      const errJson = await res.json();
      if (errJson && errJson.detail) {
        errorDetail = Array.isArray(errJson.detail)
          ? errJson.detail.map((item) => item.msg || String(item)).join('; ')
          : String(errJson.detail);
      }
    } catch {}
    throw new Error(errorDetail);
  }

  return await res.json();
}

/**
 * Submit video for asynchronous super-resolution job (HTTP 202).
 */
export async function submitVideoUpscale(file, chunkSize = 7, overlap = 2, signal = null, modelId = '') {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('chunk_size', String(chunkSize));
  formData.append('overlap', String(overlap));
  formData.append('model_id', modelId);

  const res = await fetch('/api/video/upscale', {
    method: 'POST',
    body: formData,
    signal,
  });

  if (!res.ok) {
    let errorDetail = `Lỗi gửi video (${res.status})`;
    try {
      const errJson = await res.json();
      if (errJson && errJson.detail) {
        errorDetail = Array.isArray(errJson.detail)
          ? errJson.detail.map((item) => item.msg || String(item)).join('; ')
          : String(errJson.detail);
      }
    } catch {}
    throw new Error(errorDetail);
  }

  return await res.json();
}

/**
 * Poll job status.
 */
export async function fetchVideoJobStatus(jobId, signal = null) {
  const res = await fetch(`/api/video/jobs/${jobId}`, { signal });
  if (!res.ok) {
    throw new Error(`Không thể lấy trạng thái job (${res.status})`);
  }
  return await res.json();
}

/**
 * Return result URLs.
 */
export function getVideoResultUrl(jobId) {
  return `/api/video/results/${jobId}`;
}

export function getVideoDownloadUrl(jobId) {
  return `/api/video/results/${jobId}/download`;
}

