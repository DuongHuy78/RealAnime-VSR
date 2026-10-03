import test from 'node:test';
import assert from 'node:assert/strict';
import { captureComparison, containedMediaRect, sourceCrop } from '../src/utils/cropComparison.js';

test('selection excludes the letterbox bars around a 4:3 video', () => {
  const media = { videoWidth: 320, videoHeight: 240, getBoundingClientRect: () => ({ left: 10, top: 20, width: 600, height: 600 }) };
  assert.deepEqual(containedMediaRect(media), { left: 10, top: 95, width: 600, height: 450 });
});

test('selection at the bottom-right edge remains inside source pixels', () => {
  assert.deepEqual(sourceCrop({ x: 0.9, y: 0.9, width: 0.3, height: 0.3 }, 320, 240), { x: 288, y: 216, width: 32, height: 24 });
});

test('same source region maps exactly to the x4 result, with equal output canvases', () => {
  const original = { naturalWidth: 320, naturalHeight: 240 };
  const result = { naturalWidth: 1280, naturalHeight: 960 };
  const draws = [];
  const canvases = [];
  const previousDocument = globalThis.document;
  globalThis.document = { createElement: () => {
    const canvas = { getContext: () => ({ drawImage: (...args) => draws.push(args) }), toDataURL: () => 'data:image/png;base64,test' };
    canvases.push(canvas);
    return canvas;
  } };
  try {
    const crop = captureComparison(original, result, { x: 0.25, y: 0.25, width: 0.2, height: 0.2 });
    assert.deepEqual(crop.crop, { x: 80, y: 60, width: 64, height: 48 });
    assert.deepEqual(draws[0], [original, 80, 60, 64, 48, 0, 0, 256, 192]);
    assert.deepEqual(draws[1], [result, 320, 240, 256, 192, 0, 0, 256, 192]);
    assert.ok(canvases.every(canvas => canvas.width === 256 && canvas.height === 192));
  } finally {
    if (previousDocument === undefined) delete globalThis.document; else globalThis.document = previousDocument;
  }
});

test('video frames must be decoded before capture', () => {
  const loading = { videoWidth: 320, videoHeight: 240, readyState: 1 };
  const result = { videoWidth: 1280, videoHeight: 960, readyState: 2 };
  assert.throws(() => captureComparison(loading, result, { x: 0, y: 0, width: 1, height: 1 }), /chưa tải xong/);
});
