import React, { useState, useEffect, useRef, useCallback } from 'react';
import Header from './components/Header';
import UploadPanel from './components/UploadPanel';
import CompareSlider from './components/CompareSlider';
import VideoStudio from './components/VideoStudio';
import { activateModel, fetchHealth, fetchModels, upscaleImage } from './api/client';

const DEMO_SAMPLES = [
  {
    id: 'anime',
    label: 'Anime Character',
    before: '/demo/anime_before.jpg',
    after: '/demo/anime_after.jpg',
  },
  {
    id: 'cyber',
    label: 'Cyber Cityscape',
    before: '/demo/cyber_before.jpg',
    after: '/demo/cyber_after.jpg',
  },
  {
    id: 'portrait',
    label: 'Studio Portrait',
    before: '/demo/portrait_before.jpg',
    after: '/demo/portrait_after.jpg',
  },
  {
    id: 'sedan',
    label: 'Automotive Detail',
    before: '/demo/sedan_before.jpg',
    after: '/demo/sedan_after.jpg',
  },
];

function createDemoMeta(sample) {
  return {
    file: null,
    url: sample.before,
    name: `${sample.id}_sample.jpg`,
    size: 'Ảnh minh họa',
    width: null,
    height: null,
    outWidth: null,
    outHeight: null,
  };
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(true);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [isSwitchingModel, setIsSwitchingModel] = useState(false);
  const [modelError, setModelError] = useState(null);
  const modelSwitchRef = useRef(false);
  const statusRequestRef = useRef(0);
  const [imageMeta, setImageMeta] = useState(() => createDemoMeta(DEMO_SAMPLES[0]));
  const [comparisonUrl, setComparisonUrl] = useState(DEMO_SAMPLES[0].after);
  const [isDemoSample, setIsDemoSample] = useState(true);
  const [isUpscaling, setIsUpscaling] = useState(false);
  const [upscaleError, setUpscaleError] = useState(null);
  const [resultMeta, setResultMeta] = useState(null);
  const [activeMode, setActiveMode] = useState('image'); // 'image' | 'video'
  const requestRef = useRef(null);
  const requestVersionRef = useRef(0);

  const refreshBackendData = useCallback(async () => {
    const version = ++statusRequestRef.current;
    const [healthData, modelsData] = await Promise.all([fetchHealth(), fetchModels()]);
    if (version !== statusRequestRef.current) return;
    setHealth(healthData);
    setIsCheckingHealth(false);
    setModels(modelsData);
    setSelectedModel((current) => modelsData.some((model) => model.id === current)
      ? current
      : (modelsData.find((model) => model.loaded)?.id || modelsData[0]?.id || ''));
  }, []);

  useEffect(() => {
    refreshBackendData();
    const interval = setInterval(refreshBackendData, 15000);
    return () => {
      statusRequestRef.current += 1;
      clearInterval(interval);
    };
  }, [refreshBackendData]);

  const handleSelectModel = async (modelId) => {
    if (modelSwitchRef.current || isUpscaling) return;
    const previousModel = selectedModel;
    modelSwitchRef.current = true;
    statusRequestRef.current += 1;
    setSelectedModel(modelId);
    setIsSwitchingModel(true);
    setIsCheckingHealth(true);
    setModelError(null);
    try {
      await activateModel(modelId);
    } catch (error) {
      setSelectedModel(previousModel);
      setModelError(error.message || 'Không thể đổi mô hình. Kiểm tra kết nối backend.');
    } finally {
      await refreshBackendData();
      modelSwitchRef.current = false;
      setIsSwitchingModel(false);
    }
  };

  const modelSelectorProps = {
    models, selectedModel, onSelectModel: handleSelectModel,
    onRefreshModels: () => { setIsCheckingHealth(true); return refreshBackendData(); }, isSwitchingModel,
    isRefreshingModels: isCheckingHealth, modelError,
  };

  // Read actual dimensions of the illustrative images, never placeholder dimensions.
  useEffect(() => {
    if (imageMeta?.file || !imageMeta?.url || imageMeta.width) return;
    let active = true;
    const image = new Image();
    image.onload = () => {
      if (!active) return;
      setImageMeta((current) => current?.url === imageMeta.url
        ? { ...current, width: image.naturalWidth, height: image.naturalHeight,
            outWidth: image.naturalWidth * 4, outHeight: image.naturalHeight * 4 }
        : current);
    };
    image.src = imageMeta.url;
    return () => { active = false; };
  }, [imageMeta?.file, imageMeta?.url, imageMeta?.width]);

  useEffect(() => {
    if (!imageMeta?.file || !imageMeta.url?.startsWith('blob:')) return;
    const objectUrl = imageMeta.url;
    return () => URL.revokeObjectURL(objectUrl);
  }, [imageMeta?.file, imageMeta?.url]);

  useEffect(() => () => {
    requestVersionRef.current += 1;
    requestRef.current?.abort();
  }, []);

  const cancelActiveRequest = () => {
    requestVersionRef.current += 1;
    requestRef.current?.abort();
    requestRef.current = null;
    setIsUpscaling(false);
  };

  const handleImageSelected = (meta) => {
    cancelActiveRequest();
    setImageMeta(meta);
    setComparisonUrl(null);
    setResultMeta(null);
    setUpscaleError(null);
    setIsDemoSample(false);
  };

  const handleSelectSample = (sample) => {
    cancelActiveRequest();
    setImageMeta(createDemoMeta(sample));
    setComparisonUrl(sample.after);
    setResultMeta(null);
    setUpscaleError(null);
    setIsDemoSample(true);
  };

  const handleRunUpscale = async () => {
    if (requestRef.current || isUpscaling || modelSwitchRef.current) return;
    if (!imageMeta?.file) {
      setUpscaleError('Vui lòng tải lên ảnh nguồn trước khi thực hiện siêu phân giải.');
      return;
    }

    const model = models.find((item) => item.id === selectedModel && item.available);
    if (health?.status !== 'ok' || !model) {
      setUpscaleError('Backend hoặc model ảnh chưa sẵn sàng. Vui lòng kiểm tra lại kết nối.');
      return;
    }

    const controller = new AbortController();
    const requestVersion = ++requestVersionRef.current;
    requestRef.current = controller;
    setIsUpscaling(true);
    setUpscaleError(null);

    try {
      const data = await upscaleImage(imageMeta.file, model.id, controller.signal);
      if (requestVersionRef.current !== requestVersion || controller.signal.aborted) return;
      setComparisonUrl(data.result_url);
      setResultMeta(data);
      setIsDemoSample(false);
    } catch (err) {
      if (requestVersionRef.current === requestVersion && !controller.signal.aborted) {
        console.error('Upscale failed:', err);
        setUpscaleError(err.message || 'Lỗi không xác định khi suy luận AI.');
      }
    } finally {
      if (requestVersionRef.current === requestVersion) {
        requestRef.current = null;
        setIsUpscaling(false);
      }
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-[#121316] text-[#e5e7eb] overflow-hidden">
      {/* Top Header with Mode Switcher */}
      <Header
        health={health}
        isCheckingHealth={isCheckingHealth}
        activeMode={activeMode}
        onModeChange={setActiveMode}
      />

      {/* Main Studio Body */}
      {activeMode === 'image' ? (
        <main className="flex-1 flex flex-col lg:flex-row overflow-hidden">
          {/* Left Column: Upload, Settings & Specs */}
          <UploadPanel
            imageMeta={imageMeta}
            modelSelectorProps={modelSelectorProps}
            onImageSelected={handleImageSelected}
            onSelectSample={handleSelectSample}
            samples={DEMO_SAMPLES}
            isUpscaling={isUpscaling}
            upscaleError={upscaleError}
            resultMeta={resultMeta}
            onRunUpscale={handleRunUpscale}
            onDismissError={() => setUpscaleError(null)}
            health={health}
          />

          {/* Center / Right Viewport: Interactive Comparison */}
          <CompareSlider
            key={`${imageMeta?.url || ''}|${comparisonUrl || ''}`}
            originalUrl={imageMeta?.url}
            comparisonUrl={comparisonUrl}
            isDemoSample={isDemoSample}
            isUpscaling={isUpscaling}
            resultMeta={resultMeta}
          />
        </main>
      ) : (
        <VideoStudio health={health} modelSelectorProps={modelSelectorProps} />
      )}
    </div>
  );
}

