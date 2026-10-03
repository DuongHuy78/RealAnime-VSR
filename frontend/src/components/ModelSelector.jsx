import React, { useId } from 'react';

export default function ModelSelector({
  models, selectedModel, onSelectModel, onRefreshModels,
  isSwitchingModel, isRefreshingModels, modelError, disabled = false,
}) {
  const selectId = useId();
  const selected = models.find((model) => model.id === selectedModel);
  return (
    <div className="bg-[#1d1f25] rounded-lg p-3.5 border border-[#262930] space-y-2 text-xs font-mono">
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={selectId} className="text-[11px] font-semibold text-[#7d8494] uppercase tracking-wider">
          Mô hình AI (.pth)
        </label>
        <button
          type="button"
          onClick={onRefreshModels}
          disabled={isRefreshingModels || isSwitchingModel}
          className="text-[11px] text-[#b9f73a] hover:underline disabled:opacity-40"
        >
          {isRefreshingModels ? 'Đang quét...' : 'Làm mới'}
        </button>
      </div>
      <select
        id={selectId}
        value={selectedModel}
        onChange={(event) => onSelectModel(event.target.value)}
        disabled={disabled || isSwitchingModel || models.length === 0}
        className="w-full bg-[#17191d] border border-[#343842] rounded px-3 py-2 text-white focus:outline-none focus:border-[#b9f73a] disabled:opacity-50"
      >
        <option value="" disabled>{models.length ? 'Chọn checkpoint' : 'Chưa tìm thấy file .pth'}</option>
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.checkpoint_path || model.name}{model.loaded ? ` · x${model.scale} · đang dùng` : model.error ? ' · lỗi' : ' · chưa nạp'}
          </option>
        ))}
      </select>
      {selected && !selected.loaded && !isSwitchingModel && (
        <button
          type="button"
          onClick={() => onSelectModel(selected.id)}
          disabled={disabled}
          className="text-[11px] text-[#b9f73a] hover:underline disabled:opacity-40"
        >
          Nạp mô hình đã chọn
        </button>
      )}
      {isSwitchingModel ? (
        <p role="status" className="text-[11px] text-[#b9f73a]">Đang nạp mô hình, chờ tác vụ hiện tại hoàn tất...</p>
      ) : selected?.loaded ? (
        <p className="text-[11px] text-[#b9f73a]">Đã nạp · {selected.device.toUpperCase()}</p>
      ) : (
        <p className="text-[11px] text-[#7d8494]">Chọn file để nạp mô hình RealBasicVSR x4.</p>
      )}
      <p className="text-[10px] text-[#7d8494] leading-relaxed">
        Đặt checkpoint trong thư mục gốc dự án hoặc models/, rồi nhấn Làm mới.
      </p>
      {modelError && <p role="alert" className="text-[11px] text-red-300 leading-relaxed">{modelError}</p>}
      {models.filter((model) => model.error).map((model) => (
        <p key={model.id} className="text-[10px] text-red-300 leading-relaxed">{model.error}</p>
      ))}
    </div>
  );
}
