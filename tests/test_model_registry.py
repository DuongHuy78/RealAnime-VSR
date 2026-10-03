import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import torch

from backend.app import model_registry as registry
from backend.app.config import WORK_DIR


class FakeNet:
    def __init__(self):
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self


class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=WORK_DIR)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models_dir = self.root / "models"
        self.models_dir.mkdir()
        self.patcher = patch.multiple(
            registry, REPO_ROOT=self.root, MODELS_DIR=self.models_dir,
            CHECKPOINT_PATH=self.root / "best.pth", AI_DEVICE="cpu",
            _MODEL_CACHE={}, _MODEL_DEVICES={}, _MODEL_SIGNATURES={},
            _MODEL_ERRORS={}, _CHECKPOINTS={}, _DEFAULT_CHECKPOINT=None,
            _ACTIVE_MODEL_ID=None, _ACTIVE_DEVICE=None, _INFERENCE_SEMAPHORE=None,
        )
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.loader = patch.object(registry, "load_realbasicvsr", side_effect=lambda *a, **kw: (FakeNet(), torch.device("cpu"))).start()
        self.addCleanup(patch.stopall)

    def checkpoint(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
        return path

    def test_discovery_includes_root_nested_models_and_duplicate_names(self):
        self.checkpoint("best.pth")
        self.checkpoint("models/best.pth")
        self.checkpoint("models/nested/anime.PTH")
        self.checkpoint("ignored.pt")
        self.checkpoint("other/ignored.pth")
        models = registry.list_available_models()
        self.assertEqual(len(models), 3)
        self.assertEqual(len({model.id for model in models}), 3)
        self.assertEqual(registry.discover_checkpoints()[registry.DEFAULT_MODEL_ID], self.root / "best.pth")
        self.assertTrue(all(not model.loaded and model.device == "none" for model in models))
        self.loader.assert_not_called()

    def test_refresh_finds_new_files_without_changing_existing_ids(self):
        self.checkpoint("v3.pth")
        before = registry.discover_checkpoints()
        self.checkpoint("a_new.pth")
        after = registry.discover_checkpoints()
        self.assertEqual(len(after), 2)
        for model_id, path in before.items():
            self.assertEqual(after[model_id], path)

    def test_missing_default_uses_discovered_checkpoint_and_switch_reuses_cache(self):
        self.checkpoint("v3.pth")
        self.checkpoint("v4.pth")
        self.assertEqual(registry.load_model_registry(), (1, "cpu"))
        ids = list(registry.discover_checkpoints())
        original = registry.get_model(ids[0])[0]
        registry.activate_model(ids[1])
        registry.activate_model(ids[0])
        self.assertIs(registry.get_model()[0], original)
        self.assertEqual(self.loader.call_count, 2)

    def test_replaced_weights_are_reloaded_and_deleted_file_is_rejected(self):
        path = self.checkpoint("v3.pth")
        registry.load_model_registry()
        original = registry.get_model()[0]
        path.write_bytes(b"new checkpoint with a different size")
        registry.activate_model(registry.DEFAULT_MODEL_ID)
        self.assertIsNot(registry.get_model()[0], original)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            registry.activate_model(registry.DEFAULT_MODEL_ID)
        self.assertEqual(registry.list_available_models(), [])

    def test_invalid_checkpoint_preserves_active_model_and_reports_error(self):
        self.checkpoint("v3.pth")
        self.checkpoint("v4.pth")
        registry.load_model_registry()
        active = registry.get_model()
        target = list(registry.discover_checkpoints())[1]
        self.loader.side_effect = RuntimeError("state_dict mismatch")
        with self.assertRaises(registry.ModelLoadError):
            registry.activate_model(target)
        self.assertEqual(registry.get_model(), active)
        failed = next(model for model in registry.list_available_models() if model.id == target)
        self.assertFalse(failed.available)
        self.assertIn("không tương thích", failed.error)

    def test_unknown_ids_and_paths_cannot_load_files(self):
        self.checkpoint("v3.pth")
        for value in ("../v3.pth", str(self.root / "v3.pth"), "unknown"):
            with self.assertRaises(FileNotFoundError):
                registry.activate_model(value)
        self.loader.assert_not_called()

    def test_switch_parks_previous_gpu_model_on_cpu(self):
        self.checkpoint("v3.pth")
        self.checkpoint("v4.pth")
        with patch.object(registry, "resolve_preferred_device", return_value=(torch.device("cuda:0"), "cuda")), patch.object(registry, "_warmup") as warmup, patch.object(torch.cuda, "empty_cache"):
            registry.load_model_registry()
            original = registry.get_model()[0]
            target = list(registry.discover_checkpoints())[1]
            def verify_device_during_switch(*args):
                previous = next(model for model in registry.list_available_models() if model.loaded)
                self.assertEqual(previous.device, "cpu")
            warmup.side_effect = verify_device_during_switch
            registry.activate_model(target)
            self.assertEqual(original.device.type, "cpu")
            self.assertEqual(registry.get_model()[0].device.type, "cuda")
            self.assertEqual(sum(device.type == "cuda" for device in registry._MODEL_DEVICES.values()), 1)

    def test_switch_waits_until_inference_releases_shared_lock(self):
        self.checkpoint("v3.pth")
        self.checkpoint("v4.pth")
        registry.load_model_registry()
        target = list(registry.discover_checkpoints())[1]
        started = threading.Event()
        def switch():
            started.set()
            return registry.activate_model(target)
        with ThreadPoolExecutor(max_workers=1) as executor:
            with registry.get_inference_lock():
                future = executor.submit(switch)
                self.assertTrue(started.wait(timeout=2))
                self.assertFalse(future.done())
                self.assertEqual(registry.get_active_model_ids(), [registry.DEFAULT_MODEL_ID])
            future.result(timeout=2)
        self.assertEqual(registry.get_active_model_ids(), [target])

    def test_cuda_failure_in_auto_mode_activates_new_model_on_cpu(self):
        self.checkpoint("v3.pth")
        self.checkpoint("v4.pth")
        with patch.object(registry, "AI_DEVICE", "auto"), patch.object(registry, "resolve_preferred_device", return_value=(torch.device("cuda:0"), "cuda")), patch.object(registry, "_warmup") as warmup, patch.object(torch.cuda, "empty_cache"):
            registry.load_model_registry()
            target = list(registry.discover_checkpoints())[1]
            warmup.side_effect = RuntimeError("CUDA warmup failed")
            net, device = registry.activate_model(target)
            self.assertEqual(device.type, "cpu")
            self.assertEqual(net.device.type, "cpu")
            self.assertEqual(registry.get_active_model_ids(), [target])
            self.assertTrue(all(device.type == "cpu" for device in registry._MODEL_DEVICES.values()))

    def test_explicit_cuda_failure_restores_previous_model_and_device(self):
        self.checkpoint("v3.pth")
        self.checkpoint("v4.pth")
        with patch.object(registry, "AI_DEVICE", "cuda"), patch.object(registry, "resolve_preferred_device", return_value=(torch.device("cuda:0"), "cuda")), patch.object(registry, "_warmup") as warmup, patch.object(torch.cuda, "empty_cache"):
            registry.load_model_registry()
            original = registry.get_model()
            target = list(registry.discover_checkpoints())[1]
            warmup.side_effect = RuntimeError("CUDA warmup failed")
            with self.assertRaises(registry.ModelLoadError):
                registry.activate_model(target)
            self.assertEqual(registry.get_model(), original)
            self.assertEqual(original[0].device.type, "cuda")
            self.assertEqual(registry.get_active_model_ids(), [registry.DEFAULT_MODEL_ID])

    def test_empty_directory_starts_degraded_without_loading(self):
        self.assertEqual(registry.load_model_registry(), (0, "none"))
        self.assertEqual(registry.list_available_models(), [])
        self.loader.assert_not_called()
