import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

from backend.tools.rapidocr_torch import _install_safe_model_loader


class RapidOcrTorchSecurityTest(unittest.TestCase):
    def test_loader_forces_weights_only_and_strict_state_dict(self):
        torch = types.ModuleType("torch")
        torch.load = Mock(return_value={"weight": object()})
        base_model = Mock()
        model = Mock()
        base_model.return_value = model

        main_module = types.ModuleType(
            "rapidocr.inference_engine.pytorch.networks.main")
        main_module.ModelLoader = type("ModelLoader", (), {})
        arch_module = types.ModuleType(
            "rapidocr.inference_engine.pytorch.networks.architectures.base_model")
        arch_module.BaseModel = base_model

        modules = {
            "torch": torch,
            "rapidocr": types.ModuleType("rapidocr"),
            "rapidocr.inference_engine": types.ModuleType("inference_engine"),
            "rapidocr.inference_engine.pytorch": types.ModuleType("pytorch"),
            "rapidocr.inference_engine.pytorch.networks": types.ModuleType("networks"),
            "rapidocr.inference_engine.pytorch.networks.main": main_module,
            "rapidocr.inference_engine.pytorch.networks.architectures": types.ModuleType("architectures"),
            "rapidocr.inference_engine.pytorch.networks.architectures.base_model": arch_module,
        }
        old_modules = {name: sys.modules.get(name) for name in modules}
        try:
            sys.modules.update(modules)
            _install_safe_model_loader()
            loader = main_module.ModelLoader()
            loaded = loader._build_and_load_model({}, Path("model.pth"))
        finally:
            for name, old in old_modules.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old

        self.assertIs(loaded, model)
        torch.load.assert_called_once_with(
            Path("model.pth"), map_location="cpu", weights_only=True)
        model.load_state_dict.assert_called_once_with(
            {"weight": unittest.mock.ANY}, strict=True)


if __name__ == "__main__":
    unittest.main()
