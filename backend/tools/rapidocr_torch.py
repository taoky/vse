"""Security-hardened RapidOCR PyTorch adapter.

RapidOCR 3.9.2 loads its ``.pth`` state dictionaries with
``weights_only=False``.  That enables Python pickle object construction and is
not acceptable for downloaded model files.  This adapter replaces only that
loader method and keeps RapidOCR's built-in SHA-256 verification in place.
"""

import os
from pathlib import Path

import numpy as np


LANGUAGE_MAP = {
    "ch": "ch",
    "chinese_cht": "chinese_cht",
    "en": "en",
    "ko": "korean",
    "japan": "japan",
    "vi": "latin",
    "es": "latin",
    "tr": "latin",
}


def rapidocr_model_profile(language, accurate=False):
    """Return (OCR version, model tier) without importing RapidOCR."""
    lang = LANGUAGE_MAP.get(language, "ch")
    if lang in {"ch", "chinese_cht", "en", "japan", "latin"}:
        return "PP-OCRv6", "medium" if accurate else "small"
    return ("PP-OCRv5" if lang == "ch" else "PP-OCRv4"), "mobile"


def rapidocr_model_dir(default_dir):
    return Path(os.environ.get("VSE_RAPIDOCR_MODEL_DIR", default_dir)).resolve()


def _install_safe_model_loader():
    import torch
    from rapidocr.inference_engine.pytorch.networks.architectures.base_model import BaseModel
    from rapidocr.inference_engine.pytorch.networks.main import ModelLoader

    def load_weights_only(self, arch_config, model_path):
        model = BaseModel(arch_config)
        state_dict = torch.load(
            Path(model_path), map_location="cpu", weights_only=True
        )
        if not isinstance(state_dict, dict):
            raise RuntimeError("RapidOCR model is not a weights-only state dictionary")
        model.load_state_dict(state_dict, strict=True)
        return model

    ModelLoader._build_and_load_model = load_weights_only


def torch_device_status():
    try:
        import torch
    except ImportError:
        return False, False, "PyTorch is not installed"

    available = torch.cuda.is_available()
    is_rocm = bool(getattr(torch.version, "hip", None))
    if not available:
        return False, is_rocm, "PyTorch cannot access a GPU"
    return True, is_rocm, torch.cuda.get_device_name(0)


class RapidOcrTorch:
    def __init__(self, language="ch", use_gpu=True, model_root_dir=None,
                 accurate=False):
        _install_safe_model_loader()
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import (
            EngineType, LangDet, LangRec, ModelType, OCRVersion,
        )

        lang = LANGUAGE_MAP.get(language, "ch")
        version_name, model_type_name = rapidocr_model_profile(language, accurate)
        # PP-OCRv6 unifies Chinese, English, Japanese and Latin scripts. Keep
        # the existing specialized model family for unsupported languages.
        version = OCRVersion(version_name)
        model_type = ModelType(model_type_name)
        det_lang = LangDet.EN if lang == "en" else LangDet.CH
        params = {
            "Global.use_cls": False,
            "Global.text_score": 0.0,
            "Det.engine_type": EngineType.TORCH,
            "Det.ocr_version": version,
            "Det.model_type": model_type,
            "Det.lang_type": det_lang,
            "Cls.engine_type": EngineType.TORCH,
            "Rec.engine_type": EngineType.TORCH,
            "Rec.ocr_version": version,
            "Rec.model_type": model_type,
            "Rec.lang_type": LangRec(lang),
            "EngineConfig.torch.use_cuda": bool(use_gpu),
        }
        if model_root_dir:
            params["Global.model_root_dir"] = Path(model_root_dir)
        self.engine = RapidOCR(params=params)

    def predict(self, image):
        result = self.engine(image, use_det=True, use_cls=False, use_rec=True)
        boxes = result.boxes if result.boxes is not None else np.empty((0, 4, 2))
        texts = result.txts or ()
        scores = result.scores or ()
        return boxes, texts, scores

    def detect(self, image):
        result = self.engine(image, use_det=True, use_cls=False, use_rec=False)
        boxes = result.boxes if result.boxes is not None else np.empty((0, 4, 2))
        return np.asarray(boxes), float(getattr(result, "elapse", 0.0) or 0.0)
