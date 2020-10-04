import os


PADDLE_BACKEND = "paddle"
RAPIDOCR_TORCH_BACKEND = "rapidocr-torch"
SUPPORTED_OCR_BACKENDS = (PADDLE_BACKEND, RAPIDOCR_TORCH_BACKEND)


def get_ocr_backend():
    backend = os.environ.get("VSE_OCR_BACKEND", PADDLE_BACKEND).strip().lower()
    if backend not in SUPPORTED_OCR_BACKENDS:
        choices = ", ".join(SUPPORTED_OCR_BACKENDS)
        raise RuntimeError(f"Unsupported VSE_OCR_BACKEND={backend!r}; choose: {choices}")
    return backend


def uses_rapidocr_torch():
    return get_ocr_backend() == RAPIDOCR_TORCH_BACKEND
