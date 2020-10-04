from backend.config import tr
try:
    import paddle
except ImportError:
    paddle = None

from backend.tools.ocr_backend import uses_rapidocr_torch


class AcceleratorBackend:
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"

class HardwareAccelerator:

    # 类变量，用于存储单例实例
    _instance = None

    @classmethod
    def instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = HardwareAccelerator()
            cls._instance.initialize()
        return cls._instance

    def __init__(self):
        self.__cuda = False
        self.__rocm = False
        self.__onnx_providers = []
        self.__torch_gpu = False
        self.__torch_rocm = False
        self.__enabled = True

    def initialize(self):
        self.check_paddle()
        self.check_torch()
        self.check_onnx()

    def check_paddle(self):
        """Detect only accelerators that the installed Paddle build can use."""
        self.__cuda = False
        self.__rocm = False
        if paddle is None:
            return
        try:
            available_devices = paddle.device.get_available_device()
        except Exception:
            available_devices = []
        has_gpu_device = any(str(device).startswith('gpu') for device in available_devices)

        is_rocm_build = getattr(paddle, 'is_compiled_with_rocm', lambda: False)()
        if is_rocm_build and has_gpu_device:
            self.__rocm = True
        elif paddle.is_compiled_with_cuda() and has_gpu_device:
            self.__cuda = True

    def check_torch(self):
        self.__torch_gpu = False
        self.__torch_rocm = False
        try:
            import torch
            self.__torch_gpu = bool(torch.cuda.is_available())
            self.__torch_rocm = self.__torch_gpu and bool(
                getattr(torch.version, 'hip', None))
        except (ImportError, RuntimeError):
            pass

    def check_onnx(self):
        self.__onnx_providers = []
        if self.supports_paddle_gpu():
            return
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            for provider in available_providers:
                if provider in [
                    "CPUExecutionProvider"
                ]:
                    continue
                if provider not in [
                    "DmlExecutionProvider",         # DirectML，适用于 Windows GPU
                    "ROCMExecutionProvider",        # AMD ROCm
                    "MIGraphXExecutionProvider",    # AMD MIGraphX
                    "VitisAIExecutionProvider",     # AMD VitisAI，适用于 RyzenAI & Windows, 实测和DirectML性能似乎差不多
                    "OpenVINOExecutionProvider",    # Intel GPU
                    "MetalExecutionProvider",       # Apple macOS
                    "CoreMLExecutionProvider",      # Apple macOS
                    "CUDAExecutionProvider",        # Nvidia GPU
                ]:
                    print(tr['Main']['OnnxExectionProviderNotSupportedSkipped'].format(provider))
                    continue
                print(tr['Main']['OnnxExecutionProviderDetected'].format(provider))
                self.__onnx_providers.append(provider)
        except ModuleNotFoundError as e:
            print(tr['Main']['OnnxRuntimeNotInstall'])

    def has_accelerator(self):
        """Whether the current OCR implementation has an active accelerator."""
        if not self.__enabled:
            return False
        # ONNX providers are recorded for a future ONNX OCR implementation,
        # but the current PaddleOCR pipeline cannot execute through them.
        if uses_rapidocr_torch():
            return self.supports_torch_gpu()
        return self.supports_paddle_gpu()

    def supports_torch_gpu(self):
        return self.__enabled and self.__torch_gpu

    def supports_paddle_gpu(self):
        """Whether PaddleOCR can execute on a native GPU backend."""
        if not self.__enabled:
            return False
        return self.__cuda or self.__rocm

    @property
    def paddle_device(self):
        return 'gpu:0' if self.supports_paddle_gpu() else 'cpu'

    @property
    def paddle_backend(self):
        if not self.__enabled:
            return AcceleratorBackend.CPU
        if self.__rocm:
            return AcceleratorBackend.ROCM
        if self.__cuda:
            return AcceleratorBackend.CUDA
        return AcceleratorBackend.CPU

    @property
    def accelerator_name(self):
        if not self.__enabled:
            return "CPU"
        if uses_rapidocr_torch() and self.__torch_gpu:
            return "PyTorch ROCm" if self.__torch_rocm else "PyTorch CUDA"
        if self.__cuda:
            return "CUDA"
        elif self.__rocm:
            return "ROCm"
        elif len(self.__onnx_providers) > 0:
            return "ONNX: " + ", ".join(self.__onnx_providers)
        else:
            return "CPU"

    @property
    def onnx_providers(self):
        if not self.__enabled:
            return []
        return self.__onnx_providers

    def has_cuda(self):
        if not self.__enabled:
            return False
        return self.__cuda

    def has_rocm(self):
        if not self.__enabled:
            return False
        return self.__rocm or (uses_rapidocr_torch() and self.__torch_rocm)

    def set_enabled(self, enable):
        self.__enabled = enable
