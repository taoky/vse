import unittest
import os
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault(
    'PADDLE_PDX_CACHE_HOME',
    os.path.abspath('.cache/paddlex-test'))
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

from backend.tools.ocr import OcrRecogniser


MODEL_CONFIG = SimpleNamespace(
    DET_MODEL_PATH=None,
    REC_MODEL_PATH=None,
    DET_MODEL_NAME='PP-OCRv6_small_det',
    REC_MODEL_NAME='PP-OCRv6_small_rec',
)


class FakeAccelerator:
    def __init__(self, device):
        self.paddle_device = device


class OcrDeviceSelectionTest(unittest.TestCase):
    def init_and_get_kwargs(self, device):
        recogniser = OcrRecogniser()
        recogniser.hardware_accelerator = FakeAccelerator(device)
        with patch('backend.tools.ocr.PaddleModelConfig', return_value=MODEL_CONFIG), \
                patch('backend.tools.ocr._create_paddle_ocr', return_value=object()) as paddle_ocr:
            recogniser.init_model()
        return paddle_ocr.call_args.kwargs

    def test_cpu_disables_mkldnn(self):
        kwargs = self.init_and_get_kwargs('cpu')
        self.assertEqual(kwargs['device'], 'cpu')
        self.assertIs(kwargs['enable_mkldnn'], False)

    def test_rocm_uses_paddle_gpu_device(self):
        kwargs = self.init_and_get_kwargs('gpu:0')
        self.assertEqual(kwargs['device'], 'gpu:0')
        self.assertNotIn('enable_mkldnn', kwargs)


if __name__ == '__main__':
    unittest.main()
