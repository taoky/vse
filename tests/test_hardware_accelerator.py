import unittest
from unittest.mock import patch

from backend.tools.hardware_accelerator import AcceleratorBackend, HardwareAccelerator


class HardwareAcceleratorTest(unittest.TestCase):
    def test_cpu_build(self):
        accelerator = HardwareAccelerator()
        with patch('backend.tools.hardware_accelerator.paddle.device.get_available_device', return_value=['cpu']), \
                patch('backend.tools.hardware_accelerator.paddle.is_compiled_with_rocm', return_value=False), \
                patch('backend.tools.hardware_accelerator.paddle.is_compiled_with_cuda', return_value=False):
            accelerator.check_paddle()

        self.assertEqual(accelerator.paddle_backend, AcceleratorBackend.CPU)
        self.assertEqual(accelerator.paddle_device, 'cpu')
        self.assertFalse(accelerator.supports_paddle_gpu())

    def test_rocm_build(self):
        accelerator = HardwareAccelerator()
        with patch('backend.tools.hardware_accelerator.paddle.device.get_available_device', return_value=['cpu', 'gpu:0']), \
                patch('backend.tools.hardware_accelerator.paddle.is_compiled_with_rocm', return_value=True), \
                patch('backend.tools.hardware_accelerator.paddle.is_compiled_with_cuda', return_value=False):
            accelerator.check_paddle()

        self.assertEqual(accelerator.paddle_backend, AcceleratorBackend.ROCM)
        self.assertEqual(accelerator.paddle_device, 'gpu:0')
        self.assertTrue(accelerator.has_rocm())
        self.assertFalse(accelerator.has_cuda())

    def test_cuda_build(self):
        accelerator = HardwareAccelerator()
        with patch('backend.tools.hardware_accelerator.paddle.device.get_available_device', return_value=['cpu', 'gpu:0']), \
                patch('backend.tools.hardware_accelerator.paddle.is_compiled_with_rocm', return_value=False), \
                patch('backend.tools.hardware_accelerator.paddle.is_compiled_with_cuda', return_value=True):
            accelerator.check_paddle()

        self.assertEqual(accelerator.paddle_backend, AcceleratorBackend.CUDA)
        self.assertTrue(accelerator.has_cuda())
        self.assertFalse(accelerator.has_rocm())

    def test_onnx_provider_is_not_reported_as_active_ocr_acceleration(self):
        accelerator = HardwareAccelerator()
        accelerator._HardwareAccelerator__onnx_providers = ['ROCMExecutionProvider']

        self.assertFalse(accelerator.has_accelerator())
        self.assertEqual(accelerator.paddle_device, 'cpu')


if __name__ == '__main__':
    unittest.main()
