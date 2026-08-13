
from backend.tools.paddle_model_config import PaddleModelConfig
from backend.tools.hardware_accelerator import HardwareAccelerator
from backend.tools.ocr_backend import uses_rapidocr_torch
from backend.config import BASE_DIR, config
import os
import numpy as np

try:
    from paddleocr import TextDetection
except ImportError:
    TextDetection = None


class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    def __init__(self):
        hardware_accelerator = HardwareAccelerator.instance()
        self.rapidocr = uses_rapidocr_torch()
        if self.rapidocr:
            from backend.tools.rapidocr_torch import RapidOcrTorch, rapidocr_model_dir
            self.text_detector = RapidOcrTorch(
                language=config.language.value,
                use_gpu=hardware_accelerator.supports_torch_gpu(),
                accurate=config.mode.value != 'fast',
                model_root_dir=rapidocr_model_dir(
                    os.path.join(BASE_DIR, "models", "rapidocr")),
            )
            return

        model_config = PaddleModelConfig(hardware_accelerator)
        # 使用 TextDetection 公开 API（PaddleOCR 3.x）
        device = hardware_accelerator.paddle_device
        kwargs = {'device': device}
        if device == 'cpu':
            kwargs['enable_mkldnn'] = False
        if model_config.DET_MODEL_PATH:
            kwargs['model_dir'] = model_config.DET_MODEL_PATH
        if model_config.DET_MODEL_NAME:
            kwargs['model_name'] = model_config.DET_MODEL_NAME
        self.text_detector = TextDetection(**kwargs)

    def detect_subtitle(self, img):
        """
        检测图像中的文本框
        :param img: 输入图像
        :return: (dt_boxes, elapse) dt_boxes为numpy数组，elapse为耗时
        """
        if self.rapidocr:
            return self.text_detector.detect(img)

        results = list(self.text_detector.predict(img))
        if not results:
            return np.array([]), 0
        res = results[0]
        dt_polys = res.get('dt_polys', np.array([]))
        return dt_polys, 0
