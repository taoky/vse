import os
from backend.config import BASE_DIR, config


class PaddleModelConfig:
    def __init__(self, hardware_accelerator):
        self.hardware_accelerator = hardware_accelerator
        # 设置识别语言
        self.REC_CHAR_TYPE = config.language.value

        # 模型文件目录
        self.MODEL_BASE = os.path.join(BASE_DIR, 'models')
        # PP-OCRv6 is used for its supported languages; Korean keeps the
        # dedicated PP-OCRv5 recognizer.
        self.MODEL_VERSION = 'V6'
        # 默认图形识别的shape为3, 48, 320
        self.REC_IMAGE_SHAPE = '3,48,320'
        # 初始化模型路径
        self.REC_MODEL_PATH = None
        self.DET_MODEL_PATH = None
        self.DET_MODEL_NAME = None
        self.REC_MODEL_NAME = None

        # 语言组定义
        self.LATIN_LANG = [
            'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
            'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
            'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
            'sw', 'tl', 'tr', 'uz', 'vi', 'latin', 'german', 'french',
            'fi', 'eu', 'gl', 'lb', 'rm', 'ca', 'qu',
        ]
        self.ARABIC_LANG = ['ar', 'fa', 'ug', 'ur', 'ps', 'sd', 'bal']
        self.CYRILLIC_LANG = [
            'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
            'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'cyrillic',
            'sr', 'kk', 'ky', 'tg', 'mk', 'tt', 'cv', 'ba', 'mhr', 'mo',
            'udm', 'kv', 'os', 'bua', 'xal', 'tyv', 'sah', 'kaa',
        ]
        self.DEVANAGARI_LANG = [
            'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
            'sa', 'bgc', 'devanagari',
        ]
        self.OTHER_LANG = [
            'ch', 'japan', 'korean', 'en', 'ta', 'kn', 'te', 'ka',
            'chinese_cht',
        ]
        self.MULTI_LANG = (self.LATIN_LANG + self.ARABIC_LANG + self.CYRILLIC_LANG
                           + self.DEVANAGARI_LANG + self.OTHER_LANG)

        # 如果设置了识别文本语言类型，则设置为对应的语言
        if self.REC_CHAR_TYPE in self.MULTI_LANG:
            resolved = self._resolve_models()
            if resolved:
                self.MODEL_VERSION = 'V6' if self._supports_v6() else 'V5'
                self.DET_MODEL_PATH, self.REC_MODEL_PATH, self.DET_MODEL_NAME, self.REC_MODEL_NAME = resolved

    def _get_v5_rec_model_name(self, lang):
        """
        根据语言获取V5识别模型目录名
        参考: https://www.paddleocr.ai/main/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html
        """
        if lang in ('ch', 'chinese_cht', 'japan'):
            return 'PP-OCRv5_server_rec_infer'
        elif lang == 'en':
            return 'PP-OCRv5_server_rec_infer'
        elif lang == 'korean':
            return 'korean_PP-OCRv5_mobile_rec_infer'
        elif lang in self.LATIN_LANG:
            return 'latin_PP-OCRv5_mobile_rec_infer'
        elif lang in self.ARABIC_LANG:
            return 'arabic_PP-OCRv5_mobile_rec_infer'
        elif lang in self.CYRILLIC_LANG:
            return 'cyrillic_PP-OCRv5_mobile_rec_infer'
        elif lang in self.DEVANAGARI_LANG:
            return 'devanagari_PP-OCRv5_mobile_rec_infer'
        elif lang == 'th':
            return 'th_PP-OCRv5_mobile_rec_infer'
        elif lang == 'el':
            return 'el_PP-OCRv5_mobile_rec_infer'
        elif lang == 'ta':
            return 'ta_PP-OCRv5_mobile_rec_infer'
        elif lang == 'te':
            return 'te_PP-OCRv5_mobile_rec_infer'
        return None

    def _supports_v6(self):
        return self.REC_CHAR_TYPE in (
            ['ch', 'chinese_cht', 'en', 'japan'] + self.LATIN_LANG
        )

    def _v6_model_type(self):
        return 'small' if config.mode.value == 'fast' else 'medium'

    @staticmethod
    def _read_model_name_from_yaml(model_dir):
        """从 inference.yml 中读取 Global.model_name"""
        yaml_path = os.path.join(model_dir, 'inference.yml')
        if not os.path.exists(yaml_path):
            return None
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                in_global = False
                for line in f:
                    stripped = line.strip()
                    if stripped == 'Global:':
                        in_global = True
                        continue
                    if in_global:
                        if stripped and not stripped.startswith('#') and ':' in stripped:
                            if stripped.startswith('model_name:'):
                                return stripped.split(':', 1)[1].strip().strip('"').strip("'")
                        # 遇到下一个顶级 section 则退出
                        if stripped and not stripped.startswith('model_name') and not stripped.startswith(' ') and stripped.endswith(':'):
                            break
        except Exception:
            pass
        return None

    def _resolve_models(self):
        """
        解析模型路径，返回 (det_model_path, rec_model_path, det_model_name, rec_model_name) 或 None
        """
        if self._supports_v6():
            model_type = self._v6_model_type()
            v6_base = os.path.join(self.MODEL_BASE, 'V6')
            det_model_name = f'PP-OCRv6_{model_type}_det'
            rec_model_name = f'PP-OCRv6_{model_type}_rec'
            det_model_path = os.path.join(v6_base, f'{det_model_name}_infer')
            rec_model_path = os.path.join(v6_base, f'{rec_model_name}_infer')

            # Source-only installations let PaddleOCR download official named
            # models. A locally bundled pair is used only when both exist.
            if os.path.exists(det_model_path) and os.path.exists(rec_model_path):
                return (det_model_path, rec_model_path,
                        self._read_model_name_from_yaml(det_model_path),
                        self._read_model_name_from_yaml(rec_model_path))
            return None, None, det_model_name, rec_model_name

        v5_base = os.path.join(self.MODEL_BASE, 'V5')

        # 快速模式优先使用 mobile 模型，否则使用 server 模型
        if config.mode.value == 'fast':
            det_model_path = os.path.join(v5_base, 'PP-OCRv5_mobile_det_infer')
            if not os.path.exists(det_model_path):
                det_model_path = os.path.join(v5_base, 'PP-OCRv5_server_det_infer')
        else:
            det_model_path = os.path.join(v5_base, 'PP-OCRv5_server_det_infer')
        if not os.path.exists(det_model_path):
            # Source-only installations intentionally do not carry the large
            # bundled model files. Let PaddleOCR obtain its official named
            # models in the configured PaddleX cache instead.
            det_model_name = ('PP-OCRv5_mobile_det'
                              if config.mode.value == 'fast'
                              else 'PP-OCRv5_server_det')
            rec_model_name = ('PP-OCRv5_mobile_rec'
                              if config.mode.value == 'fast'
                              else self._get_v5_rec_model_name(self.REC_CHAR_TYPE))
            if rec_model_name and rec_model_name.endswith('_infer'):
                rec_model_name = rec_model_name[:-6]
            return None, None, det_model_name, rec_model_name

        det_model_name = self._read_model_name_from_yaml(det_model_path)

        # 快速模式：中文(简/繁)、英文、日文使用通用 mobile 模型，其他语言使用对应的专用模型
        if config.mode.value == 'fast' and self.REC_CHAR_TYPE in ('ch', 'chinese_cht', 'en', 'japan'):
            rec_model_path = os.path.join(v5_base, 'PP-OCRv5_mobile_rec_infer')
            if os.path.exists(rec_model_path):
                rec_model_name = self._read_model_name_from_yaml(rec_model_path)
                return det_model_path, rec_model_path, det_model_name, rec_model_name
            # mobile 不存在则 fallback 到按语言选择

        # 获取识别模型
        rec_model_dir_name = self._get_v5_rec_model_name(self.REC_CHAR_TYPE)
        if rec_model_dir_name is None:
            return None

        rec_model_path = os.path.join(v5_base, f'{rec_model_dir_name}_infer'
                                      if not rec_model_dir_name.endswith('_infer')
                                      else rec_model_dir_name)

        if not os.path.exists(rec_model_path):
            rec_model_path = os.path.join(v5_base, rec_model_dir_name)

        if not os.path.exists(rec_model_path):
            return None

        rec_model_name = self._read_model_name_from_yaml(rec_model_path)
        return det_model_path, rec_model_path, det_model_name, rec_model_name
