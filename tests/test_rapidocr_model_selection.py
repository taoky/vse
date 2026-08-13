import unittest

from backend.tools.rapidocr_torch import rapidocr_model_profile


class RapidOcrModelSelectionTest(unittest.TestCase):
    def test_fast_mode_uses_v6_small(self):
        for language in ('ch', 'chinese_cht', 'en', 'japan', 'vi', 'es', 'tr'):
            with self.subTest(language=language):
                self.assertEqual(
                    rapidocr_model_profile(language),
                    ('PP-OCRv6', 'small'))

    def test_accurate_mode_uses_v6_medium(self):
        self.assertEqual(
            rapidocr_model_profile('ch', accurate=True),
            ('PP-OCRv6', 'medium'))

    def test_korean_keeps_specialized_fallback(self):
        self.assertEqual(
            rapidocr_model_profile('ko'),
            ('PP-OCRv4', 'mobile'))


if __name__ == '__main__':
    unittest.main()
