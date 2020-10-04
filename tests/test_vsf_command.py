import unittest

from backend.tools.vsf_command import build_vsf_command


class VideoSubFinderCommandTest(unittest.TestCase):
    def build(self, use_cuda):
        return build_vsf_command(
            '/opt/vsf', '/video with spaces.mp4', '/tmp/output', '/tmp/raw.srt',
            0.25, 0.0, 0.1, 0.9, 8, 'OPENCV', use_cuda=use_cuda)

    def test_non_cuda_backend_never_gets_cuda_switch(self):
        self.assertNotIn('--use_cuda', self.build(False))

    def test_cuda_backend_gets_cuda_switch(self):
        self.assertIn('--use_cuda', self.build(True))

    def test_video_path_remains_one_argument(self):
        command = self.build(False)
        self.assertEqual(command[command.index('-i') + 1], '/video with spaces.mp4')


if __name__ == '__main__':
    unittest.main()
