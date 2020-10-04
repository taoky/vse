"""Small, side-effect-free ROCm readiness check for the VSE launch script."""

import glob
import json
import os
import sys

import torch


def collect_status():
    render_nodes = glob.glob('/dev/dri/renderD*')
    return {
        'torch_version': torch.__version__,
        'torch_hip_version': getattr(torch.version, 'hip', None),
        'torch_gpu_available': bool(torch.cuda.is_available()),
        'torch_gpu_count': torch.cuda.device_count(),
        'torch_gpu_name': (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else None),
        'kfd_exists': os.path.exists('/dev/kfd'),
        'kfd_readable': os.access('/dev/kfd', os.R_OK),
        'kfd_writable': os.access('/dev/kfd', os.W_OK),
        'render_nodes': render_nodes,
        'writable_render_nodes': [node for node in render_nodes if os.access(node, os.R_OK | os.W_OK)],
    }


def main():
    status = collect_status()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if not status['torch_hip_version']:
        print('PyTorch is not a ROCm build.', file=sys.stderr)
        return 1
    if not status['torch_gpu_available']:
        print('The ROCm PyTorch build cannot see a GPU device.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
