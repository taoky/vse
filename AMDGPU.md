# AMDGPU / PyTorch ROCm

The AMD path uses RapidOCR's PyTorch engine. ROCm PyTorch intentionally uses
the `torch.cuda` API, so RapidOCR's `cuda:0` device selects the AMD GPU.
VideoSubFinder remains on the source-built CPU implementation and never
receives its NVIDIA-only `--use_cuda` option on AMD.

RapidOCR 3.9.2 normally deserializes `.pth` files with
`torch.load(..., weights_only=False)`. VSE replaces that loader before any
model is opened, enforces `weights_only=True` and strict state-dict loading,
and retains RapidOCR's SHA-256 download verification.

Run the readiness check through:

```sh
VSE_PYTHON=/path/to/rocm/environment/bin/python ./run-rocm.sh
```

The launcher refuses to start if PyTorch is not a ROCm build or cannot see a
GPU. It also reports access to `/dev/kfd` and DRM render nodes.

Use a separate Python 3.12 environment. Install an AMD-supported ROCm PyTorch
wheel first, then install `requirements_rapidocr_torch.txt` with `uv`.
`requirements.txt` retains Paddle for the separate CPU fallback environment;
the ROCm environment does not install or import Paddle.

`run-rocm.sh` stores the hash-verified weights under
`.cache/rapidocr-safe-test`. Override this with `VSE_RAPIDOCR_MODEL_DIR` when
using a read-only, pre-audited model directory.

Official references:

- https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-pytorch.html
- https://github.com/RapidAI/RapidOCR
