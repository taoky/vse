# Vendored source snapshots

This directory contains source snapshots required to audit and reproduce the
local Linux build. Nested Git metadata and generated build files are excluded.

- `VideoSubFinder`: SWHL/VideoSubFinder commit
  `d6ca256a87d2e3ab71c4544bc7379fb5475fcf09`, with the local OpenCV 5,
  hardening, shell-command removal, and CPU-only build patches.
- `RapidOCR`: RapidAI/RapidOCR commit
  `3efd66a6ba32ff6b7ae1c7a36b8a3cb54c93e53c` (release 3.9.2), installed
  directly from this snapshot. The vendored loader and VSE's adapter both
  enforce `torch.load(..., weights_only=True)` before a model is opened.

Both upstream projects retain their original license files in their source
directories.
