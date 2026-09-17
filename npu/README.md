# NPU Benchmarking Track

Fifth benchmarking axis alongside the four GPUs covered elsewhere in this
repo (RTX 6000 Ada, A100, L40S, DGX Spark/Blackwell): a Qualcomm Hexagon
NPU, via the Lenovo ThinkCentre Neo 50q QC (Snapdragon X1-26-100).

## Contents

- `onnx_export_indexer.py` — static-shape, masked rewrite of the top-k
  indexer's golden reference (`golden_indexer_reference.run()`) for ONNX
  export compatibility (ONNX requires fixed shapes; the original has
  per-batch dynamic top-k/seq_len). Validated exact match against the
  golden reference on both test batches (50/50, 80/80).
- `indexer.onnx` — the exported model.
- `run_npu_indexer_export.py`, `run_on_gpu_test.py`, `run_on_npu.py` —
  execution scripts for running the exported model via onnxruntime-qnn on
  the NPU/GPU execution providers.
- `NPU_BLOCKER.md` — current blocker: `QcSoCServiceUtils.dll` crashes on
  QNN device init for both NPU and GPU targets. The export pipeline
  itself is verified working; the crash is isolated to QNN
  device-targeted execution on this specific hardware/driver. See that
  file for the full stack trace and diagnosis.

## Status

Export and validation complete. Blocked on the QNN device-init crash
before any actual on-device NPU execution/benchmarking can happen — see
`NPU_BLOCKER.md`.
