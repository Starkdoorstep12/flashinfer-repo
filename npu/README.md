# NPU Benchmarking (Planned — Not Yet Started In This Repo)

This directory is a placeholder for a fifth benchmarking axis alongside
the four GPUs covered elsewhere in this repo (RTX 6000 Ada, A100, L40S,
DGX Spark/Blackwell): a Qualcomm Hexagon NPU.

## Motivation

Exploring an NPU target alongside the GPU work as a less-saturated,
higher-novelty angle than a GPU-only kernel optimization comparison for
the MLSys 2026 submission.

## Hardware and toolchain

- **Device**: Lenovo ThinkCentre Neo 50q QC (Snapdragon X1-26-100),
  Hexagon NPU, rated up to 45 TOPS. Provided by the advisor as separate
  hardware for this axis.
- **OS**: Windows 11 Pro.
- **Toolchain**: Qualcomm QAIRT/QNN SDK, Python 3.12 ARM64,
  onnxruntime-qnn. Confirmed working: the toolchain detects and can
  target the Hexagon NPU device.
- All NPU-side development happens on the ThinkCentre itself — the
  QAIRT/QNN SDK and NPU execution do not run on the primary dev laptop
  (Intel i7-10510U, no NPU). Code developed there will be synced into
  this directory once it exists.

## Current blocker

`QcSoCServiceUtils.dll` (a Qualcomm/Lenovo OEM driver DLL) crashes with a
stack-overflow-class exception whenever a tool queries SoC/device info on
the HTP/NPU path. Reproduced in both `qnn-platform-validator.exe` and an
onnxruntime-qnn `InferenceSession` targeting the NPU device. A Lenovo
firmware update did not resolve it. Not yet fixed.

## Status

No code or environment setup has been committed to this repo yet — all
work so far has happened directly on the ThinkCentre. This file exists so
the planned scope is visible in the repo ahead of that work landing here.
