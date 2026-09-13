# NPU Blocker: QcSoCServiceUtils.dll crash on QNN device init

## Status: UNRESOLVED - blocks all QNN-targeted execution (NPU and GPU) on this device

## Environment
- Device: Lenovo ThinkCentre Neo 50q Gen 4 QC (Snapdragon X1-26-100, Hexagon NPU, ARM64)
- OS: Windows 11 Pro
- QAIRT SDK: 2.50.40.260831 (also have 2.31.0.250130 installed, untested)
- onnxruntime: 1.27.0, onnxruntime-qnn: 2.6.0
- Python: 3.12.10 (native ARM64)

## Symptom
Any tool that queries SoC/device capability info through the Qualcomm platform layer
crashes with exception code 0xc00000fd (stack-overflow class) inside
C:\Windows\SYSTEM32\QcSoCServiceUtils.dll (v1.0.4160.6000).

## Reproductions
1. qnn-platform-validator.exe --backend dsp --coreVersion - crashes.
2. qnn-platform-validator.exe --backend dsp --testBackend - crashes.
3. Python onnxruntime + onnxruntime_qnn, InferenceSession(...) construction with
   session_options.add_provider_for_devices([npu_device], {}) - crashes.
4. Same as (3) but targeting the GPU device exposed by the same QNN plugin instead of
   NPU - crashes at the SAME fault offset (0x5b0c), indicating the crash is in a
   shared device-query path hit before any backend-specific (HTP vs GPU) code runs, not
   something NPU-specific.

All four reproductions show identical Faulting module: QcSoCServiceUtils.dll,
Exception code: 0xc00000fd.

## What did NOT fix it
- Installing the one available Windows Update optional driver: "Lenovo Ltd. firmware
  1.0.0.65" + reboot - crash reproduces identically afterward.

## What we've ruled out
- Not an architecture/emulation issue (session confirmed running natively as ARM64,
  PROCESSOR_ARCHITEW6432 empty).
- Not specific to the HTP/NPU backend - GPU device hits the same fault offset.
- Not a corrupted download (file size and version info on the crashing exe/SDK checked
  out fine).

## Working hypothesis
QcSoCServiceUtils.dll is an OEM (Lenovo/Qualcomm) system service DLL shipped with the
chipset/platform driver package, not part of the QAIRT SDK itself. The crash likely sits
in whatever device-capability query QNN's device-selection/init path calls into on this
DLL, before any backend (HTP or GPU) actually executes. Given this is early first-gen
Snapdragon X silicon in a fairly new SKU, this looks like an OEM driver bug rather than
anything fixable from the SDK or application side.

## Confirmed NOT blocked
The actual ONNX export pipeline (npu/onnx_export_indexer.py,
npu/run_npu_indexer_export.py) works correctly end-to-end on CPU: the static-shape,
masked rewrite of the top-k indexer was validated against golden_indexer_reference.py's
run() and produces an exact match (50/50 and 80/80 tokens matching) on the same test
shapes used in correctness_test_indexer.py. The exported indexer.onnx is valid and
loads correctly under the plain CPU execution provider - the crash is specific to QNN
device-targeted execution, not the model or export pipeline.

## Next steps to try
1. Check for a newer Lenovo/Qualcomm driver release beyond the one firmware update
   already tried (support.lenovo.com, model: ThinkCentre Neo 50q Gen 4 QC).
2. Try the older QAIRT SDK already installed alongside this one (2.31.0.250130) - a
   different SDK build may avoid whatever code path triggers this in the newer SDK.
3. Report to Qualcomm (via the developer portal / QAIRT SDK issue channel) and/or Lenovo
   support with the exact crash signature above - worth flagging as early-silicon driver
   bug given the SKU is new.
4. Ask supervising faculty whether this is a known issue on other lab units of this same
   hardware.
