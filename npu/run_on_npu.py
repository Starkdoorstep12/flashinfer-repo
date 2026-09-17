import sys
import torch
import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

sys.path.insert(0, r"C:\flashinfer-repo\npu")
from run_npu_indexer_export import q, K_padded, w, valid_mask, token_ids

ep_lib_path = qnn_ep.get_library_path()
lib_registration_name = "QNNExecutionProvider"
ort.register_execution_provider_library(lib_registration_name, ep_lib_path)

all_ep_devices = ort.get_ep_devices()
npu_devices = [d for d in all_ep_devices if d.ep_name == lib_registration_name and str(d.device.type) == "OrtHardwareDeviceType.NPU"]

print("NPU devices found:", len(npu_devices))

session_options = ort.SessionOptions()
session_options.add_provider_for_devices(npu_devices, {})

session = ort.InferenceSession(r"C:\flashinfer-repo\npu\indexer.onnx", sess_options=session_options)

inputs = {
    "q": q.numpy(),
    "K_padded": K_padded.numpy(),
    "weights": w.numpy(),
    "valid_mask": valid_mask.numpy(),
    "token_ids": token_ids.numpy(),
}

result = session.run(None, inputs)
topk_indices_npu = result[0]

print("NPU output shape:", topk_indices_npu.shape)
for b in range(topk_indices_npu.shape[0]):
    valid = topk_indices_npu[b][topk_indices_npu[b] != -1]
    print(f"Batch {b}: {len(valid)} valid tokens from NPU run")
