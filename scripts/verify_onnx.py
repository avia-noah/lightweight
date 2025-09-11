import onnxruntime as ort
print("ONNX Runtime:", ort.__version__)
print("Providers:", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDA provider missing"
print("CUDAExecutionProvider is available.")
