import sys, cv2, torch
from src.common.device import best_device

print("Python:", sys.version.split()[0])
print("OpenCV:", cv2.__version__)
print("Torch:", torch.__version__, "CUDA(wheel):", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

dev = best_device()
print("Selected device:", dev)

# simple GPU work
x = torch.randn(1024, 1024, device=dev)
y = x @ x.t()
print("Matmul mean:", float(y.mean()))

# OpenCV synthetic image sanity
import numpy as np
img = np.zeros((120, 240, 3), dtype=np.uint8)
cv2.putText(img, "OpenCV OK", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
out = "/tmp/opencv_sanity.png"
cv2.imwrite(out, img)
print("Wrote:", out)
