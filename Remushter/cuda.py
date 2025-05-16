# cuda_test.py
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA unavailable - checking why...")
    print(f"CUDA built with PyTorch: {torch.version.cuda or 'None'}")
    print(
        f"Available devices: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}")
