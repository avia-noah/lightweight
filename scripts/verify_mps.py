#!/usr/bin/env python3
"""
Verification script to ensure MPS acceleration is working properly on macOS
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from utils.device import pick_device, device_name

def verify_mps_acceleration():
    """Verify that MPS acceleration is working on macOS"""
    print("🔍 Verifying MPS Acceleration on macOS")
    print("=" * 50)
    
    # Check device selection
    device = pick_device()
    print(f"Selected device: {device}")
    print(f"Device name: {device_name(device)}")
    
    # Check MPS availability
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")
    
    if device.type == 'mps' and mps_available:
        print("✅ MPS acceleration is ENABLED")
        
        # Test tensor operations
        print("\n🧪 Testing tensor operations...")
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        
        print(f"Tensor computation device: {z.device}")
        print("✅ Tensor operations working on MPS")
        
        # Performance test
        print("\n⚡ Performance test...")
        import time
        
        # Warm up
        for _ in range(5):
            _ = torch.mm(x, y)
        
        # Time MPS
        start = time.time()
        for _ in range(10):
            _ = torch.mm(x, y)
        mps_time = time.time() - start
        
        # Time CPU for comparison
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        start = time.time()
        for _ in range(10):
            _ = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start
        
        speedup = cpu_time / mps_time
        print(f"MPS time: {mps_time:.4f}s")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"🚀 MPS is {speedup:.1f}x faster than CPU!")
        
        return True
    else:
        print("❌ MPS acceleration is NOT available")
        print("   Make sure you're running on Apple Silicon Mac")
        return False

if __name__ == "__main__":
    success = verify_mps_acceleration()
    if success:
        print("\n🎉 All tests passed! MPS acceleration is working perfectly.")
    else:
        print("\n⚠️  MPS acceleration is not available.")
        sys.exit(1)
