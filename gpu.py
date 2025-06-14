import torch

print("=== BASIC GPU CHECK ===")

# Check if CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")
    
    # Check if it's A100
    if "A100" in gpu_name:
        print("✅ A100 DETECTED!")
    else:
        print(f"⚠️  Not A100 (Current: {gpu_name})")
    
    # Memory info
    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Memory: {memory_gb:.1f} GB")
    
    # Quick test
    device = torch.device('cuda')
    test_tensor = torch.randn(100, 100, device=device)
    print("✅ GPU Test: PASSED")
    
else:
    print("❌ No GPU available!")

print("=== END CHECK ===")