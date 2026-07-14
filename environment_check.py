"""
Check whether the environment supports GPU and specific Precision formats.
"""
import torch


def check_cuda_available():
    """
    Check if GPU is available
    """
    assert torch.cuda.is_available(), "No GPU detected"
    return torch.cuda.get_device_name(0)


def check_gpu_memory():
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return round(total, 2)


def check_bf16_support():
    """
    Can this GPU run BF16 format?
    """
    return torch.cuda.is_bf16_supported()


def print_env_summary():
    """
    Print the environment (hardware) summary
    """
    print(f"GPU: {check_cuda_available()}")
    print(f"Memory: {check_gpu_memory()} GB")
    print(f"BF16 supported: {check_bf16_support()}")


if __name__ == "__main__":
    print_env_summary()