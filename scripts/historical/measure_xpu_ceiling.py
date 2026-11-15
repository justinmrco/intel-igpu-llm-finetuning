"""
Directly measure the maximum allocatable XPU memory.

Allocates tensors of increasing size on the XPU until OOM.
Reports the largest successful allocation in GiB.
"""
import torch


def measure_ceiling(step_gib=0.1, max_gib=12.0):
    device = "xpu"
    if not torch.xpu.is_available():
        print("XPU not available")
        return None

    torch.xpu.empty_cache()
    baseline = torch.xpu.memory_allocated() / (1024 ** 3)
    total = torch.xpu.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Baseline allocated: {baseline:.3f} GiB")
    print(f"Total capacity reported: {total:.3f} GiB")
    print(f"Stepping by {step_gib} GiB, up to {max_gib} GiB")
    print()

    buffers = []
    largest_ok = baseline
    size_gib = baseline + step_gib

    while size_gib <= max_gib:
        try:
            n = int(step_gib * (1024 ** 3) // 4)
            buf = torch.empty(n, dtype=torch.float32, device=device)
            buffers.append(buf)
            largest_ok = torch.xpu.memory_allocated() / (1024 ** 3)
            print(f"Cumulative {size_gib:.1f} GiB: OK "
                  f"({largest_ok:.3f} GiB allocated)")
            size_gib += step_gib
        except torch.OutOfMemoryError:
            print(f"Cumulative {size_gib:.1f} GiB: OOM")
            break

    print()
    print(f"Largest successful allocation: {largest_ok:.3f} GiB")
    print(f"Total reported capacity: {total:.3f} GiB")
    return largest_ok


if __name__ == "__main__":
    measure_ceiling()