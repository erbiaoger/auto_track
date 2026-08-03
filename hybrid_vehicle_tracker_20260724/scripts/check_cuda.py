from __future__ import annotations

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; GPU training is mandatory")
    tensor = torch.arange(4096, device="cuda", dtype=torch.float32)
    print(
        {
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "test_sum": tensor.sum().item(),
        }
    )


if __name__ == "__main__":
    main()
