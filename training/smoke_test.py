"""Verify all six benchmark architectures produce five logits."""

from __future__ import annotations

import gc

import torch

from training.models import SUPPORTED_MODELS, build_model


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda")
    for name in SUPPORTED_MODELS:
        model = build_model(name, pretrained=False).eval().to(device)
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 224, 224, device=device))
        assert output.shape == (1, 5), (name, output.shape)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"{name}: output={tuple(output.shape)}, parameters={parameters}")
        del output, model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
