import torch
import pytest

def test_chunker():
    from h_net_dynamic_chunking.h_net_dynamic_chunking import DynamicChunkingDownsampler

    downsampler = DynamicChunkingDownsampler(512)

    tokens = torch.randn(3, 1024, 512).requires_grad_()

    downsampled, upsample_fn, *_ = downsampler(tokens)

    assert upsample_fn(downsampled).shape == tokens.shape
