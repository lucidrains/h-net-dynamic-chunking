import torch
import pytest

@pytest.mark.parametrize('handle_residual_proj', (False, True))
def test_chunker(handle_residual_proj):
    from h_net_dynamic_chunking.h_net_dynamic_chunking import DynamicChunkingDownsampler

    downsampler = DynamicChunkingDownsampler(512, handle_residual_proj = handle_residual_proj)

    tokens = torch.randn(3, 1024, 512).requires_grad_()

    downsampled, upsample_fn, aux_loss = downsampler(tokens)

    aux_loss.mean().backward()

    assert upsample_fn(downsampled).shape == tokens.shape
