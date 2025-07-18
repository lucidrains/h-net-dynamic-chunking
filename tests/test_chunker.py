import torch
import pytest

@pytest.mark.parametrize('handle_residual_proj', (False, True))
@pytest.mark.parametrize('straight_through_frac_vecs', (False, True))
def test_chunker(
    handle_residual_proj,
    straight_through_frac_vecs
):
    from h_net_dynamic_chunking.h_net_dynamic_chunking import DynamicSequenceChunker

    downsampler = DynamicSequenceChunker(512, handle_residual_proj = handle_residual_proj, straight_through_frac_vecs = straight_through_frac_vecs)

    tokens = torch.randn(3, 1024, 512).requires_grad_()

    downsampled, upsample_fn, aux_loss = downsampler(tokens)

    aux_loss.mean().backward()

    assert upsample_fn(downsampled).shape == tokens.shape

def test_hnet():
    from torch import nn
    from h_net_dynamic_chunking.h_net import HNet
    from h_net_dynamic_chunking.h_net_dynamic_chunking import Intermediates

    net = HNet(
        nn.Identity(),
        HNet(
            nn.Identity(),
            HNet(
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
                dim = 2048
            ),
            nn.Identity(),
            dim = 1024,
            dim_inner = 2048
        ),
        nn.Identity(),
        dim = 512,
        dim_inner = 1024,
    )

    tokens = torch.randn(1, 1024, 512)

    out, aux_loss = net(tokens) # (1, 1024, 512), (1,)
    assert aux_loss.numel() == 1

    net, aux_loss, intermediates = net(tokens, return_intermediates = True)
    assert all(isinstance(el, Intermediates) for el in intermediates)
