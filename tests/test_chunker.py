import torch
import pytest
param = pytest.mark.parametrize

@param('handle_residual_proj', (False, True))
@param('straight_through_frac_vecs', (False, True))
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

def test_multihead_hnet():
    from h_net_dynamic_chunking import MultiHeadDynamicSequenceChunker

    downsampler1 = MultiHeadDynamicSequenceChunker(512, heads = 4)
    downsampler2 = MultiHeadDynamicSequenceChunker(512, heads = 4)
    downsampler3 = MultiHeadDynamicSequenceChunker(512, heads = 4)

    tokens = torch.randn(3, 1024, 512).requires_grad_()

    downsampled1, upsample_fn1, aux_loss1 = downsampler1(tokens)

    # hierarchical network 1 ...

    downsampled2, upsample_fn2, aux_loss2 = downsampler2(downsampled1)

    # hierarchical network 2 ...

    downsampled3, upsample_fn3, aux_loss3 = downsampler3(downsampled2)

    # inner most network

    # reconstituting

    assert upsample_fn1(upsample_fn2(upsample_fn3(downsampled3))).shape == tokens.shape

def test_two_inner_networks():

    from torch.nn import GRU, LSTM
    from h_net_dynamic_chunking import MultiHeadDynamicSequenceChunker

    downsampler = MultiHeadDynamicSequenceChunker(512, heads = 2, heads_merged_with_batch = False)

    tokens = torch.randn(3, 1024, 512).requires_grad_()

    downsampled, upsample_fn, aux_loss = downsampler(tokens)

    network1 = GRU(512, 512, batch_first = True)
    network2 = LSTM(512, 512, batch_first = True)

    first_head, second_head = downsampled

    out1, _ = network1(first_head)
    out2, _ = network2(second_head)

    # reconstituting

    assert upsample_fn([out1, out2]).shape == tokens.shape


def test_hierarchial_ar_loss():
    from h_net_dynamic_chunking import MultiHeadDynamicSequenceChunker

    downsampler1 = MultiHeadDynamicSequenceChunker(512, heads = 2)
    downsampler2 = MultiHeadDynamicSequenceChunker(512, heads = 2, add_hier_ar_loss = True)
    downsampler3 = MultiHeadDynamicSequenceChunker(512, heads = 2)

    tokens = torch.randn(3, 1024, 512).requires_grad_()

    downsampled1, upsample_fn1, aux_loss1 = downsampler1(tokens)

    # hierarchical network 1 ...

    downsampled2, upsample_fn2, aux_loss2 = downsampler2(downsampled1)

    # hierarchical network 2 ...

    downsampled3, upsample_fn3, aux_loss3 = downsampler3(downsampled2)

    # inner most network

    # reconstituting

    upsampled3 = upsample_fn3(downsampled3)

    upsampled2, hier_ar_loss2 = upsample_fn2(upsampled3)

    upsampled = upsample_fn1(upsampled2)

    assert hier_ar_loss2.numel() == 1

def test_access_downsampled_from_h_net_intermediate():
    from torch import nn
    from h_net_dynamic_chunking.h_net import HNet

    net = HNet(
        nn.Identity(),
        nn.Linear(1024, 1024),
        nn.Identity(),
        dim = 512,
        dim_inner = 1024,
    )

    tokens = torch.randn(1, 1024, 512)

    out, aux_loss, intermediates = net(tokens, return_intermediates = True)

    downsampled = intermediates.input_downsampled_tokens
    batch, down_seq, dim = downsampled.shape
    assert batch == 1 and dim == 512 and down_seq <= 1024

@param('use_vq', (False, True))
def test_vq_and_fetch_indices_from_intermediates(use_vq):
    from torch import nn
    from vector_quantize_pytorch import VectorQuantize
    from h_net_dynamic_chunking.h_net import HNet, exists

    vq = VectorQuantize(
        dim = 1024,
        codebook_size = 256,
        use_cosine_sim = True
    ) if use_vq else None

    net = HNet(
        nn.Identity(),
        nn.Linear(1024, 1024),
        nn.Identity(),
        dim = 512,
        dim_inner = 1024,
        vq = vq
    )

    tokens = torch.randn(1, 1024, 512)
    out, aux_loss, intermediates = net(tokens, return_intermediates = True)

    indices = intermediates.quantized_downsampled_indices
    
    if use_vq:
        assert exists(indices)
        assert indices.ndim == 2
    else:
        assert not exists(indices)

def test_inner_network_pos_kwarg():
    from torch import nn
    from x_transformers import Decoder
    from h_net_dynamic_chunking.h_net import HNet

    dim = 512
    dim_inner = 256

    inner_network = Decoder(
        dim = dim_inner,
        depth = 2,
        heads = 4,
        rotary_pos_emb = True
    )

    net = HNet(
        nn.Identity(),
        inner_network,
        nn.Identity(),
        dim = dim,
        dim_inner = dim_inner,
        inner_network_rel_pos_kwarg = 'pos'
    )

    tokens = torch.randn(2, 128, dim)
    out, aux_loss = net(tokens)

    assert out.shape == tokens.shape
