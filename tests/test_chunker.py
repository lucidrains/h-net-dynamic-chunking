import torch
import pytest
param = pytest.mark.parametrize

@param('handle_residual_proj', (False, True))
@param('straight_through_frac_vecs', (False, True))
@param('embed_chunk_lens', (False, True))
def test_chunker(
    handle_residual_proj,
    straight_through_frac_vecs,
    embed_chunk_lens
):
    from h_net_dynamic_chunking.h_net_dynamic_chunking import DynamicSequenceChunker

    downsampler = DynamicSequenceChunker(512, handle_residual_proj = handle_residual_proj, straight_through_frac_vecs = straight_through_frac_vecs, embed_chunk_lens = embed_chunk_lens)

    tokens = torch.randn(3, 1024, 512).requires_grad_()

    downsampled, upsample_fn, aux_loss = downsampler(tokens)

    aux_loss.mean().backward()

    assert upsample_fn(downsampled).shape == tokens.shape

@param('has_lens', (False, True))
@param('handle_residual_proj', (False, True))
@param('embed_chunk_lens', (False, True))
def test_chunker_with_lens(
    has_lens,
    handle_residual_proj,
    embed_chunk_lens
):
    from h_net_dynamic_chunking.h_net_dynamic_chunking import DynamicSequenceChunker

    batch, max_len, dim = 4, 64, 128

    downsampler = DynamicSequenceChunker(dim, handle_residual_proj = handle_residual_proj, embed_chunk_lens = embed_chunk_lens)

    tokens = torch.randn(batch, max_len, dim).requires_grad_()

    lens = torch.tensor([17, 32, 64, 25]) if has_lens else None

    # test chunk_lens sum

    chunk_lens = downsampler(tokens, lens = lens, return_only_chunk_lens = True)
    chunk_sums = chunk_lens.sum(dim = -1)

    expected = lens if has_lens else torch.full((batch,), max_len)
    assert (chunk_sums == expected).all(), f'chunk_lens.sum() {chunk_sums.tolist()} != expected {expected.tolist()}'

    # test full forward + upsample

    downsampled, upsample_fn, aux_loss = downsampler(tokens, lens = lens)

    upsampled = upsample_fn(downsampled)
    assert upsampled.shape == tokens.shape

    # test backward

    (upsampled.sum() + aux_loss).backward()

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

    out = net(tokens) # (1, 1024, 512), (1,)
    assert out.loss.numel() == 1

    out = net(tokens, return_intermediates = True)
    assert all(isinstance(el, Intermediates) for el in out.intermediates)

    # validate absolute_chunk_lens at every level sum to original seq len

    seq_len = tokens.shape[1]

    for intermediate in out.intermediates:
        assert (intermediate.absolute_chunk_lens.sum(dim = -1) == seq_len).all()

@param('has_lens', (False, True))
@param('handle_residual_proj', (False, True))
@param('embed_chunk_lens', (False, True))
def test_multihead_chunker_with_lens(
    has_lens,
    handle_residual_proj,
    embed_chunk_lens
):
    from h_net_dynamic_chunking.multi_head_h_net_dynamic_chunking import MultiHeadDynamicSequenceChunker

    batch, max_len, dim, heads = 4, 64, 128, 4

    downsampler = MultiHeadDynamicSequenceChunker(dim, heads=heads, handle_residual_proj=handle_residual_proj, embed_chunk_lens=embed_chunk_lens)

    tokens = torch.randn(batch, max_len, dim).requires_grad_()

    lens = torch.tensor([17, 32, 64, 25]) if has_lens else None

    # test chunk_lens sum

    chunk_lens = downsampler(tokens, lens = lens, return_only_chunk_lens = True)
    chunk_sums = chunk_lens.sum(dim = -1)

    expected = lens if has_lens else torch.full((batch,), max_len)

    # since chunk_lens has shape (heads * batch), we expect it to match repeated lens
    expected = expected.repeat(heads)

    assert (chunk_sums == expected).all(), f'chunk_lens.sum() {chunk_sums.tolist()} != expected {expected.tolist()}'

    # test full forward + upsample

    downsampled, upsample_fn, aux_loss = downsampler(tokens, lens = lens)

    upsampled = upsample_fn(downsampled)
    assert upsampled.shape == tokens.shape

    # test backward

    (upsampled.sum() + aux_loss).backward()

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

    out = net(tokens, return_intermediates = True)

    downsampled = out.intermediates.input_downsampled_tokens
    batch, down_seq, dim = downsampled.shape
    assert batch == 1 and dim == 512 and down_seq <= 1024

@param('use_vq', (False, True))
def test_vq_and_fetch_indices_from_intermediates(use_vq):
    from torch import nn
    from vector_quantize_pytorch import VectorQuantize
    from h_net_dynamic_chunking.h_net import HNet, exists

    if use_vq:
        vq = dict(
            dim = 512,
            codebook_size = 256
        )
    else:
        vq = None

    net = HNet(
        nn.Identity(),
        nn.Linear(1024, 1024),
        nn.Identity(),
        dim = 512,
        dim_inner = 1024,
        vq = vq
    )

    tokens = torch.randn(1, 1024, 512)
    out = net(tokens, return_intermediates = True)

    indices = out.intermediates.quantized_downsampled_indices

    if use_vq:
        assert exists(indices)
        assert indices.ndim == 2
        downsampled = out.intermediates.input_downsampled_tokens
        assert indices.shape == downsampled.shape[:2], f'indices shape {indices.shape} does not match downsampled chunks {downsampled.shape[:2]}'
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
    out = net(tokens)

    assert out.output.shape == tokens.shape

def test_hnet_cache_parity():
    import torch
    from x_transformers import Decoder
    from h_net_dynamic_chunking import HNet

    dim = 64
    heads = 1
    seq_len = 16

    encoder = Decoder(dim = dim, depth = 1, heads = heads)
    inner_network = Decoder(dim = dim, depth = 2, heads = heads)
    decoder = Decoder(dim = dim, depth = 1, heads = heads)

    hnet = HNet(
        encoder = encoder,
        network = inner_network,
        decoder = decoder,
        dim = dim,
        heads = heads,
    )
    hnet.eval()

    tokens = torch.randn(1, seq_len, dim)

    with torch.no_grad():
        parallel_out = hnet(tokens).output

    cache = None
    sequential_outs = []

    with torch.no_grad():
        for i in range(seq_len):
            token = tokens[:, i:i+1]
            out = hnet(token, cache=cache, return_hiddens = True)
            cache = out.next_cache
            sequential_outs.append(out.output)

    sequential_out = torch.cat(sequential_outs, dim = 1)
    assert torch.allclose(parallel_out, sequential_out, atol = 1e-4)

def test_nested_hnet_cache_parity():
    import torch
    from x_transformers import Decoder
    from h_net_dynamic_chunking import HNet

    dim = 64
    heads = 1
    seq_len = 16

    encoder = Decoder(dim = dim, depth = 1, heads = heads)
    inner_network1 = Decoder(dim = dim, depth = 2, heads = heads)
    inner_network2 = Decoder(dim = dim, depth = 2, heads = heads)
    inner_network3 = Decoder(dim = dim, depth = 2, heads = heads)
    decoder = Decoder(dim = dim, depth = 1, heads = heads)

    hnet_inner = HNet(
        encoder = inner_network1,
        network = inner_network2,
        decoder = inner_network3,
        dim = dim,
        heads = heads,
    )

    hnet = HNet(
        encoder = encoder,
        network = hnet_inner,
        decoder = decoder,
        dim = dim,
        heads = heads,
    )
    hnet.eval()

    tokens = torch.randn(1, seq_len, dim)

    with torch.no_grad():
        parallel_out = hnet(tokens).output

    cache = None
    sequential_outs = []

    with torch.no_grad():
        for i in range(seq_len):
            token = tokens[:, i:i+1]
            out = hnet(token, cache=cache, return_hiddens=True)
            cache = out.next_cache
            sequential_outs.append(out.output)

    sequential_out = torch.cat(sequential_outs, dim = 1)
    assert torch.allclose(parallel_out, sequential_out, atol = 1e-4)

def test_deeply_nested_hnet_cache_parity():
    import torch
    from x_transformers import Decoder
    from h_net_dynamic_chunking import HNet

    dim = 64
    heads = 1
    seq_len = 16

    encoder = Decoder(dim = dim, depth = 1, heads = heads)

    inner_network1 = Decoder(dim = dim, depth = 1, heads = heads)
    inner_network2 = Decoder(dim = dim, depth = 1, heads = heads)
    inner_network3 = Decoder(dim = dim, depth = 1, heads = heads)

    inner_inner_network1 = Decoder(dim = dim, depth = 1, heads = heads)
    inner_inner_network2 = Decoder(dim = dim, depth = 1, heads = heads)
    inner_inner_network3 = Decoder(dim = dim, depth = 1, heads = heads)

    decoder = Decoder(dim = dim, depth = 1, heads = heads)

    hnet_inner_inner = HNet(
        encoder = inner_inner_network1,
        network = inner_inner_network2,
        decoder = inner_inner_network3,
        dim = dim,
        heads = heads,
    )

    hnet_inner = HNet(
        encoder = inner_network1,
        network = hnet_inner_inner,
        decoder = inner_network3,
        dim = dim,
        heads = heads,
    )

    hnet = HNet(
        encoder = encoder,
        network = hnet_inner,
        decoder = decoder,
        dim = dim,
        heads = heads,
    )
    hnet.eval()

    tokens = torch.randn(1, seq_len, dim)

    with torch.no_grad():
        parallel_out = hnet(tokens).output

    cache = None
    sequential_outs = []

    with torch.no_grad():
        for i in range(seq_len):
            token = tokens[:, i:i+1]
            out = hnet(token, cache=cache, return_hiddens=True)
            cache = out.next_cache
            sequential_outs.append(out.output)

    sequential_out = torch.cat(sequential_outs, dim = 1)
    assert torch.allclose(parallel_out, sequential_out, atol = 1e-4)
