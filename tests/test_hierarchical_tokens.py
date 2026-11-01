import pytest
import torch

from torch import nn

from h_net_dynamic_chunking import (
    HNet,
    HierarchyLevel,
    HierarchicalTokens,
    repeat_interleave_by_run_lens
)

param = pytest.mark.parametrize

# helpers

def build_nested_hnet(depth, dim, heads = 1, vq = None, dim_inner = None, **chunker_kwargs):
    if depth == 1:
        return HNet(
            encoder = nn.Identity(),
            network = nn.Identity(),
            decoder = nn.Identity(),
            dim = dim,
            heads = heads,
            vq = vq,
            **chunker_kwargs
        )

    return HNet(
        encoder = nn.Identity(),
        network = build_nested_hnet(
            depth - 1,
            dim_inner or dim,
            heads = heads,
            vq = vq,
            dim_inner = dim_inner,
            **chunker_kwargs
        ),
        decoder = nn.Identity(),
        dim = dim,
        dim_inner = dim_inner,
        heads = heads,
        vq = vq,
        **chunker_kwargs
    )

# tests

@param('depth', (1, 2))
@param('heads', (1, 2))
@param('has_lens', (False, True))
def test_tokenize_shapes_and_run_lengths(depth, heads, has_lens):
    torch.manual_seed(0)

    dim = 32
    batch, seq_len = 2, 32

    net = build_nested_hnet(depth, dim, heads = heads, boundary_threshold = 0.05).eval()

    tokens = torch.randn(batch, seq_len, dim)
    lens = torch.tensor([32, 17]) if has_lens else None

    with torch.no_grad():
        hierarchy = net.tokenize(tokens, lens = lens)

    assert isinstance(hierarchy, HierarchicalTokens)
    assert hierarchy.finest.num_tokens == seq_len

    expected_depth = depth + 1
    assert hierarchy.depth == expected_depth, f'expected depth {expected_depth}, got {hierarchy.depth}'

    # each nested multi-headed chunker folds its heads into the batch

    aligned_batch = batch * heads ** depth
    aligned_lens = lens.repeat(heads ** depth) if has_lens else torch.full((aligned_batch,), seq_len)

    assert hierarchy.finest.lens.tolist() == aligned_lens.tolist()

    for index, level in enumerate(hierarchy):
        assert isinstance(level, HierarchyLevel)
        assert level.embeds.shape[0] == aligned_batch
        assert level.run_lens.shape == (aligned_batch, level.num_tokens)
        assert level.lens.shape == (aligned_batch,)
        assert level.mask.sum(dim = -1).tolist() == level.lens.tolist()

        if index == 0:
            continue

        # run lengths sum to the valid length of the finer level

        assert (level.run_lens.sum(dim = -1) == hierarchy.levels[index - 1].lens).all()

        # one level expansion

        expanded = hierarchy.repeat_interleave(level = index, to_level = index - 1)
        assert expanded.shape[0] == aligned_batch
        assert expanded.shape[1] == hierarchy.levels[index - 1].num_tokens
        assert expanded.shape[-1] == level.dim

@param('heads', (1, 2))
def test_tokenize_matches_chunker_run_lengths(heads):
    torch.manual_seed(0)

    dim = 32
    batch, seq_len = 3, 24
    lens = torch.tensor([24, 17, 9])

    net = build_nested_hnet(1, dim, heads = heads, boundary_threshold = 0.05).eval()
    tokens = torch.randn(batch, seq_len, dim)

    with torch.no_grad():
        hierarchy = net.tokenize(tokens, lens = lens)
        chunk_lens = net.dynamic_sequence_chunker(tokens, lens = lens, return_only_chunk_lens = True)

    assert torch.equal(hierarchy.levels[1].run_lens, chunk_lens)
    assert torch.equal(hierarchy.levels[1].lens, (chunk_lens > 0).sum(dim = -1))

@param('depth', (1, 2, 3))
def test_repeat_interleave_matches_manual(depth):
    torch.manual_seed(0)

    dim = 16
    batch, seq_len = 2, 40
    lens = torch.tensor([40, 23])

    net = build_nested_hnet(depth, dim, boundary_threshold = 0.2).eval()
    tokens = torch.randn(batch, seq_len, dim)

    with torch.no_grad():
        hierarchy = net.tokenize(tokens, lens = lens)

    assert hierarchy.depth >= 2

    for index in range(1, hierarchy.depth):
        level = hierarchy.levels[index]
        finer = hierarchy.levels[index - 1]

        expanded = hierarchy.repeat_interleave(level = index, to_level = index - 1)

        for batch_index in range(batch):
            valid = finer.lens[batch_index]

            manual = torch.repeat_interleave(
                level.embeds[batch_index, :level.lens[batch_index]],
                level.run_lens[batch_index, :level.lens[batch_index]],
                dim = 0
            )

            assert torch.allclose(expanded[batch_index, :valid], manual)

            # padded positions are zero

            assert (expanded[batch_index, valid:] == 0.).all()

        # expanding all the way to base matches doing it one level at a time

        expanded_to_base = hierarchy.repeat_interleave(level = index, to_level = 0)

        recursive = level.embeds
        for inner_index in range(index, 0, -1):
            recursive = repeat_interleave_by_run_lens(
                recursive,
                hierarchy.levels[inner_index].run_lens,
                target_len = hierarchy.levels[inner_index - 1].num_tokens
            )

        assert torch.allclose(expanded_to_base, recursive)

def test_repeat_interleave_all_levels():
    torch.manual_seed(0)

    dim = 16
    batch, seq_len = 2, 32
    lens = torch.tensor([32, 19])

    net = build_nested_hnet(2, dim, boundary_threshold = 0.2).eval()
    tokens = torch.randn(batch, seq_len, dim)

    with torch.no_grad():
        hierarchy = net.tokenize(tokens, lens = lens)
        expanded = hierarchy.repeat_interleave_all()

    assert len(expanded) == hierarchy.depth
    assert torch.equal(expanded[0], hierarchy.finest.embeds)
    assert expanded[0].shape == (batch, seq_len, dim)

    for index, level_expanded in enumerate(expanded):
        assert level_expanded.shape[:2] == (batch, seq_len)
        assert level_expanded.shape[-1] == hierarchy.levels[index].dim

    # coarsest expansion is the full sequence

    assert torch.allclose(expanded[-1], hierarchy.repeat_interleave())

@param('depth', (1, 2))
def test_tokenize_vq_indices(depth):
    torch.manual_seed(0)

    dim = 32
    batch, seq_len = 2, 40
    lens = torch.tensor([40, 23])

    net = build_nested_hnet(
        depth,
        dim,
        vq = dict(dim = dim, codebook_size = 64),
        boundary_threshold = 0.2
    ).eval()

    tokens = torch.randn(batch, seq_len, dim)

    with torch.no_grad():
        hierarchy = net.tokenize(tokens, lens = lens)

    assert all(index is not None for index in hierarchy.indices)

    vq_modules = []

    def collect_vq(module):
        vq_modules.append(module.vq)

        if isinstance(module.network, HNet):
            collect_vq(module.network)

    collect_vq(net)

    for index, level in enumerate(hierarchy):
        assert level.indices is not None
        assert level.indices.shape == (batch, level.num_tokens)
        assert level.indices.dtype == torch.long

        valid = level.mask
        vq = vq_modules[max(0, index - 1)]
        codes = vq.get_output_from_indices(level.indices.clamp(min = 0))

        assert torch.allclose(codes[valid], level.embeds[valid], atol = 1e-6)

        # padding is marked with -1

        assert (level.indices[~valid] == -1).all()

    expanded_indices = hierarchy.repeat_interleave(use_indices = True)
    assert expanded_indices.shape == (batch, seq_len)
    assert expanded_indices.dtype == torch.long

    for batch_index, valid in enumerate(lens.tolist()):
        assert (expanded_indices[batch_index, valid:] == -1).all()

    expanded_all = hierarchy.repeat_interleave_all(use_indices = True)
    assert len(expanded_all) == hierarchy.depth
    assert all(expanded.shape == (batch, seq_len) for expanded in expanded_all)

    rle = hierarchy.to_rle(use_indices = True)
    assert len(rle) == hierarchy.depth - 1

    assert torch.equal(rle[0][0], hierarchy.coarsest.indices)
    assert torch.equal(rle[0][1], hierarchy.coarsest.run_lens)

def test_tokenize_vq_partial_hierarchy():
    torch.manual_seed(0)

    dim = 32
    batch, seq_len = 2, 32
    lens = torch.tensor([32, 21])

    net = HNet(
        encoder = nn.Identity(),
        network = HNet(
            encoder = nn.Identity(),
            network = nn.Identity(),
            decoder = nn.Identity(),
            dim = dim
        ),
        decoder = nn.Identity(),
        dim = dim,
        vq = dict(dim = dim, codebook_size = 32),
        boundary_threshold = 0.2
    ).eval()

    tokens = torch.randn(batch, seq_len, dim)

    with torch.no_grad():
        hierarchy = net.tokenize(tokens, lens = lens)

    assert hierarchy.depth >= 3
    assert any(index is not None for index in hierarchy.indices)
    assert not all(index is not None for index in hierarchy.indices)
    assert hierarchy.levels[0].indices is not None
    assert hierarchy.levels[1].indices is not None
    assert hierarchy.levels[2].indices is None

    # level 1 can still be expanded with codes

    assert hierarchy.repeat_interleave(level = 1, use_indices = True).shape == (batch, seq_len)

    with pytest.raises(ValueError):
        hierarchy.repeat_interleave(level = 2, use_indices = True)

    with pytest.raises(ValueError):
        hierarchy.repeat_interleave_all(use_indices = True)

@param('depth', (1, 2))
@param('heads', (1, 2))
def test_tokenize_matches_forward_intermediates(depth, heads):
    torch.manual_seed(0)

    dim = 32
    batch, seq_len = 2, 32
    lens = torch.tensor([32, 23])

    net = build_nested_hnet(depth, dim, heads = heads, boundary_threshold = 0.2).eval()
    tokens = torch.randn(batch, seq_len, dim)

    with torch.no_grad():
        hierarchy = net.tokenize(tokens, lens = lens)
        out = net(tokens, lens = lens, return_intermediates = True)

    intermediates = out.intermediates

    if hasattr(intermediates, '_fields'):
        intermediates = (intermediates,)

    assert len(intermediates) == hierarchy.depth - 1

    for index, intermediate in enumerate(intermediates, start = 1):
        level = hierarchy.levels[index]

        # nested multi-headed h-nets fold heads into the batch, repeat lower
        # levels so every level of the hierarchy is aligned

        repeats = level.embeds.shape[0] // intermediate.chunk_lens.shape[0]
        assert repeats * intermediate.chunk_lens.shape[0] == level.embeds.shape[0]

        expected_run_lens = intermediate.chunk_lens.repeat(repeats, 1)

        assert torch.equal(level.run_lens, expected_run_lens)
        assert torch.equal(level.lens, (expected_run_lens > 0).sum(dim = -1))

def test_tokenize_skips_inner_network_and_decoder():
    torch.manual_seed(0)

    class Boom(nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError('should not be called')

    dim = 16
    tokens = torch.randn(2, 24, dim)

    net = HNet(
        encoder = nn.Identity(),
        network = Boom(),
        decoder = Boom(),
        dim = dim
    ).eval()

    with torch.no_grad():
        hierarchy = net.tokenize(tokens)

    assert hierarchy.depth == 2

    # nested h-net with a plain innermost network

    inner = HNet(
        encoder = nn.Identity(),
        network = Boom(),
        decoder = Boom(),
        dim = dim
    )

    net = HNet(
        encoder = nn.Identity(),
        network = inner,
        decoder = Boom(),
        dim = dim
    ).eval()

    with torch.no_grad():
        hierarchy = net.tokenize(tokens)

    assert hierarchy.depth >= 2

def test_tokenize_empty_chunks():
    torch.manual_seed(0)

    dim = 16
    tokens = torch.randn(2, 12, dim)
    lens = torch.tensor([0, 0])

    net = build_nested_hnet(
        2,
        dim,
        vq = dict(dim = dim, codebook_size = 16)
    ).eval()

    with torch.no_grad():
        hierarchy = net.tokenize(tokens, lens = lens)

    assert hierarchy.depth == 2
    assert hierarchy.levels[1].num_tokens == 0
    assert hierarchy.repeat_interleave().shape == (2, 12, dim)
    assert hierarchy.repeat_interleave(use_indices = True).shape == (2, 12)

def test_repeat_interleave_utility():
    # batched embeddings

    tokens = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])[None]
    run_lens = torch.tensor([[2, 0, 1]])

    out = repeat_interleave_by_run_lens(tokens, run_lens, target_len = 4)
    expected = torch.tensor([[1., 2.], [1., 2.], [5., 6.], [0., 0.]])[None]

    assert torch.equal(out, expected)

    # batched indices

    indices = torch.tensor([[7, 8, 9]])
    out_indices = repeat_interleave_by_run_lens(indices, run_lens, target_len = 4, pad_value = -1)
    assert out_indices.tolist() == [[7, 7, 9, -1]]

    # unbatched

    out_unbatched = repeat_interleave_by_run_lens(tokens[0], run_lens[0], target_len = 4)
    assert torch.equal(out_unbatched, expected[0])

def test_tokenize_with_dim_change():
    torch.manual_seed(0)

    dim, dim_inner = 16, 32
    batch, seq_len = 2, 32
    lens = torch.tensor([32, 20])

    net = HNet(
        encoder = nn.Identity(),
        network = HNet(
            encoder = nn.Identity(),
            network = nn.Identity(),
            decoder = nn.Identity(),
            dim = dim_inner
        ),
        decoder = nn.Identity(),
        dim = dim,
        dim_inner = dim_inner,
        boundary_threshold = 0.2
    ).eval()

    tokens = torch.randn(batch, seq_len, dim)

    with torch.no_grad():
        hierarchy = net.tokenize(tokens, lens = lens)

    assert hierarchy.depth >= 2
    assert hierarchy.levels[0].dim == dim
    assert hierarchy.levels[1].dim == dim
    assert hierarchy.levels[-1].dim == dim_inner

    # expansion keeps the coarse level's dimension

    assert hierarchy.repeat_interleave().shape == (batch, seq_len, dim_inner)
