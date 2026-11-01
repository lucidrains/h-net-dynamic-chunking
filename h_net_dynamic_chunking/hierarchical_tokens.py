from collections import namedtuple

import torch
from einops import rearrange, repeat
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch_einops_utils import lens_to_mask

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def repeat_interleave_by_run_lens(
    tokens,
    run_lens,
    target_len = None,
    pad_value = 0.
):
    if run_lens.ndim == 1:
        out = torch.repeat_interleave(tokens, run_lens, dim = 0)

        if exists(target_len) and out.shape[0] < target_len:
            out = pad(out, (0, 0) * (out.ndim - 1) + (0, target_len - out.shape[0]), value = pad_value)

        return out

    rows = [torch.repeat_interleave(token, lens, dim = 0) for token, lens in zip(tokens, run_lens)]
    out = pad_sequence(rows, batch_first = True, padding_value = pad_value)

    if exists(target_len) and out.shape[1] < target_len:
        out = pad(out, (0, 0) * (out.ndim - 2) + (0, target_len - out.shape[1]), value = pad_value)

    return out

# hierarchy level

class HierarchyLevel(namedtuple('HierarchyLevel', 'embeds run_lens lens indices', defaults = (None,))):
    __slots__ = ()

    @property
    def mask(self):
        return lens_to_mask(self.lens, self.num_tokens)

    @property
    def num_tokens(self):
        return self.embeds.shape[1]

    @property
    def dim(self):
        return self.embeds.shape[-1]

# hierarchy

class HierarchicalTokens:
    def __init__(self, levels):
        self.levels = tuple(levels)

    def __len__(self):
        return len(self.levels)

    def __getitem__(self, index):
        return self.levels[index]

    def __iter__(self):
        return iter(self.levels)

    def __repr__(self):
        shapes = [level.embeds.shape for level in self.levels]
        return f'HierarchicalTokens(depth = {self.depth}, shapes = {shapes})'

    @property
    def depth(self):
        return len(self.levels)

    @property
    def finest(self):
        return self.levels[0]

    @property
    def coarsest(self):
        return self.levels[-1]

    @property
    def embeds(self):
        return tuple(level.embeds for level in self.levels)

    @property
    def indices(self):
        return tuple(level.indices for level in self.levels)

    @property
    def run_lens(self):
        return tuple(level.run_lens for level in self.levels)

    @property
    def lens(self):
        return tuple(level.lens for level in self.levels)

    def tokens(self, level = -1, use_indices = False):
        level = self.levels[level]

        if not use_indices:
            return level.embeds

        if not exists(level.indices):
            raise ValueError('this level has no vector quantized indices')

        return level.indices

    def repeat_interleave(self, level = -1, to_level = 0, use_indices = False):
        level = level % self.depth
        to_level = to_level % self.depth

        assert to_level <= level, 'can only expand from coarse to fine'

        tokens = self.tokens(level, use_indices = use_indices)
        pad_value = -1 if use_indices else 0.

        for index in range(level, to_level, -1):
            tokens = repeat_interleave_by_run_lens(
                tokens,
                self.levels[index].run_lens,
                target_len = self.levels[index - 1].num_tokens,
                pad_value = pad_value
            )

        return tokens

    def repeat_interleave_all(self, use_indices = False):
        return tuple(
            self.repeat_interleave(level = index, use_indices = use_indices)
            for index in range(self.depth)
        )

    def to_rle(self, use_indices = False):
        return tuple(
            (self.tokens(index, use_indices = use_indices), self.levels[index].run_lens)
            for index in range(self.depth - 1, 0, -1)
        )

# tokenization

def select_boundary_indices(indices, boundary_mask, num_chunks):
    if indices.shape[0] != boundary_mask.shape[0]:
        indices = repeat(indices, 'b ... -> (h b) ...', h = boundary_mask.shape[0] // indices.shape[0])

    return pad_sequence(
        indices[boundary_mask].split(num_chunks.tolist()),
        batch_first = True,
        padding_value = -1
    )

def flatten_heads(tokens):
    return rearrange(tokens, 'h b ... -> (h b) ...') if tokens.ndim == 4 else tokens

def repeat_level(level, repeats):
    if repeats == 1:
        return level

    return HierarchyLevel(
        embeds = repeat(level.embeds, 'b ... -> (r b) ...', r = repeats),
        run_lens = repeat(level.run_lens, 'b ... -> (r b) ...', r = repeats),
        lens = repeat(level.lens, 'b -> (r b)', r = repeats),
        indices = repeat(level.indices, 'b ... -> (r b) ...', r = repeats) if exists(level.indices) else None
    )

def build_levels(hnet, tokens, lens = None, vq_mask = None):
    batch, seq_len, device = *tokens.shape[:2], tokens.device
    input_mask = lens_to_mask(lens, max_len = seq_len) if exists(lens) else None

    # encode

    from x_transformers import Decoder

    encoded = hnet.encoder(tokens, mask = input_mask) if isinstance(hnet.encoder, Decoder) else hnet.encoder(tokens)

    # maybe quantize

    indices = None

    if exists(hnet.vq):
        vq_mask = default(vq_mask, input_mask)
        encoded, indices, _ = hnet.vq(encoded, mask = vq_mask)

        if exists(indices) and exists(vq_mask):
            indices = indices.masked_fill(~vq_mask, -1)

    level_lens = lens if exists(lens) else torch.full((batch,), seq_len, device = device, dtype = torch.long)

    # chunk

    outputs, intermediate = hnet.dynamic_sequence_chunker(encoded, lens = lens, return_intermediates = True)

    downsampled = flatten_heads(outputs.downsampled)
    chunk_lens = intermediate.chunk_lens
    num_chunks = (chunk_lens > 0).sum(dim = -1)

    # multi-headed chunkers fold heads into the batch

    heads = downsampled.shape[0] // batch
    assert heads * batch == downsampled.shape[0]

    base_run_lens = input_mask.long() if exists(input_mask) else torch.ones((batch, seq_len), dtype = torch.long, device = device)

    if heads > 1:
        encoded = repeat(encoded, 'b ... -> (h b) ...', h = heads)
        level_lens = repeat(level_lens, 'b -> (h b)', h = heads)
        base_run_lens = repeat(base_run_lens, 'b ... -> (h b) ...', h = heads)

        if exists(indices):
            indices = repeat(indices, 'b ... -> (h b) ...', h = heads)

    downsampled_indices = select_boundary_indices(indices, intermediate.boundary_mask, num_chunks) if exists(indices) else None

    levels = [
        HierarchyLevel(encoded, base_run_lens, level_lens, indices),
        HierarchyLevel(downsampled, chunk_lens, num_chunks, downsampled_indices)
    ]

    # recurse into nested h-net

    if hnet._is_nested_hnet and downsampled.shape[1] > 0:
        inner_levels = build_levels(hnet.network, hnet.proj_in(downsampled), lens = num_chunks)

        inner_batch = inner_levels[0].embeds.shape[0]
        batch_ratio = inner_batch // levels[1].embeds.shape[0]

        assert batch_ratio * levels[1].embeds.shape[0] == inner_batch

        if batch_ratio > 1:
            levels = [repeat_level(level, batch_ratio) for level in levels]

        levels.extend(inner_levels[1:])

    return levels

def tokenize_hnet(hnet, tokens, lens = None, vq_mask = None):
    return HierarchicalTokens(build_levels(hnet, tokens, lens = lens, vq_mask = vq_mask))
