from __future__ import annotations

import torch
from torch import segment_reduce, repeat_interleave, exp
from torch.nn import Module, Linear

from einops import rearrange

from torch_einops_utils import lens_to_mask

# helpers

def nn_init_dirac_(weight, scale = 2.):
    from torch.nn.init import eye_
    eye_(weight)
    with torch.no_grad():
        weight.mul_(scale)

# functions

def segmented_softmax(
    logits,     # float[n d]
    lengths,    # int[s]
    eps = 1e-5
):
    seg_max = segment_reduce(logits, 'amax', lengths = lengths, unsafe = True)
    seg_max = seg_max.clamp(min = -1e9)

    attn = exp(logits - repeat_interleave(seg_max, lengths, dim = 0))

    seg_sum = segment_reduce(attn, 'sum', lengths = lengths, unsafe = True)
    attn = attn / repeat_interleave(seg_sum.clamp(min = eps), lengths, dim = 0)

    return attn

# classes

class SegmentedAttentionPool(Module):
    def __init__(self, dim):
        super().__init__()
        self.to_attn_logits = Linear(dim, dim, bias = False)

        nn_init_dirac_(self.to_attn_logits.weight)

    def forward(
        self,
        tokens,     # float[b n d]
        chunk_lens  # int[b c]
    ):
        batch, seq_len, dim = tokens.shape

        seq_lens = chunk_lens.sum(dim = -1)
        mask = lens_to_mask(seq_lens, max_len = seq_len)

        packed = tokens[mask]
        logits = self.to_attn_logits(packed)

        flat_lens = chunk_lens.flatten()

        attn = segmented_softmax(logits, flat_lens)
        pooled = segment_reduce(packed * attn, 'sum', lengths = flat_lens, unsafe = True)

        return rearrange(pooled, '(b c) d -> b c d', b = batch)
