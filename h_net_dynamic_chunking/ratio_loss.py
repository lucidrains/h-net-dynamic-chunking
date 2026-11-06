# auxiliary ratio loss from section 2.3.2 of the paper, eq (10)

from functools import partial
from math import ceil

import torch
from torch.nn.functional import pad

from einops import rearrange
from torch_einops_utils import masked_mean

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main

def chunked_ratio_loss(
    probs,
    frac_vecs,
    mask = None,
    target_avg_token_length = 6.,
    chunk_size = None
):
    N = target_avg_token_length

    is_chunked = exists(chunk_size) and chunk_size < probs.shape[1]

    if is_chunked:
        length = probs.shape[1]

        # pad to multiple of chunk size, masking out tail

        chunked_len = ceil(length / chunk_size) * chunk_size
        pad_len = chunked_len - length

        mask = default(mask, torch.ones_like(probs, dtype = torch.bool))

        if pad_len > 0:
            probs = pad(probs, (0, pad_len))
            frac_vecs = pad(frac_vecs, (0, pad_len))
            mask = pad(mask, (0, pad_len), value = False)

        # fold chunks into batch -> (b k) n

        to_chunks = partial(rearrange, pattern = 'b (k n) -> (b k) n', n = chunk_size)
        probs, frac_vecs, mask = map(to_chunks, (probs, frac_vecs, mask))

    G = masked_mean(probs, mask = mask, dim = -1)
    F = masked_mean(frac_vecs, mask = mask, dim = -1)

    loss = N / (N - 1) * ((N - 1) * F * G + (1. - F) * (1. - G))

    if not is_chunked:
        return loss.mean()

    # ignore windows that are entirely padding

    return masked_mean(loss, mask = mask.any(dim = -1), dim = -1)

# orthogonal boundary regularizer (zero-sum projected within-window mutual information)

def binary_entropy(p, eps = 1e-6):
    p = p.clamp(min = eps, max = 1. - eps)
    return -(p * p.log() + (1. - p) * (1. - p).log())

def boundary_entropy_loss(
    probs,
    mask = None,
    target_avg_token_length = 6.,
    chunk_size = None,
    exclude_first = True
):
    if exclude_first and probs.shape[1] > 1:
        probs = probs[:, 1:]
        if exists(mask):
            mask = mask[:, 1:]

    length = probs.shape[1]

    if length == 0 or (exists(mask) and not mask.any()):
        return probs.sum() * 0.

    mask = default(mask, torch.ones_like(probs, dtype = torch.bool))

    win_size = default(chunk_size, max(2, round(2 * target_avg_token_length)))
    win_size = min(win_size, length)

    chunked_len = ceil(length / win_size) * win_size
    pad_len = chunked_len - length

    if pad_len > 0:
        probs = pad(probs, (0, pad_len), value = 0.)
        mask = pad(mask, (0, pad_len), value = False)

    to_chunks = partial(rearrange, pattern = 'b (k n) -> (b k) n', n = win_size)
    probs, mask = map(to_chunks, (probs, mask))

    valid_win = mask.sum(dim = -1) >= 2

    # subtract zero-valued (g - g.detach()) so gradients are zero-sum projected per window (zero gradient on window mean g)

    g = masked_mean(probs, mask = mask, dim = -1)
    probs_centered = probs - rearrange(g - g.detach(), '... -> ... 1')

    ent = masked_mean(binary_entropy(probs_centered), mask = mask, dim = -1)
    return masked_mean(ent, mask = valid_win)



