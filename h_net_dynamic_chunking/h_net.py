from __future__ import annotations

from collections import namedtuple

import torch
from torch import nn, tensor
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange

from torch_einops_utils import exclusive_cumsum, lens_to_mask, pad_left_at_dim, maybe

from h_net_dynamic_chunking.h_net_dynamic_chunking import DynamicSequenceChunker
from h_net_dynamic_chunking.multi_head_h_net_dynamic_chunking import MultiHeadDynamicSequenceChunker

from vector_quantize_pytorch import VectorQuantize

from x_transformers import Decoder

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_empty(t):
    return t.numel() == 0

def cast_tuple(t):
    return t if type(t) is tuple else (t,)

def pick(d, *keys):
    d = default(d, dict())
    return tuple(d.get(k) for k in keys)

def calc_absolute_chunk_lens(
    outer_absolute_lens, # int[b n]
    inner_chunk_lens     # int[b m]
):
    outer_cumsum = outer_absolute_lens.cumsum(dim = -1)

    padded_outer_cumsum = pad_left_at_dim(outer_cumsum, 1, value = 0)

    max_inner_cumsum = padded_outer_cumsum.shape[-1] - 1

    inner_cumsum = inner_chunk_lens.cumsum(dim = -1)
    inner_cumsum = inner_cumsum.clamp(max = max_inner_cumsum)

    absolute_cumsum = padded_outer_cumsum.gather(1, inner_cumsum)

    padded_absolute_cumsum = pad_left_at_dim(absolute_cumsum, 1, value = 0)

    return padded_absolute_cumsum[:, 1:] - padded_absolute_cumsum[:, :-1]

# cache shifting
# right-align kv cache for valid entries using stable argsort on boolean mask

@maybe
def shift_mask_right(mask):
    indices = mask.long().argsort(dim = -1, stable = True)
    return mask.gather(1, indices)

def shift_cache_right(cache, mask):
    if not exists(mask) or not exists(cache):
        return cache

    indices = mask.long().argsort(dim = -1, stable = True)

    if isinstance(cache, dict):
        for k in ('encoder', 'decoder'):
            if k in cache:
                cache[k] = shift_cache_right(cache[k], mask)

        if exists(cache.get('input_mask')):
            cache['input_mask'] = cache['input_mask'].gather(1, indices)

        return cache

    if not hasattr(cache, 'attn_intermediates') or not exists(cache.attn_intermediates):
        return cache

    for attn_inter in cache.attn_intermediates:
        if not exists(attn_inter.cached_kv):
            continue

        k, v = attn_inter.cached_kv
        indices_expanded = rearrange(indices, 'b n -> b 1 n 1').expand_as(k)

        attn_inter.cached_kv = (
            k.gather(2, indices_expanded),
            v.gather(2, indices_expanded)
        )

    return cache

# classes

HNetReturn = namedtuple('HNetReturn', [
    'output',
    'loss',
    'intermediates',
    'next_cache'
])

class HNet(Module):
    def __init__(
        self,
        encoder: Module | dict,
        network: Module | HNet | dict,
        decoder: Module | dict,
        dim,
        dim_inner = None,
        vq: VectorQuantize | dict | None = None,
        inner_network_rel_pos_kwarg: str | None = None,
        **dynamic_sequence_chunking_kwargs
    ):
        super().__init__()

        def instantiate(m):
            default_decoder_kwarg = dict(prenorm_has_final_norm = False)

            if isinstance(m, dict):
                m = {**default_decoder_kwarg, **m}
                return Decoder(**m)

            return m

        self.encoder = instantiate(encoder)
        self.network = instantiate(network)
        self.decoder = instantiate(decoder)

        self.inner_network_rel_pos_kwarg = inner_network_rel_pos_kwarg

        heads = dynamic_sequence_chunking_kwargs.pop('heads', 1)
        chunker_klass = DynamicSequenceChunker if heads == 1 else MultiHeadDynamicSequenceChunker

        if heads > 1:
            dynamic_sequence_chunking_kwargs['heads'] = heads

        self.dynamic_sequence_chunker = chunker_klass(
            dim = dim,
            handle_residual_proj = True,
            **dynamic_sequence_chunking_kwargs
        )

        # maybe dimension expansion

        dim_inner = default(dim_inner, dim)
        need_proj = dim != dim_inner

        self.proj_in = nn.Linear(dim, dim_inner) if need_proj else nn.Identity()
        self.proj_out = nn.Linear(dim_inner, dim) if need_proj else nn.Identity()

        # maybe vector quantization

        if isinstance(vq, dict):
            vq = VectorQuantize(**vq)

        assert not exists(vq) or isinstance(vq, VectorQuantize)
        self.vq = vq

        self.register_buffer('zero', tensor(0.), persistent = False)

        self._is_nested_hnet = isinstance(self.network, HNet)
        self._is_multi_head = heads > 1

    @property
    def device(self):
        return self.zero.device

    def forward(
        self,
        tokens,
        lens = None,
        return_intermediates = False,
        return_hiddens = False,
        cache = None,
        vq_mask = None
    ):
        is_caching = exists(cache) or return_hiddens

        assert not (self._is_multi_head and is_caching), 'caching is not yet supported for multi-head dynamic sequence chunking'

        # unpack cache

        next_cache = dict()

        chunker_cache, upsample_cache, encoder_cache, decoder_cache, inner_cache, inner_mask_cache, input_mask_cache = pick(cache, 'chunker', 'upsample', 'encoder', 'decoder', 'inner', 'inner_mask', 'input_mask')

        # input mask for current step

        if exists(lens):
            curr_input_mask = lens_to_mask(lens, max_len = tokens.shape[1])
        else:
            curr_input_mask = torch.ones(tokens.shape[:2], device = tokens.device, dtype = torch.bool)

        # accumulate mask cache and derive positional info

        if is_caching:
            chunker_cache = default(chunker_cache, dict())
            upsample_cache = default(upsample_cache, dict())

            if exists(input_mask_cache):
                enc_seq_start_pos = (~input_mask_cache).sum(dim = -1)
                input_mask_cache = torch.cat((input_mask_cache, curr_input_mask), dim = -1)
            else:
                enc_seq_start_pos = torch.zeros(tokens.shape[0], device = tokens.device, dtype = torch.long)
                input_mask_cache = curr_input_mask

        # encode

        if is_caching and isinstance(self.encoder, Decoder):
            encoded, encoder_cache = self.encoder(
                tokens,
                mask = curr_input_mask,
                cache = encoder_cache,
                return_hiddens = True,
                self_attn_kv_mask = input_mask_cache,
                seq_start_pos = enc_seq_start_pos
            )

            next_cache['encoder'] = encoder_cache

        elif isinstance(self.encoder, Decoder):
            encoded = self.encoder(tokens, mask = curr_input_mask)
        else:
            encoded = self.encoder(tokens)

        # maybe quantize before downsampling

        maybe_commit_loss = self.zero

        if exists(self.vq):
            encoded, indices, maybe_commit_loss = self.vq(encoded, mask = vq_mask)

        # downsample via dynamic chunker

        (downsampled, upsample, aux_ratio_loss), intermediate = self.dynamic_sequence_chunker(
            encoded,
            lens = lens,
            return_intermediates = True,
            cache = chunker_cache
        )

        maybe_projected_downsampled = self.proj_in(downsampled)

        if exists(self.vq):
            batch = maybe_projected_downsampled.shape[0]
            has_chunks = not is_empty(maybe_projected_downsampled)

            quantized_downsampled_indices = torch.empty((batch, 0), dtype = torch.long, device = self.device)

            if has_chunks:
                num_chunks = intermediate.boundary_mask.long().sum(dim = -1)
                boundary_indices = indices[intermediate.boundary_mask]
                quantized_downsampled_indices = pad_sequence(boundary_indices.split(num_chunks.tolist()), batch_first = True, padding_value = -1)

            intermediate = intermediate._replace(
                quantized_downsampled_indices = quantized_downsampled_indices
            )

        # maybe pass boundary positions to inner network
        # inspired by HealthFormer (https://www.medrxiv.org/content/10.64898/2026.03.25.26349262v1)

        inner_network_extra_kwargs = dict()

        if exists(self.inner_network_rel_pos_kwarg):
            boundary_positions = exclusive_cumsum(intermediate.chunk_lens)
            inner_network_extra_kwargs[self.inner_network_rel_pos_kwarg] = boundary_positions

        # inner network

        has_tokens_for_inner = not is_empty(maybe_projected_downsampled)

        inner_intermediates = ()
        maybe_inner_aux_ratio_loss = self.zero

        if not has_tokens_for_inner:
            inner_network_output = maybe_projected_downsampled

        else:
            curr_inner_mask = intermediate.chunk_lens > 0

            if is_caching:
                if exists(inner_mask_cache):
                    inner_seq_start_pos = (~inner_mask_cache).sum(dim = -1)
                    inner_mask_cache = torch.cat((inner_mask_cache, curr_inner_mask), dim = -1)
                else:
                    inner_seq_start_pos = torch.zeros(maybe_projected_downsampled.shape[0], device = tokens.device, dtype = torch.long)
                    inner_mask_cache = curr_inner_mask

            if self._is_nested_hnet:
                inner_lens = curr_inner_mask.sum(dim = -1)
                out = self.network(maybe_projected_downsampled, lens = inner_lens, cache = inner_cache, return_hiddens = is_caching, return_intermediates = True)

                inner_network_output = out.output
                maybe_inner_aux_ratio_loss = out.loss
                inner_cache = out.next_cache
                inner_intermediates = out.intermediates

            elif is_caching and isinstance(self.network, Decoder):

                inner_network_output, inner_cache = self.network(
                    maybe_projected_downsampled,
                    mask = curr_inner_mask,
                    cache = inner_cache,
                    return_hiddens = True,
                    self_attn_kv_mask = inner_mask_cache,
                    seq_start_pos = inner_seq_start_pos,
                    input_not_include_cache = True,
                    **inner_network_extra_kwargs
                )

            elif isinstance(self.network, Decoder):
                inner_network_output = self.network(maybe_projected_downsampled, mask = curr_inner_mask, **inner_network_extra_kwargs)

            else:
                inner_network_output = self.network(maybe_projected_downsampled, **inner_network_extra_kwargs)

        if return_hiddens:
            next_cache['inner'] = inner_cache
            next_cache['inner_mask'] = inner_mask_cache

        upsampled = upsample(self.proj_out(inner_network_output), cache = upsample_cache)

        extra_loss = self.zero

        if isinstance(upsampled, tuple):
            upsampled, extra_loss = upsampled

        # decode

        if is_caching and isinstance(self.decoder, Decoder):
            output, decoder_cache = self.decoder(
                upsampled,
                mask = curr_input_mask,
                cache = decoder_cache,
                return_hiddens = True,
                self_attn_kv_mask = input_mask_cache,
                seq_start_pos = enc_seq_start_pos
            )

            next_cache['decoder'] = decoder_cache

        elif isinstance(self.decoder, Decoder):
            output = self.decoder(upsampled, mask = curr_input_mask)
        else:
            output = self.decoder(upsampled)

        total_loss = aux_ratio_loss + maybe_inner_aux_ratio_loss + extra_loss + maybe_commit_loss

        # shift caches right-aligned for next step

        if return_hiddens:
            if exists(inner_mask_cache):
                next_cache['inner'] = shift_cache_right(next_cache.get('inner'), inner_mask_cache)

                inner_mask_cache = shift_mask_right(inner_mask_cache)
                next_cache['inner_mask'] = inner_mask_cache

            next_cache.update(
                chunker = chunker_cache,
                upsample = upsample_cache,
                input_mask = input_mask_cache
            )

        intermediate_out = intermediate

        if self._is_nested_hnet:
            new_inner_intermediates = []
            curr_absolute_lens = intermediate.absolute_chunk_lens

            for inner_intermediate in cast_tuple(inner_intermediates):
                new_abs_lens = calc_absolute_chunk_lens(curr_absolute_lens, inner_intermediate.chunk_lens)
                new_inner_intermediates.append(inner_intermediate._replace(absolute_chunk_lens = new_abs_lens))
                curr_absolute_lens = new_abs_lens

            intermediate_out = (intermediate_out, *new_inner_intermediates)

        return HNetReturn(
            output,
            total_loss,
            intermediate_out if return_intermediates else None,
            next_cache if return_hiddens else None
        )
