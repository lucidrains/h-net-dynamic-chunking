from __future__ import annotations

from collections import namedtuple

import inspect

import torch
from torch import nn, tensor
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence

from torch_einops_utils import exclusive_cumsum, pad_left_at_dim

from h_net_dynamic_chunking.h_net_dynamic_chunking import DynamicSequenceChunker
from h_net_dynamic_chunking.multi_head_h_net_dynamic_chunking import MultiHeadDynamicSequenceChunker

from vector_quantize_pytorch import VectorQuantize

from x_transformers import Decoder
from x_transformers.x_transformers import AttentionLayers

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

# check if a module's forward accepts cache and return_hiddens kwargs

def accepts_cache(module):
    if isinstance(module, (Decoder, AttentionLayers)):
        return True
    sig = inspect.signature(module.forward).parameters
    return 'cache' in sig and ('return_hiddens' in sig or 'kwargs' in sig)

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

        # convenience, if hierarchical layer should have dimension expansion
        # would make sense

        dim_inner = default(dim_inner, dim)
        need_proj = dim != dim_inner

        self.proj_in = nn.Linear(dim, dim_inner) if need_proj else nn.Identity()
        self.proj_out = nn.Linear(dim_inner, dim) if need_proj else nn.Identity()

        # maybe do vector quantization
        # just use own library

        if isinstance(vq, dict):
            vq = VectorQuantize(**vq)

        assert not exists(vq) or isinstance(vq, VectorQuantize)
        self.vq = vq

        self.register_buffer('zero', tensor(0.), persistent = False)

        # determine cache support for encoder, decoder, and inner network at init time

        is_nested_hnet = isinstance(self.network, HNet)
        self._is_nested_hnet = is_nested_hnet

        self._encoder_accepts_cache = accepts_cache(self.encoder)
        self._decoder_accepts_cache = accepts_cache(self.decoder)
        self._inner_accepts_cache = is_nested_hnet or accepts_cache(self.network)
        self._is_multi_head = heads > 1

        # inner cache kwargs

        inner_sig = inspect.signature(self.network.forward).parameters

        self._inner_cache_kwargs = dict(return_hiddens = True)

        if 'input_not_include_cache' in inner_sig or 'kwargs' in inner_sig:
            self._inner_cache_kwargs['input_not_include_cache'] = True

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

        # unpack cache, ensuring we don't mutate the user's input cache

        next_cache = dict()

        chunker_cache, upsample_cache, encoder_cache, decoder_cache, inner_cache = pick(cache, 'chunker', 'upsample', 'encoder', 'decoder', 'inner')

        if is_caching:
            chunker_cache = default(chunker_cache, dict())
            upsample_cache = default(upsample_cache, dict())

        # encode

        if self._encoder_accepts_cache and is_caching:
            encoded, encoder_cache = self.encoder(tokens, cache = encoder_cache, return_hiddens = True)
            next_cache['encoder'] = encoder_cache
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

        network_kwargs = dict()

        # maybe pass boundary positions to inner network
        # inspired by HealthFormer (https://www.medrxiv.org/content/10.64898/2026.03.25.26349262v1)

        if exists(self.inner_network_rel_pos_kwarg):
            boundary_positions = exclusive_cumsum(intermediate.chunk_lens)
            network_kwargs[self.inner_network_rel_pos_kwarg] = boundary_positions

        # inner network - skip if no boundaries this step (cache handles output)

        has_tokens_for_inner = not is_empty(maybe_projected_downsampled)

        inner_intermediates = ()
        maybe_inner_aux_ratio_loss = self.zero

        if not has_tokens_for_inner:
            inner_network_output = maybe_projected_downsampled

        elif self._is_nested_hnet:
            inner_lens = (intermediate.chunk_lens > 0).sum(dim=-1)
            out = self.network(maybe_projected_downsampled, lens = inner_lens, cache = inner_cache, return_hiddens = is_caching, return_intermediates = True)

            inner_network_output = out.output
            maybe_inner_aux_ratio_loss = out.loss
            inner_cache = out.next_cache
            inner_intermediates = out.intermediates

        else:
            if self._inner_accepts_cache and is_caching:
                inner_kwargs = {**self._inner_cache_kwargs, **network_kwargs}
                inner_network_output, inner_cache = self.network(maybe_projected_downsampled, cache = inner_cache, **inner_kwargs)
            else:
                inner_network_output = self.network(maybe_projected_downsampled, **network_kwargs)

        if return_hiddens:
            next_cache['inner'] = inner_cache

        upsampled = upsample(self.proj_out(inner_network_output), cache = upsample_cache)

        extra_loss = self.zero

        if isinstance(upsampled, tuple):
            upsampled, extra_loss = upsampled

        # decode

        if self._decoder_accepts_cache and is_caching:
            output, decoder_cache = self.decoder(upsampled, cache = decoder_cache, return_hiddens = True)
            next_cache['decoder'] = decoder_cache
        else:
            output = self.decoder(upsampled)

        total_loss = aux_ratio_loss + maybe_inner_aux_ratio_loss + extra_loss + maybe_commit_loss

        if return_hiddens:
            next_cache.update(
                chunker = chunker_cache,
                upsample = upsample_cache
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
