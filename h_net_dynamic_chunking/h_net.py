from __future__ import annotations

from collections import namedtuple

import inspect

from torch import nn, tensor
from torch.nn import Module

from torch_einops_utils import exclusive_cumsum

from h_net_dynamic_chunking.h_net_dynamic_chunking import DynamicSequenceChunker
from h_net_dynamic_chunking.multi_head_h_net_dynamic_chunking import MultiHeadDynamicSequenceChunker

from vector_quantize_pytorch import VectorQuantize
from x_transformers import Decoder

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cast_tuple(t):
    return t if type(t) is tuple else (t,)

def pick(d, *keys):
    d = default(d, dict())
    return tuple(d.get(k) for k in keys)

# check if a module's forward accepts cache and return_hiddens kwargs

def accepts_cache(module):
    if isinstance(module, Decoder):
        return True
    sig = inspect.signature(module.forward).parameters
    return 'cache' in sig and 'return_hiddens' in sig

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
        vq: VectorQuantize | None = None,
        inner_network_rel_pos_kwarg: str | None = None,
        **dynamic_sequence_chunking_kwargs
    ):
        super().__init__()

        def instantiate(m):
            return Decoder(**m) if isinstance(m, dict) else m

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

    def forward(
        self,
        tokens,
        return_intermediates = False,
        return_hiddens = False,
        cache = None
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

        # downsample via dynamic chunker

        (downsampled, upsample, aux_ratio_loss), intermediate = self.dynamic_sequence_chunker(
            encoded,
            return_intermediates = True,
            cache = chunker_cache
        )

        maybe_projected_downsampled = self.proj_in(downsampled)

        network_kwargs = dict()

        # maybe pass boundary positions to inner network
        # inspired by HealthFormer (https://www.medrxiv.org/content/10.64898/2026.03.25.26349262v1)

        if exists(self.inner_network_rel_pos_kwarg):
            boundary_positions = exclusive_cumsum(intermediate.chunk_lens)
            network_kwargs[self.inner_network_rel_pos_kwarg] = boundary_positions

        # maybe quantize

        maybe_commit_loss = self.zero

        if exists(self.vq):
            maybe_projected_downsampled, indices, maybe_commit_loss = self.vq(maybe_projected_downsampled)

        # inner network - skip if no boundaries this step (cache handles output)

        has_tokens_for_inner = maybe_projected_downsampled.shape[1] > 0

        inner_intermediates = ()
        maybe_inner_aux_ratio_loss = self.zero

        if not has_tokens_for_inner:
            inner_network_output = maybe_projected_downsampled

        elif self._is_nested_hnet:
            out = self.network(maybe_projected_downsampled, cache = inner_cache, return_hiddens = is_caching, return_intermediates = True)

            inner_network_output = out.output
            maybe_inner_aux_ratio_loss = out.loss
            inner_cache = out.next_cache
            inner_intermediates = out.intermediates

        else:
            if self._inner_accepts_cache and is_caching:
                inner_network_output, inner_cache = self.network(maybe_projected_downsampled, cache = inner_cache, return_hiddens = True, **network_kwargs)
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

        if exists(self.vq):
            intermediate = intermediate._replace(quantized_downsampled_indices = indices)

        intermediate_out = intermediate

        if self._is_nested_hnet:
            intermediate_out = (intermediate_out, *cast_tuple(inner_intermediates))

        return HNetReturn(
            output,
            total_loss,
            intermediate_out if return_intermediates else None,
            next_cache if return_hiddens else None
        )
