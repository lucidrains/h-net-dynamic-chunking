from __future__ import annotations

import torch
from torch import nn, tensor
from torch.nn import Module

from h_net_dynamic_chunking.h_net_dynamic_chunking import DynamicSequenceChunker
from h_net_dynamic_chunking.multi_head_h_net_dynamic_chunking import MultiHeadDynamicSequenceChunker

from vector_quantize_pytorch import VectorQuantize

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cast_tuple(t):
    return t if type(t) is tuple else (t,)

# classes

class HNet(Module):
    def __init__(
        self,
        encoder: Module,
        network: Module | HNet,
        decoder: Module,
        dim,
        dim_inner = None,
        vq: VectorQuantize | None = None,
        **dynamic_sequence_chunking_kwargs
    ):
        super().__init__()

        self.encoder = encoder
        self.network = network
        self.decoder = decoder

        heads = dynamic_sequence_chunking_kwargs.get('heads', 1)
        chunker_klass = DynamicSequenceChunker if heads == 1 else MultiHeadDynamicSequenceChunker

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

    def forward(
        self,
        tokens,
        return_intermediates = False
    ):

        encoded = self.encoder(tokens)

        (downsampled, upsample, aux_ratio_loss), intermediate = self.dynamic_sequence_chunker(encoded, return_intermediates = True)

        maybe_projected_downsampled = self.proj_in(downsampled)

        is_nested_hnet = isinstance(self.network, HNet)

        network_kwargs = dict(return_intermediates = True) if is_nested_hnet else dict()

        # maybe quantize

        if exists(self.vq):
            maybe_projected_downsampled, indices, commit_loss = self.vq(maybe_projected_downsampled)

        # the inner transformer working on temporal compressed selected boundary tokens

        inner_hierarchy_out = self.network(maybe_projected_downsampled, **network_kwargs)

        # maybe multiple hierarchies

        if is_nested_hnet:
            inner_network_output, maybe_inner_aux_ratio_loss, inner_intermediates = inner_hierarchy_out
        else:
            inner_network_output = inner_hierarchy_out

            inner_intermediates = ()
            maybe_inner_aux_ratio_loss = self.zero

        maybe_projected_inner_network_output = self.proj_out(inner_network_output)

        upsampled = upsample(maybe_projected_inner_network_output)

        extra_loss = self.zero

        if isinstance(upsampled, tuple):
            upsampled, extra_loss = upsampled

        output = self.decoder(upsampled)

        total_loss = aux_ratio_loss + maybe_inner_aux_ratio_loss + extra_loss

        output_with_loss = (output, total_loss)

        if not return_intermediates:
            return output_with_loss

        # take care of adding the quantized indices

        if exists(self.vq):
            intermediate = intermediate._replace(quantized_downsampled_indices = indices)

        intermediate_out = intermediate

        if is_nested_hnet:
            intermediate_out = (intermediate_out, *cast_tuple(inner_intermediates))

        return (*output_with_loss, intermediate_out)
