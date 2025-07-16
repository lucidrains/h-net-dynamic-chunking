from __future__ import annotations

import torch
from torch import nn, tensor
from torch.nn import Module

from h_net_dynamic_chunking.h_net_dynamic_chunking import DynamicSequenceChunker

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class HNet(Module):
    def __init__(
        self,
        encoder: Module,
        network: Module | HNet,
        decoder: Module,
        dim,
        dim_inner = None,
        **dynamic_sequence_chunking_kwargs
    ):
        super().__init__()

        self.encoder = encoder
        self.network = network
        self.decoder = decoder

        self.dynamic_sequence_chunker = DynamicSequenceChunker(
            dim = dim,
            handle_residual_proj = True,
            **dynamic_sequence_chunking_kwargs
        )

        dim_inner = default(dim_inner, dim)
        need_proj = dim != dim_inner

        self.proj_in = nn.Linear(dim, dim_inner) if need_proj else nn.Identity()
        self.proj_out = nn.Linear(dim_inner, dim) if need_proj else nn.Identity()

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        tokens
    ):

        encoded = self.encoder(tokens)

        downsampled, upsample, aux_ratio_loss = self.dynamic_sequence_chunker(encoded)

        downsampled = self.proj_in(downsampled)

        inner_hierarchy_out = self.network(downsampled)

        if isinstance(self.network, HNet):
            inner_network_output, maybe_inner_aux_ratio_loss = inner_hierarchy_out
        else:
            inner_network_output = inner_hierarchy_out
            maybe_inner_aux_ratio_loss = self.zero

        inner_network_output = self.proj_out(inner_network_output)

        upsampled = upsample(inner_network_output)

        output = self.decoder(upsampled)

        return output, aux_ratio_loss + maybe_inner_aux_ratio_loss
