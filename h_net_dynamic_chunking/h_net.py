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
        **dynamic_sequence_chunking_kwargs
    ):
        super().__init__()

        self.encoder = encoder
        self.network = network
        self.decoder = decoder

        self.dynamic_sequence_chunker = DynamicSequenceChunker(**dynamic_sequence_chunking_kwargs)
        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        tokens
    ):

        encoded = self.encoder(tokens)

        downsampled, upsample, aux_ratio_loss = self.dynamic_sequence_chunker(encoded)

        inner_hierarchy_out = self.network(downsampled)

        if isinstance(self.network, HNet):
            downsampled, maybe_inner_aux_ratio_loss = inner_hierarchy_out
        else:
            downsampled = inner_hierarchy_out
            maybe_inner_aux_ratio_loss = self.zero

        upsampled = upsample(downsampled)

        output = self.decoder(upsampled)

        return output, aux_ratio_loss + maybe_inner_aux_ratio_loss
