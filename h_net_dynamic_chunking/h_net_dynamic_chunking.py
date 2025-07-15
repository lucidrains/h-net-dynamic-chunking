# following section 2.2 of the paper

import torch
from torch import cat
from torch.nn import Module, Linear, Parameter
from torch.nn.functional import cosine_similarity

from einops import repeat, rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class CosineSimRouting(Module):
    def __init__(
        self,
        dim,
        dim_queries_keys = None,
        boundary_threshold = 0.5,
        target_ratio = 1. / 6.,        # 1/N in eq(10)
        ratio_loss_weight = 3e-2
    ):
        super().__init__()
        dim_queries_keys = default(dim_queries_keys, dim)

        # linear to queries and keys

        self.to_queries_keys = Linear(dim, dim_queries_keys * 2, bias = False)

        # start key token, so first token can be segmented / chunked out

        self.start_key_token = Parameter(torch.randn(dim_queries_keys) * 1e-2) # presumably, need a start key token for the first token, open an issue if i got it wrong

        # threshold to determine boundary

        self.boundary_threshold = boundary_threshold

        # ratio aux loss related

        self.target_avg_token_length = 1. / target_ratio
        self.ratio_loss_weight = ratio_loss_weight

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    def forward(
        self,
        tokens # float[b n d]
    ):
        batch, length = tokens.shape[:2]

        queries, keys = self.to_queries_keys(tokens).chunk(2, dim = -1)

        start_keys = repeat(self.start_key_token, 'd -> b 1 d', b = batch)

        keys = cat((start_keys, keys), dim = 1)

        # each query looks at the previous key to determine if distance is greater than some threshold for determining a boundary exists (they use 0.5 as threshold)

        cosine_sim  = cosine_similarity(queries, keys[:, :-1], dim = -1)

        prob_boundary = (1. - cosine_sim) * 0.5 # cosine sim is -1. to 1., this transforms it to 0. to 1.

        boundaries = prob_boundary > self.boundary_threshold # bool[b n]

        # for the upsampler

        confidence = torch.where(boundaries, prob_boundary, 1. - prob_boundary)

        # defaults if not training

        upsampler_output_scale = 1.
        aux_ratio_loss = self.zero

        if self.training or tokens.requires_grad:
            # straight through for 1. multiplier on the expanded processed boundary tokens

            upsampler_output_scale = confidence * (1. - confidence).detach()

            # auxiliary ratio loss in section 2.3.2, eq (10)
            # lets follow their notation

            N = self.target_avg_token_length

            F = prob_boundary.sum(dim = -1) / length
            G = boundaries.sum(dim = -1) / length

            aux_ratio_loss = N / (N - 1) * ((N - 1) * F * G + (1. - F) * (1. - G))

            aux_loss = aux_ratio_loss.mean() * self.ratio_loss_weight

        return prob_boundary, boundaries, upsampler_output_scale, aux_loss
