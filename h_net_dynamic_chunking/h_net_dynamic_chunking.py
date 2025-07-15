# following section 2.2 of the paper

import torch
from torch import cat
from torch.nn import Module, Linear, Parameter
import torch.nn.functional as F

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
        boundary_threshold = 0.5
    ):
        super().__init__()
        dim_queries_keys = default(dim_queries_keys, dim)

        # linear to queries and keys

        self.to_queries_keys = Linear(dim, dim_queries_keys * 2, bias = False)

        # start key token, so first token can be segmented / chunked out

        self.start_key_token = Parameter(torch.randn(dim_queries_keys) * 1e-2) # presumably, need a start key token for the first token, open an issue if i got it wrong

        # threshold to determine boundary

        self.boundary_threshold = boundary_threshold

    def forward(
        self,
        tokens # float[b n d]
    ):
        batch = tokens.shape[0]

        queries, keys = self.to_queries_keys(tokens).chunk(2, dim = -1)

        start_keys = repeat(self.start_key_token, 'd -> b 1 d', b = batch)

        keys = cat((start_keys, keys), dim = 1)

        # each query looks at the previous key to determine if distance is greater than some threshold for determining a boundary exists (they use 0.5 as threshold)

        cosine_sim  = F.cosine_similarity(queries, keys[:, :-1], dim = -1)

        prob_boundary = (1. - cosine_sim) * 0.5 # cosine sim is -1. to 1., this transforms it to 0. to 1.

        boundaries = prob_boundary > self.boundary_threshold # bool[b n]

        # for the upsampler

        confidence = torch.where(boundaries, prob_boundary, 1. - prob_boundary)

        # straight through for 1. multiplier on the expanded processed boundary tokens

        upsampler_output_scale = confidence * (1. - confidence).detach()

        return prob_boundary, boundaries, upsampler_output_scale
