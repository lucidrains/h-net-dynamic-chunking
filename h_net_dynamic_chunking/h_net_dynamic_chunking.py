# following section 2.2 of the paper

from collections import namedtuple

import torch
from torch import cat, arange
from torch.nested import nested_tensor
from torch.nn import Module, Linear, Parameter
from torch.nn.functional import cosine_similarity, pad

from einx import multiply
from einops import repeat, rearrange

from assoc_scan import AssocScan

# constants

Outputs = namedtuple('Outputs', [
    'downsampled',
    'upsample_fn',
    'weighted_aux_ratio_loss'
])

Intermediates = namedtuple('Intermediates', [
    'mask',
    'probs',
    'chunk_lens',
    'boundary_mask',
    'residual',
    'upsampler_output_scale',
    'aux_ratio_loss'
])

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def straight_through(t, value):
    return t + (value - t).detach()

def frac_gradient(t, frac = 1.):
    if frac == 1:
        return

    t_grad = t * frac
    return straight_through(t_grad, t)

# classes

class DynamicSequenceChunker(Module):
    def __init__(
        self,
        dim,
        dim_queries_keys = None,
        boundary_threshold = 0.5,
        target_avg_token_length = 6.,       # N in eq(10)
        ratio_loss_weight = 3e-2,
        handle_residual_proj = False,       # turning this on will automatically handle a projection of the residual and its application in the inverse upsample function
        assoc_scan_use_accelerated = False,
        learning_rate_difference = 0.75,    # in the paper, they report that as one moves up a hierarchy, the learning rate needs to decrease. we'll default to 0.75 for the rough 2.0 -> 1.5 somewhere in the appendix from level 0 -> 1
        straight_through_frac_vecs = True,  # improvisation where F receives gradients through straight-through with sigmoid
    ):
        super().__init__()
        dim_queries_keys = default(dim_queries_keys, dim)

        # linear to queries and keys

        self.to_queries_keys = Linear(dim, dim_queries_keys * 2, bias = False)

        # start key token, so first token can be segmented / chunked out

        self.start_key_token = Parameter(torch.randn(dim_queries_keys) * 1e-2) # presumably, need a start key token for the first token, open an issue if i got it wrong

        # threshold to determine boundary

        assert 0. < boundary_threshold < 1.

        self.boundary_threshold = boundary_threshold

        # smoothing related

        self.smooth_assoc_scan = AssocScan(use_accelerated = assoc_scan_use_accelerated)

        # maybe residual proj

        self.handle_residual_proj = handle_residual_proj

        if handle_residual_proj:
            self.residual_proj = Linear(dim, dim)

        # learning rate modulation, appendix C
        # the multiplier on the learning rate as one goes from outer to inner of the h-net, and inverse of this value from inner to outer

        self.learning_rate_difference = learning_rate_difference

        # ratio aux loss related

        self.target_avg_token_length = target_avg_token_length

        self.straight_through_frac_vecs = straight_through_frac_vecs

        self.ratio_loss_weight = ratio_loss_weight

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    def upsample(
        self,
        downsampled,
        intermediates: Intermediates,
        apply_scale = True
    ):
        batch, needs_grad, device = downsampled.shape[0], downsampled.requires_grad, downsampled.device

        mask = intermediates.mask
        residual = intermediates.residual

        downsampled_without_padding = downsampled[mask]
        chunk_lens_without_padding = intermediates.chunk_lens[mask]

        seq = arange(downsampled_without_padding.shape[0], device = device)

        repeated_indices = torch.repeat_interleave(seq, chunk_lens_without_padding, dim = 0)
        upsampled = downsampled_without_padding[repeated_indices]

        upsampled = rearrange(upsampled, '(b n) d -> b n d', b = batch)

        scale = intermediates.upsampler_output_scale

        if needs_grad and apply_scale and exists(scale):
            upsampled = multiply('b n d, b n', upsampled, scale)

        if self.handle_residual_proj:
            upsampled = upsampled + self.residual_proj(residual)

        upsampled = frac_gradient(upsampled, self.learning_rate_difference)

        return upsampled

    def forward(
        self,
        tokens, # float[b n d],
        return_intermediates = False
    ):
        batch, length, device = *tokens.shape[:2], tokens.device

        residual = tokens

        queries, keys = self.to_queries_keys(tokens).chunk(2, dim = -1)

        start_keys = repeat(self.start_key_token, 'd -> b 1 d', b = batch)

        keys = cat((start_keys, keys), dim = 1)

        # each query looks at the previous key to determine if distance is greater than some threshold for determining a boundary exists (they use 0.5 as threshold)

        cosine_sim  = cosine_similarity(queries, keys[:, :-1], dim = -1)

        probs = (1. - cosine_sim) * 0.5 # cosine sim is -1. to 1., this transforms it to 0. to 1.

        boundary_mask = probs > self.boundary_threshold # bool[b n]

        boundary_mask[:, 0] = True # first token must always be boundary

        # compute some lengths, per chunk and number of chunks per batch

        num_chunks = boundary_mask.long().sum(dim = -1)

        boundary_mask_with_end = pad(boundary_mask, (0, 1), value = True)
        sel_indices = repeat(arange(boundary_mask_with_end.shape[-1], device = device), 'n -> b n', b = batch)[boundary_mask_with_end]

        sel_indices = nested_tensor(sel_indices.split((num_chunks + 1).tolist()), layout = torch.jagged, device = device)

        sel_indices = sel_indices.to_padded_tensor(padding = -1)

        mask = (sel_indices != -1)[:, 1:]

        chunk_lens = sel_indices[:, 1:] - sel_indices[:, :-1]
        chunk_lens.masked_fill_(~mask, 0)

        # downsampling - they show in their experiments that picking out the boundary tokens works just fine

        boundary_tokens = tokens[boundary_mask] # pick out boundary tokens

        tokens_nt = nested_tensor(boundary_tokens.split(num_chunks.tolist()), layout = torch.jagged, device = device, requires_grad = True)

        downsampled_tokens = tokens_nt.to_padded_tensor(padding = 0.)

        # smoothing module for improved gradients eq(5)

        probs_nt = nested_tensor(probs[boundary_mask].split(num_chunks.tolist()), layout = torch.jagged, device = device, requires_grad = True)

        boundary_probs = probs_nt.to_padded_tensor(padding = 0.)

        gates = 1. - boundary_probs

        downsampled_tokens = multiply('b n d, b n', downsampled_tokens, boundary_probs)

        smoothed_downsampled_tokens = self.smooth_assoc_scan(gates, downsampled_tokens)

        # for the upsampler

        confidence = torch.where(boundary_mask, probs, 1. - probs)

        # defaults if not training

        upsampler_output_scale = None
        aux_loss = self.zero
        weighted_aux_loss = self.zero

        needs_grad = tokens.requires_grad

        if needs_grad:
            # straight through for 1. multiplier on the expanded processed boundary tokens

            upsampler_output_scale = straight_through(confidence, 1.)

            # auxiliary ratio loss in section 2.3.2, eq (10)
            # lets follow their notation

            N = self.target_avg_token_length

            F = boundary_mask.float()
            G = probs.mean(dim = -1)

            # allow for a soft F to straight through - https://arxiv.org/abs/2505.22074

            if self.straight_through_frac_vecs:
                F_soft = (probs - self.boundary_threshold).sigmoid()
                F = straight_through(F_soft, F)

            F = F.mean(dim = -1)

            aux_ratio_loss = N / (N - 1) * ((N - 1) * F * G + (1. - F) * (1. - G))

            aux_loss = aux_ratio_loss.mean()
            weighted_aux_loss = aux_loss * self.ratio_loss_weight

        # intermediates

        intermediates = Intermediates(mask, probs, chunk_lens, boundary_mask, residual, upsampler_output_scale, aux_loss)

        # return the upsample function

        def upsample(downsampled, apply_scale = True):

            return self.upsample(downsampled, intermediates, apply_scale = apply_scale)

        # adjust learning rate

        smoothed_downsampled_tokens = frac_gradient(smoothed_downsampled_tokens, self.learning_rate_difference ** -1)

        # returning

        outputs = Outputs(smoothed_downsampled_tokens, upsample, weighted_aux_loss)

        if not return_intermediates:
            return outputs

        return outputs, intermediates
