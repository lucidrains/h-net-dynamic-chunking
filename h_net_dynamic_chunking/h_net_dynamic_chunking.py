# following section 2.2 of the paper

from collections import namedtuple

import torch
from torch import cat, arange, tensor, repeat_interleave
from torch.nn import Module, Linear, Parameter, Sequential
from torch.nn.functional import cosine_similarity, pad
from torch.nn.utils.rnn import pad_sequence

from einx import multiply, where
from einops import repeat, rearrange

from assoc_scan import AssocScan

from torch_einops_utils import lens_to_mask, pad_right_at_dim_to, masked_mean

from einops.layers.torch import Rearrange
from x_mlps_pytorch import create_mlp

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
    'gates',
    'residual',
    'upsampler_output_scale',
    'input_downsampled_tokens',
    'aux_ratio_loss',
    'quantized_downsampled_indices',
    'absolute_chunk_lens'
], defaults = (None, None))

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def straight_through(t, value):
    return t + (value - t).detach()

def frac_gradient(t, frac = 1.):
    if frac == 1:
        return t

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
        embed_chunk_lens = False,
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

        self.embed_chunk_lens = embed_chunk_lens

        self.length_embed = None
        if embed_chunk_lens:
            self.length_embed = Sequential(
                Rearrange('... -> ... 1'),
                create_mlp(dim, depth = 2, dim_in = 1, dim_out = dim)
            )

        self.register_buffer('zero', tensor(0.), persistent = False)

    def upsample(
        self,
        downsampled,
        intermediates: Intermediates,
        apply_scale = True,
        cache = None
    ):
        batch, needs_grad, device = downsampled.shape[0], downsampled.requires_grad, downsampled.device

        mask = intermediates.mask
        gates = intermediates.gates
        residual = intermediates.residual

        # smoothing module for improved gradients eq(5)

        has_downsampled = downsampled.shape[1] > 0

        if has_downsampled:
            assoc_scan_prev = cache.pop('assoc_scan_prev', None) if exists(cache) else None

            downsampled = self.smooth_assoc_scan(gates, downsampled, prev = assoc_scan_prev)

            if exists(cache):
                cache['assoc_scan_prev'] = downsampled[:, -1]

        # upsample

        if exists(cache):
            seq_len = intermediates.boundary_mask.shape[1]
            last_upsampled = cache.pop('last_upsampled', None)

            if has_downsampled:
                chunk_idx = intermediates.boundary_mask.long().cumsum(dim = -1) - 1

                upsampled = downsampled[arange(batch, device = device)[:, None], chunk_idx.clamp(min = 0)]

                if exists(last_upsampled):
                    upsampled = where('b n, b d, b n d', chunk_idx < 0, last_upsampled, upsampled)

            else:
                assert exists(last_upsampled), 'first token must always be a boundary, so last_upsampled should exist'
                upsampled = repeat(last_upsampled, 'b d -> b n d', n = seq_len)

            cache['last_upsampled'] = upsampled[:, -1]

        else:
            upsampled_flat = repeat_interleave(downsampled[mask], intermediates.chunk_lens[mask], dim = 0)
            effective_lens = intermediates.chunk_lens.sum(dim = -1)
            upsampled = pad_sequence(upsampled_flat.split(effective_lens.tolist()), batch_first = True, padding_value = 0.)

            # pad to original sequence length if needed (variable lengths)

            upsampled = pad_right_at_dim_to(upsampled, residual.shape[1], dim = 1)

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
        lens = None,
        return_intermediates = False,
        return_only_chunk_lens = False,
        cache = None
    ):
        batch, length, device = *tokens.shape[:2], tokens.device

        residual = tokens

        queries, keys = self.to_queries_keys(tokens).chunk(2, dim = -1)

        is_first_step = not exists(cache) or 'prev_key' not in cache

        if is_first_step:
            keys_to_prepend = repeat(self.start_key_token, 'd -> b 1 d', b = batch)
        else:
            keys_to_prepend = cache.pop('prev_key')

        keys_for_sim = cat((keys_to_prepend, keys[:, :-1]), dim = 1)

        if exists(cache):
            cache['prev_key'] = keys[:, -1:]

        # each query looks at the previous key to determine if distance is greater than some threshold for determining a boundary exists (they use 0.5 as threshold)

        cosine_sim  = cosine_similarity(queries, keys_for_sim, dim = -1)

        probs = (1. - cosine_sim) * 0.5 # cosine sim is -1. to 1., this transforms it to 0. to 1.

        boundary_mask = probs > self.boundary_threshold # bool[b n]

        if is_first_step:
            boundary_mask[:, 0] = True # first token must always be boundary

        # mask out positions beyond actual sequence lengths

        if exists(lens):
            seq_mask = lens_to_mask(lens, max_len = length)  # bool[b n]
            boundary_mask = boundary_mask & seq_mask

        # compute some lengths, per chunk and number of chunks per batch

        num_chunks = boundary_mask.long().sum(dim = -1)

        # place end sentinel at actual sequence length (lens) rather than padded length

        if exists(lens):
            boundary_mask_with_end = pad(boundary_mask, (0, 1), value = False)
            boundary_mask_with_end.scatter_(1, lens[:, None].long(), True)
        else:
            boundary_mask_with_end = pad(boundary_mask, (0, 1), value = True)

        sel_indices = boundary_mask_with_end.nonzero()[:, 1]

        sel_indices = pad_sequence(sel_indices.split((num_chunks + 1).tolist()), batch_first=True, padding_value=-1)

        mask = (sel_indices != -1)[:, 1:]

        chunk_lens = sel_indices[:, 1:] - sel_indices[:, :-1]
        chunk_lens.masked_fill_(~mask, 0)

        # early return chunk lens if using a trained module as a tokenizer

        if return_only_chunk_lens:
            return chunk_lens

        # downsampling - they show in their experiments that picking out the boundary tokens works just fine

        max_chunks = num_chunks.amax().item()

        if max_chunks == 0:
            downsampled_tokens = tokens[:, 0:0]
            gates = probs[:, 0:0]

        else:
            boundary_tokens = tokens[boundary_mask] # pick out boundary tokens

            downsampled_tokens = pad_sequence(boundary_tokens.split(num_chunks.tolist()), batch_first=True, padding_value=0.)

            # smoothing module for improved gradients eq(5)

            boundary_probs = pad_sequence(probs[boundary_mask].split(num_chunks.tolist()), batch_first=True, padding_value=0.)

            gates = 1. - boundary_probs

            downsampled_tokens = multiply('b n d, b n', downsampled_tokens, boundary_probs)

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

            mask_for_mean = seq_mask if exists(lens) else None

            G = masked_mean(probs, mask = mask_for_mean, dim = -1)

            # allow for a soft F to straight through - https://arxiv.org/abs/2505.22074

            if self.straight_through_frac_vecs:
                F_soft = (probs - self.boundary_threshold).sigmoid()
                F = straight_through(F_soft, F)

            F = masked_mean(F, mask = mask_for_mean, dim = -1)

            aux_ratio_loss = N / (N - 1) * ((N - 1) * F * G + (1. - F) * (1. - G))

            aux_loss = aux_ratio_loss.mean()
            weighted_aux_loss = aux_loss * self.ratio_loss_weight

        # intermediates

        intermediates = Intermediates(mask, probs, chunk_lens, boundary_mask, gates, residual, upsampler_output_scale, downsampled_tokens, aux_loss, None, chunk_lens)

        # return the upsample function

        def upsample(downsampled, apply_scale = True, cache = None):

            return self.upsample(downsampled, intermediates, apply_scale = apply_scale, cache = cache)

        # adjust learning rate

        downsampled_tokens = frac_gradient(downsampled_tokens, self.learning_rate_difference ** -1)

        # length embed

        if self.embed_chunk_lens:
            length_embeds = self.length_embed(chunk_lens.float())
            downsampled_tokens = downsampled_tokens + length_embeds

        # returning

        outputs = Outputs(downsampled_tokens, upsample, weighted_aux_loss)

        return (outputs, intermediates) if return_intermediates else outputs
