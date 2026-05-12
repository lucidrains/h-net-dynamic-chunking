# following section 2.2 of the paper

from collections import namedtuple

import torch
from torch import cat, arange, tensor, repeat_interleave
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Module, Linear, Parameter, Sequential, RMSNorm, Identity
from torch.nn.functional import cosine_similarity, pad
from torch.nn.utils.rnn import pad_sequence

from einx import multiply
from einops import repeat, rearrange
from einops.layers.torch import Rearrange, Reduce

from assoc_scan import AssocScan
from torch_einops_utils import lens_to_mask, pad_right_at_dim_to, masked_mean
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

def identity(t):
    return t

def straight_through(t, value):
    return t + (value - t).detach()

def frac_gradient(t, frac = 1.):
    if frac == 1:
        return t

    t_grad = t * frac
    return straight_through(t_grad, t)

# classes

class MultiHeadDynamicSequenceChunker(Module):
    def __init__(
        self,
        dim,
        heads = 1,
        heads_merged_with_batch = True,     # if set to False, will move the head dimension to 0, so (h b n d), in the case you want to process different heads with different networks in parallel and then recombine - or you can keep it merged and just have the same network process different 'tokenized views'
        concat_heads = True,                # default will be classic concat linear combine, as done in multi-head attention, but if turned False, will just sum pool them
        dim_queries_keys = None,
        boundary_threshold = 0.5,
        target_avg_token_length = 6.,       # N in eq(10)
        ratio_loss_weight = 3e-2,
        handle_residual_proj = False,       # turning this on will automatically handle a projection of the residual and its application in the inverse upsample function
        assoc_scan_use_accelerated = False,
        learning_rate_difference = 0.75,    # in the paper, they report that as one moves up a hierarchy, the learning rate needs to decrease. we'll default to 0.75 for the rough 2.0 -> 1.5 somewhere in the appendix from level 0 -> 1
        straight_through_frac_vecs = True,  # improvisation where F receives gradients through straight-through with sigmoid
        add_hier_ar_loss = False,            # "hierarchical autoregressive" loss - just an extra projection at the end and made to predict the next input token
        detach_hier_target = True,
        embed_chunk_lens = False,
    ):
        super().__init__()
        dim_queries_keys = default(dim_queries_keys, dim)

        # linear to queries and keys

        self.to_queries_keys = Linear(dim, heads* dim_queries_keys * 2, bias = False)

        # multi-head related

        self.heads = heads

        self.split_heads = Rearrange('b ... (h d) -> h b ... d', h = heads)

        self.heads_merged_with_batch = heads_merged_with_batch

        if heads == 1:
            self.combine_heads = Identity()

        elif concat_heads:
            self.combine_heads = Sequential(
                Rearrange('(h b) ... d -> b ... (h d)', h = heads),
                Linear(dim * heads, dim)
            )

        else:
            self.combine_heads = Reduce('(h b) n d -> b n d', 'sum', h = heads)

        # start key token, so first token can be segmented / chunked out

        self.start_key_token = Parameter(torch.randn(heads, dim_queries_keys) * 1e-2) # presumably, need a start key token for the first token, open an issue if i got it wrong

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

        # maybe hierarchical loss

        self.add_hier_ar_loss = add_hier_ar_loss
        self.detach_hier_target = detach_hier_target

        if add_hier_ar_loss:
            self.to_hier_ar_pred = Sequential(
                RMSNorm(dim),
                Linear(dim, dim)
            )

        # length embed

        self.embed_chunk_lens = embed_chunk_lens

        self.length_embed = None
        if embed_chunk_lens:
            self.length_embed = Sequential(
                Rearrange('... -> ... 1'),
                create_mlp(dim, depth = 2, dim_in = 1, dim_out = dim)
            )

        # zero

        self.register_buffer('zero', tensor(0.), persistent = False)

    def upsample(
        self,
        downsampled,
        intermediates: Intermediates,
        apply_scale = True
    ):

        # split the heads back out if needed

        if not self.heads_merged_with_batch:
            downsampled = rearrange(downsampled, 'h b ... -> (h b) ...', h = self.heads)

        batch, needs_grad, device = downsampled.shape[0], downsampled.requires_grad, downsampled.device

        mask = intermediates.mask
        gates = intermediates.gates
        residual = intermediates.residual

        # handle maybe hierarchical autoregressive loss

        if self.add_hier_ar_loss:

            hier_pred = self.to_hier_ar_pred(downsampled[:, :-1])
            hier_target = intermediates.input_downsampled_tokens[:, 1:]
            hier_pred_mask = mask[:, 1:]

            maybe_detach = torch.detach if self.detach_hier_target else identity

            ar_loss = 1. - cosine_similarity(hier_pred, maybe_detach(hier_target), dim = -1)

            ar_loss = ar_loss[hier_pred_mask].mean()

        # smoothing module for improved gradients eq(5)

        downsampled = self.smooth_assoc_scan(gates, downsampled)

        # get the mask and residual from downsample steps

        upsampled_flat = repeat_interleave(downsampled[mask], intermediates.chunk_lens[mask], dim = 0)
        effective_lens = intermediates.chunk_lens.sum(dim = -1)
        upsampled = pad_sequence(upsampled_flat.split(effective_lens.tolist()), batch_first = True, padding_value = 0.)

        upsampled = pad_right_at_dim_to(upsampled, residual.shape[1], dim = 1)

        scale = intermediates.upsampler_output_scale

        if needs_grad and apply_scale and exists(scale):
            upsampled = multiply('b n d, b n', upsampled, scale)

        # handle combining of heads

        upsampled = self.combine_heads(upsampled)

        if self.handle_residual_proj:
            upsampled = upsampled + self.residual_proj(residual)

        upsampled = frac_gradient(upsampled, self.learning_rate_difference)

        if not self.add_hier_ar_loss:
            return upsampled

        return upsampled, ar_loss

    def forward(
        self,
        tokens, # float[b n d],
        lens = None,
        return_intermediates = False,
        return_only_chunk_lens = False
    ):
        batch, length, device, heads = *tokens.shape[:2], tokens.device, self.heads

        residual = tokens

        queries, keys = self.to_queries_keys(tokens).chunk(2, dim = -1)

        queries, keys = tuple(self.split_heads(t) for t in (queries, keys))

        start_keys = repeat(self.start_key_token, 'h d -> h b 1 d', b = batch)

        # merge heads into batch

        queries = rearrange(queries, 'h b ... -> (h b) ...')
        keys = rearrange(keys, 'h b ... -> (h b) ...')
        start_keys = rearrange(start_keys, 'h b ... -> (h b) ...')

        # concat first key

        keys = cat((start_keys, keys), dim = 1)

        # each query looks at the previous key to determine if distance is greater than some threshold for determining a boundary exists (they use 0.5 as threshold)

        cosine_sim  = cosine_similarity(queries, keys[:, :-1], dim = -1)

        probs = (1. - cosine_sim) * 0.5 # cosine sim is -1. to 1., this transforms it to 0. to 1.

        boundary_mask = probs > self.boundary_threshold # bool[b n]

        boundary_mask[:, 0] = True # first token must always be boundary

        if exists(lens):
            lens = repeat(lens, 'b -> (h b)', h = heads)
            seq_mask = lens_to_mask(lens, max_len = length)
            boundary_mask = boundary_mask & seq_mask

        # compute some lengths, per chunk and number of chunks per batch

        num_chunks = boundary_mask.long().sum(dim = -1)

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

        tokens = repeat(tokens, 'b ... -> (h b) ...', h = heads)

        boundary_tokens = tokens[boundary_mask] # pick out boundary tokens

        downsampled_tokens = pad_sequence(boundary_tokens.split(num_chunks.tolist()), batch_first=True, padding_value=0.)

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

        def upsample(downsampled, apply_scale = True):

            return self.upsample(downsampled, intermediates, apply_scale = apply_scale)

        # adjust learning rate

        downsampled_tokens = frac_gradient(downsampled_tokens, self.learning_rate_difference ** -1)

        # maybe split out heads

        if not self.heads_merged_with_batch:
            downsampled_tokens = rearrange(downsampled_tokens, '(h b) ... -> h b ...', h = heads)

        # length embed

        if self.embed_chunk_lens:
            chunk_lens_for_embed = chunk_lens.float()
            if not self.heads_merged_with_batch:
                chunk_lens_for_embed = rearrange(chunk_lens_for_embed, '(h b) ... -> h b ...', h = heads)

            length_embeds = self.length_embed(chunk_lens_for_embed)
            downsampled_tokens = downsampled_tokens + length_embeds

        # returning

        outputs = Outputs(downsampled_tokens, upsample, weighted_aux_loss)

        return (outputs, intermediates) if return_intermediates else outputs
