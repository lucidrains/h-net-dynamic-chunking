# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "assoc-scan",
#     "discrete-continuous-embed-readout",
#     "einops",
#     "fire",
#     "gymnasium[box2d]>=1.0.0",
#     "gymnasium[other]",
#     "hl-gauss-pytorch",
#     "memmap-replay-buffer",
#     "numpy",
#     "torch",
#     "tqdm",
#     "vector-quantize-pytorch",
#     "wandb",
#     "x-evolution>=0.1.32",
#     "x-mlps-pytorch",
#     "x-transformers"
# ]
# ///

# inspiration from Kobayashi et al. https://arxiv.org/abs/2512.20605 and Hanjung Kim et al https://arxiv.org/abs/2603.05815

from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import shutil
from pathlib import Path
from collections import deque

import numpy as np
from tqdm import tqdm
import fire
import wandb

from accelerate import Accelerator

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from einops import rearrange, repeat

import gymnasium as gym

from x_evolution import EvoStrategy

from x_mlps_pytorch import MLP
from hl_gauss_pytorch import HLGaussLoss
from discrete_continuous_embed_readout import Readout, Embed
from memmap_replay_buffer import ReplayBuffer
from torch_einops_utils.save_load import save_load
from torch_einops_utils import lens_to_mask
from assoc_scan import AssocScan
from x_transformers import Decoder
from vector_quantize_pytorch import VectorQuantize
from h_net_dynamic_chunking.h_net import HNet

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

# actor critic - simple mlp based ppo agent

class ActorCritic(nn.Module):
    def __init__(self, state_dim = 8, action_dim = 4):
        super().__init__()

        # actor

        self.actor_mlp = MLP(state_dim, 64, 64, 64)
        self.actor_readout = Readout(dim = 64, num_discrete = action_dim)

        # critic

        self.hl_gauss_loss = HLGaussLoss(
            min_value = -400.,
            max_value = 400.,
            num_bins = 256
        )

        self.critic_mlp = MLP(state_dim, 64, 64, 256)

    def get_action_and_value(self, state, action = None, temperature = 1.):
        actor_features = self.actor_mlp(state)
        logits = self.actor_readout(actor_features)
        logits = cast_tuple(logits)[0]

        if not exists(action):
            action = self.actor_readout.sample(logits, temperature = temperature)

        probs = torch.distributions.Categorical(logits = logits)

        critic_logits = self.critic_mlp(state)
        value = self.hl_gauss_loss(critic_logits)

        return action, probs.log_prob(action), probs.entropy(), value, critic_logits

# discovery module - hierarchical transformer with h-net dynamic chunking

@save_load
class DiscoveryModule(nn.Module):
    def __init__(
        self,
        state_dim = 8,
        action_dim = 4,
        dim = 128,
        decoder1_depth = 1,
        hnet_depth = 1,
        decoder2_depth = 1,
        heads = 4,
        dim_head = 32,
        use_pope = True,
        discrete_high_actions = False,
        vq_codebook_size = 64,
        vq_decay = 0.8,
        vq_commitment_weight = 1.,
        decoder_kwargs: dict | None = None,
        hnet_kwargs: dict | None = None,
        loss_weights: dict | None = None
    ):
        super().__init__()

        self.use_pope = use_pope

        default_pos_kwargs = dict(
            polar_pos_emb = use_pope,
            rotary_pos_emb = not use_pope
        )

        decoder_kwargs = default(decoder_kwargs, default_pos_kwargs)
        for k, v in default_pos_kwargs.items():
            decoder_kwargs.setdefault(k, v)

        hnet_kwargs = default(hnet_kwargs, dict(heads = 1, target_avg_token_length = 4.))
        loss_weights = default(loss_weights, dict(
            state_to_action = 1.,
            action_to_state = 0.1,
            action_to_action = 1.,
            hnet_aux = 0.1
        ))

        self.dim = dim
        self.loss_weights = loss_weights

        # embeddings

        self.state_proj = nn.Linear(state_dim, dim)
        self.action_emb = nn.Embedding(action_dim, dim)

        # decoder sandwich: decoder1 → hnet → decoder2

        inner_decoder_kwargs = {**dict(pre_norm_has_final_norm = False), **decoder_kwargs}

        self.decoder1 = Decoder(dim = dim, depth = decoder1_depth, heads = heads, attn_dim_head = dim_head, **inner_decoder_kwargs)

        self.hnet_prenorm = nn.LayerNorm(dim, elementwise_affine = False)

        self.discrete_high_actions = discrete_high_actions

        vq = None
        if discrete_high_actions:
            vq = VectorQuantize(
                dim = dim,
                codebook_size = vq_codebook_size,
                decay = vq_decay,
                commitment_weight = vq_commitment_weight,
                rotation_trick = True,
                kmeans_init = True,
                kmeans_iters = 10
            )

        self.vq = vq

        self.hnet = HNet(
            encoder = nn.Identity(),
            network = Decoder(dim = dim, depth = hnet_depth, heads = heads, attn_dim_head = dim_head, **inner_decoder_kwargs),
            decoder = nn.Identity(),
            dim = dim,
            vq = vq,
            **hnet_kwargs
        )

        self.decoder2 = Decoder(dim = dim, depth = decoder2_depth, heads = heads, attn_dim_head = dim_head, **decoder_kwargs)

        # readouts

        self.state_readout = Readout(dim = dim, num_discrete = action_dim)
        self.action_readout = Readout(dim = dim, num_discrete = action_dim)
        self.to_next_state = nn.Linear(dim, state_dim)

    def forward(self, states, actions, mask = None, episode_lens = None, return_loss_breakdown = False, extract_high_level_actions = False):
        b, n = states.shape[:2]
        device = states.device

        if exists(episode_lens):
            assert not exists(mask)
            mask = lens_to_mask(episode_lens, max_len = n)

        # interleave [s0, a0, s1, a1, ...]

        state_repr = self.state_proj(states)
        action_repr = self.action_emb(actions)

        interleaved = torch.empty((b, 2 * n, self.dim), device = device, dtype = states.dtype)
        interleaved[:, 0::2] = state_repr
        interleaved[:, 1::2] = action_repr

        interleaved_mask = None
        if exists(mask):
            interleaved_mask = repeat(mask, 'b n -> b (n 2)')

        # decoder1 over full interleaved stream

        dec1_out = self.decoder1(interleaved, mask = interleaved_mask)

        # hnet processes only state positions

        state_out = dec1_out[:, 0::2]
        action_out = dec1_out[:, 1::2]

        hnet_ret = self.hnet(self.hnet_prenorm(state_out), lens = episode_lens, vq_mask = mask, return_intermediates = extract_high_level_actions)
        hnet_state_out = hnet_ret.output
        hnet_aux_loss = hnet_ret.loss

        if extract_high_level_actions:
            intermediates = hnet_ret.intermediates
            extracted = intermediates.quantized_downsampled_indices if self.discrete_high_actions else intermediates.input_downsampled_tokens

            if exists(mask):
                high_level_mask = intermediates.mask
                if self.discrete_high_actions:
                    extracted = extracted.masked_fill(~high_level_mask, -1)
                else:
                    extracted = extracted.masked_fill(~high_level_mask.unsqueeze(-1), 0.)

            return extracted, intermediates.chunk_lens

        # re-interleave: hnet-processed states + raw actions

        reinterleaved = torch.empty_like(interleaved)
        reinterleaved[:, 0::2] = hnet_state_out
        reinterleaved[:, 1::2] = action_out

        # decoder2 over re-interleaved stream

        dec2_out = self.decoder2(reinterleaved, mask = interleaved_mask)

        state_features = dec2_out[:, 0::2]
        action_features = dec2_out[:, 1::2]

        # losses - masked for variable-length episodes

        pred_actions = rearrange(self.state_readout(state_features), 'b n d -> (b n) d')
        target_actions = rearrange(actions, 'b n -> (b n)').long()

        if exists(mask):
            flat_mask = rearrange(mask, 'b n -> (b n)')
            pred_actions = pred_actions[flat_mask]
            target_actions = target_actions[flat_mask]

        loss_state_to_action = F.cross_entropy(pred_actions, target_actions)

        loss_action_to_state = tensor(0., device = device)
        loss_action_to_action = tensor(0., device = device)

        if n > 1:
            pred_next_states = rearrange(self.to_next_state(action_features[:, :-1]), 'b n d -> (b n) d')
            pred_next_actions = rearrange(self.action_readout(action_features[:, :-1]), 'b n d -> (b n) d')

            next_states = rearrange(states[:, 1:], 'b n d -> (b n) d')
            next_actions = rearrange(actions[:, 1:], 'b n -> (b n)').long()

            if exists(mask):
                shifted_mask = rearrange(mask[:, :-1] & mask[:, 1:], 'b n -> (b n)')

                pred_next_states = pred_next_states[shifted_mask]
                pred_next_actions = pred_next_actions[shifted_mask]
                next_states = next_states[shifted_mask]
                next_actions = next_actions[shifted_mask]

            if pred_next_states.numel() > 0:
                loss_action_to_state = F.mse_loss(pred_next_states, next_states)
                loss_action_to_action = F.cross_entropy(pred_next_actions, next_actions)

        w = self.loss_weights
        total_loss = (
            w.get('state_to_action', 1.) * loss_state_to_action +
            w.get('action_to_state', 1.) * loss_action_to_state +
            w.get('action_to_action', 1.) * loss_action_to_action +
            w.get('hnet_aux', 1.) * hnet_aux_loss
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, dict(
            state_to_action = loss_state_to_action.item(),
            action_to_state = loss_action_to_state.item(),
            action_to_action = loss_action_to_action.item(),
            hnet_aux = hnet_aux_loss.item()
        )

    @torch.no_grad()
    def get_action_and_value(self, state, action = None, cache = None, extract_high_level_actions = False, temperature = 1.):
        cache_dec1, cache_hnet, cache_dec2 = default(cache, (None, None, None))
        is_caching = exists(cache)

        # process state through decoder1 → hnet → decoder2

        state_repr = self.state_proj(state)

        dec1_out, cache_dec1 = self.decoder1(
            rearrange(state_repr, 'b d -> b 1 d'),
            cache = cache_dec1,
            return_hiddens = True,
            input_not_include_cache = is_caching
        )

        hnet_ret = self.hnet(
            self.hnet_prenorm(dec1_out),
            cache = cache_hnet,
            return_hiddens = True,
            return_intermediates = extract_high_level_actions
        )

        if extract_high_level_actions:
            intermediates = hnet_ret.intermediates
            extracted = intermediates.quantized_downsampled_indices if self.discrete_high_actions else intermediates.input_downsampled_tokens
            return extracted, intermediates.chunk_lens, (cache_dec1, hnet_ret.next_cache, cache_dec2)

        cache_hnet = hnet_ret.next_cache

        dec2_out, cache_dec2 = self.decoder2(
            hnet_ret.output,
            cache = cache_dec2,
            return_hiddens = True,
            input_not_include_cache = is_caching
        )

        # readout action from state position

        action_logits = self.state_readout(rearrange(dec2_out, 'b 1 d -> b d'))

        if not exists(action):
            action = self.state_readout.sample(action_logits, temperature = temperature)

        probs = torch.distributions.Categorical(logits = action_logits)

        # process action through decoder1 → decoder2 (skip hnet for action positions)

        action_repr = self.action_emb(action)

        a_dec1_out, cache_dec1 = self.decoder1(
            rearrange(action_repr, 'b d -> b 1 d'),
            cache = cache_dec1,
            return_hiddens = True,
            input_not_include_cache = True
        )

        _, cache_dec2 = self.decoder2(
            a_dec1_out,
            cache = cache_dec2,
            return_hiddens = True,
            input_not_include_cache = True
        )

        next_cache = (cache_dec1, cache_hnet, cache_dec2)

        return action, probs.entropy(), action_logits, next_cache

# conditioned actor critic - ppo agent conditioned on high level actions

class ConditionedActorCritic(nn.Module):
    def __init__(
        self,
        state_dim = 8,
        action_dim = 4,
        condition_dim = 96,
        actor_hidden_dims = (64, 64),
        critic_hidden_dims = (64, 256),
        hl_gauss_min_value = -400.,
        hl_gauss_max_value = 400.,
        hl_gauss_num_bins = 256
    ):
        super().__init__()

        # actor

        first_actor_dim = actor_hidden_dims[0]
        self.actor_state_proj = nn.Linear(state_dim, first_actor_dim)
        self.actor_cond_proj = nn.Linear(condition_dim, first_actor_dim)

        self.actor_mlp = MLP(first_actor_dim, *actor_hidden_dims)
        self.actor_readout = Readout(dim = actor_hidden_dims[-1], num_discrete = action_dim)

        # critic

        self.hl_gauss_loss = HLGaussLoss(
            min_value = hl_gauss_min_value,
            max_value = hl_gauss_max_value,
            num_bins = hl_gauss_num_bins
        )

        first_critic_dim = critic_hidden_dims[0]
        self.critic_state_proj = nn.Linear(state_dim, first_critic_dim)
        self.critic_cond_proj = nn.Linear(condition_dim, first_critic_dim)

        self.critic_mlp = MLP(first_critic_dim, *critic_hidden_dims)

    def forward(self, state, condition):
        features = F.silu(self.actor_state_proj(state) + self.actor_cond_proj(condition))
        features = self.actor_mlp(features)
        return self.actor_readout(features)

    def forward_with_cfg(self, state, condition, cond_scale = 1., state_scale = 1.):
        """ classifier free guidance - scale away from null-condition or null-state """

        logits = self.forward(state, condition)

        if cond_scale == 1. and state_scale == 1.:
            return logits

        assert not (cond_scale != 1. and state_scale != 1.), 'cannot guide on both condition and state simultaneously'

        if cond_scale != 1.:
            null_logits = self.forward(state, torch.zeros_like(condition))
            scale = cond_scale
        else:
            null_logits = self.forward(torch.zeros_like(state), condition)
            scale = state_scale

        is_tuple = isinstance(logits, tuple)
        logits_tensor = logits[0] if is_tuple else logits
        null_logits_tensor = null_logits[0] if is_tuple else null_logits

        cfg_logits = null_logits_tensor + (logits_tensor - null_logits_tensor) * scale
        return (cfg_logits,) if is_tuple else cfg_logits

    def get_action_and_value(self, state, condition, action = None, temperature = 1., cond_scale = 1., state_scale = 1.):
        logits = self.forward_with_cfg(state, condition, cond_scale = cond_scale, state_scale = state_scale)
        logits = cast_tuple(logits)[0]

        if not exists(action):
            action = self.actor_readout.sample(logits, temperature = temperature)

        probs = torch.distributions.Categorical(logits = logits)

        critic_features = F.silu(self.critic_state_proj(state) + self.critic_cond_proj(condition))
        critic_logits = self.critic_mlp(critic_features)
        value = self.hl_gauss_loss(critic_logits)

        return action, probs.log_prob(action), probs.entropy(), value, critic_logits

# unified metacontroller

@save_load
class Metacontroller(nn.Module):
    def __init__(
        self,
        discovery_mod,
        state_dim = 8,
        action_dim = 4,
        dim = 96,
        decoder_depth = 4,
        decoder_heads = 8,
        decoder_dim_head = 32,
        log_var_range = (-5., 2.),
        condition_dropout = 0.5,
        state_dropout = 0.2,
        condition_on_past_actions = False,
        lower_controller_kwargs: dict | None = None,
        hl_gauss_min_value = -400.,
        hl_gauss_max_value = 400.,
        hl_gauss_num_bins = 256
    ):
        super().__init__()
        assert (condition_dropout + state_dropout) <= 1., 'condition and state dropout must sum to at most 1'

        self.discovery_mod = discovery_mod
        self.discovery_mod.requires_grad_(False)
        self.discovery_mod.eval()

        discovery_dim = discovery_mod.dim

        self.condition_dropout = condition_dropout
        self.state_dropout = state_dropout

        self.condition_on_past_actions = condition_on_past_actions
        if condition_on_past_actions:
            self.fine_action_emb = Embed(dim = dim, num_discrete = action_dim)
            self.fine_action_start_token = nn.Parameter(torch.randn(dim))

        self.state_proj = nn.Linear(state_dim, dim)

        self.metacontroller = Decoder(
            dim = dim,
            depth = decoder_depth,
            heads = decoder_heads,
            attn_dim_head = decoder_dim_head,
            polar_pos_emb = True,
            rotary_pos_emb = False
        )

        is_discrete = discovery_mod.discrete_high_actions

        if is_discrete:
            self.action_emb_proj = nn.Embedding(discovery_mod.vq.codebook_size, dim)
            self.high_action_readout = Readout(
                dim = dim,
                num_discrete = discovery_mod.vq.codebook_size
            )
        else:
            self.action_emb_proj = nn.Linear(discovery_dim, dim) if discovery_dim != dim else nn.Identity()
            self.high_action_readout = Readout(
                dim = dim,
                num_continuous = discovery_dim,
                continuous_log_var_embed = True,
                continuous_dist_kwargs = dict(log_var_clamp_range = log_var_range)
            )

        self.state_readout = Readout(
            dim = dim,
            num_continuous = state_dim,
            continuous_log_var_embed = False,
            regression_loss_type = 'mse'
        )

        self.hl_gauss_loss = HLGaussLoss(
            min_value = hl_gauss_min_value,
            max_value = hl_gauss_max_value,
            num_bins = hl_gauss_num_bins
        )
        self.critic_mlp = MLP(dim, 128, 256, 256)


        lower_controller_kwargs = default(lower_controller_kwargs, dict())
        self.lower_controller = ConditionedActorCritic(
            state_dim = state_dim,
            action_dim = action_dim,
            condition_dim = dim if is_discrete else discovery_dim,
            **lower_controller_kwargs
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.discovery_mod.eval()
        return self

    def maybe_cfg_dropout(
        self,
        states,
        high_actions,
        force_drop_condition = False,
        force_drop_state = False
    ):
        """ mutually exclusive dropout on condition or state for classifier free guidance """

        assert not (force_drop_condition and force_drop_state), 'cannot drop both condition and state'

        if force_drop_condition:
            return states, torch.zeros_like(high_actions)

        if force_drop_state:
            return torch.zeros_like(states), high_actions

        if not self.training:
            return states, high_actions

        if self.condition_dropout <= 0. and self.state_dropout <= 0.:
            return states, high_actions

        batch, device = states.shape[0], states.device
        rand = torch.rand((batch, 1, 1), device = device)

        drop_cond = rand < self.condition_dropout
        drop_state = (rand >= self.condition_dropout) & (rand < (self.condition_dropout + self.state_dropout))

        high_actions = torch.where(drop_cond, 0., high_actions)
        states = torch.where(drop_state, 0., states)

        return states, high_actions

    def forward(
        self,
        states,
        actions,
        mask = None,
        episode_lens = None,
        return_loss_breakdown = False,
        force_drop_condition = False,
        force_drop_state = False
    ):
        batch, seq, device = *states.shape[:2], states.device

        if exists(episode_lens):
            assert not exists(mask)
            mask = lens_to_mask(episode_lens, max_len = seq)

        # extract teacher higher level actions

        with torch.no_grad():
            self.discovery_mod.eval()
            raw_target_high_actions, chunk_lens = self.discovery_mod(states, actions, mask = mask, episode_lens = episode_lens, extract_high_level_actions = True)

            flat_target_high_actions = rearrange(raw_target_high_actions, 'b c ... -> (b c) ...')
            flat_chunk_lens = rearrange(chunk_lens, 'b c -> (b c)')

            expanded = torch.repeat_interleave(flat_target_high_actions, flat_chunk_lens, dim = 0)
            pattern = '(b n) -> b n' if self.discovery_mod.discrete_high_actions else '(b n) d -> b n d'
            raw_target_high_actions = rearrange(expanded, pattern, b = batch)

        # interleave states and high_actions

        state_repr = self.state_proj(states)

        if self.condition_on_past_actions:
            fine_action_tokens = self.fine_action_emb(actions)
            start_token = repeat(self.fine_action_start_token, 'd -> b 1 d', b = batch)
            past_action_tokens = torch.cat((start_token, fine_action_tokens[:, :-1]), dim = 1)
            state_repr = state_repr + past_action_tokens

        action_tokens = self.action_emb_proj(raw_target_high_actions)

        interleaved = torch.empty((batch, 2 * seq, self.metacontroller.dim), device = device, dtype = states.dtype)
        interleaved[:, 0::2] = state_repr
        interleaved[:, 1::2] = action_tokens

        # interleaved mask

        interleaved_mask = None
        if exists(mask):
            interleaved_mask = repeat(mask, 'b n -> b (n 2)')

        dec_out = self.metacontroller(interleaved, mask = interleaved_mask)

        state_features = dec_out[:, 0::2]
        action_features = dec_out[:, 1::2]

        loss_high = self.high_action_readout(state_features, targets = raw_target_high_actions, loss_mask = mask)

        loss_next_state = tensor(0., device = device)
        if seq > 1:
            next_states = states[:, 1:]
            shifted_mask = mask[:, :-1] & mask[:, 1:] if exists(mask) else None
            loss_next_state = self.state_readout(action_features[:, :-1], targets = next_states, loss_mask = shifted_mask)

        # cfg dropout + lower controller loss

        is_discrete = self.discovery_mod.discrete_high_actions

        states_cond, high_cond = self.maybe_cfg_dropout(
            states,
            raw_target_high_actions if not is_discrete else action_tokens,
            force_drop_condition = force_drop_condition,
            force_drop_state = force_drop_state
        )

        logits = self.lower_controller(states_cond, high_cond)
        logits = cast_tuple(logits)[0]

        logits = rearrange(logits, 'b n d -> (b n) d')
        actions_flat = rearrange(actions, 'b n -> (b n)')

        if exists(mask):
            flat_mask = rearrange(mask, 'b n -> (b n)')
            logits = logits[flat_mask]
            actions_flat = actions_flat[flat_mask]

        loss_low = F.cross_entropy(logits, actions_flat.long())

        # combine

        loss = loss_high + loss_next_state + loss_low

        if not return_loss_breakdown:
            return loss

        return loss, dict(loss_high = loss_high.item(), loss_next_state = loss_next_state.item(), loss_low = loss_low.item())

    def get_action_and_value(
        self,
        state,
        action = None,
        cache = None,
        temperature = 1.,
        high_temperature = 1.,
        force_drop_condition = False,
        force_drop_state = False,
        cond_scale = 1.,
        state_scale = 1.,
        extract_high_level_actions = False
    ):
        is_caching = exists(cache)
        state_seq = rearrange(state, 'b d -> b 1 d')

        if is_caching:
            transformer_cache, last_high_action, last_fine_action = cache
        else:
            transformer_cache = None
            last_high_action = None
            last_fine_action = None

        state_repr = self.state_proj(state_seq)

        if self.condition_on_past_actions:
            if exists(last_fine_action):
                last_fine_action_emb = self.fine_action_emb(rearrange(last_fine_action, 'b -> b 1'))
                state_repr = state_repr + last_fine_action_emb
            else:
                start_token = repeat(self.fine_action_start_token, 'd -> b 1 d', b = state_repr.shape[0])
                state_repr = state_repr + start_token

        dec_out, transformer_cache = self.metacontroller(
            state_repr,
            cache = transformer_cache,
            input_not_include_cache = is_caching,
            return_hiddens = True
        )

        is_discrete = self.discovery_mod.discrete_high_actions

        if is_discrete:
            # high_action_readout is a Discrete Readout
            high_action_logits = self.high_action_readout(rearrange(dec_out, 'b 1 d -> b d'))

            if not exists(action) or not extract_high_level_actions:
                pred_high_action = self.high_action_readout.sample(high_action_logits, temperature=high_temperature)
            else:
                pred_high_action = action

            probs = torch.distributions.Categorical(logits = high_action_logits)
            high_log_prob = probs.log_prob(pred_high_action)
            high_entropy = probs.entropy()

            action_token = self.action_emb_proj(pred_high_action.detach())
            action_token = rearrange(action_token, 'b d -> b 1 d')
        else:
            continuous_params = self.high_action_readout(rearrange(dec_out, 'b 1 d -> b d'))
            dist = self.high_action_readout.continuous_dist.dist(continuous_params)

            if high_temperature == 0.:
                pred_high_action = dist.mean
            else:
                if high_temperature > 0. and high_temperature != 1.:
                    dist = torch.distributions.Normal(dist.mean, dist.stddev * high_temperature)

                pred_high_action = dist.sample()

            # project predicted raw action into transformer token space
            action_token = self.action_emb_proj(rearrange(pred_high_action.detach(), 'b d -> b 1 d'))

        _, transformer_cache = self.metacontroller(
            action_token,
            cache = transformer_cache,
            input_not_include_cache = True,
            return_hiddens = True
        )

        # cfg dropout for lower controller uses the raw predicted high action
        state_cond, high_cond = self.maybe_cfg_dropout(
            rearrange(state, 'b d -> b 1 d'),
            action_token if is_discrete else rearrange(pred_high_action.detach(), 'b d -> b 1 d'),
            force_drop_condition = force_drop_condition,
            force_drop_state = force_drop_state
        )

        state_cond = rearrange(state_cond, 'b 1 d -> b d')
        high_cond = rearrange(high_cond, 'b 1 d -> b d')

        action_out, _, entropy, _, critic_logits = self.lower_controller.get_action_and_value(
            state_cond,
            high_cond,
            action = action if not extract_high_level_actions else None,
            temperature = temperature,
            cond_scale = cond_scale,
            state_scale = state_scale
        )

        next_cache = (transformer_cache, pred_high_action.detach(), action_out.detach())

        # Value estimation for higher metacontroller
        critic_logits_high = self.critic_mlp(rearrange(dec_out, 'b 1 d -> b d'))
        value_high = self.hl_gauss_loss(critic_logits_high)

        if extract_high_level_actions:
            return pred_high_action, high_log_prob, high_entropy, value_high, critic_logits_high

        return action_out, entropy, critic_logits, next_cache, pred_high_action, high_log_prob, value_high


# gae using associative scan

def calc_gae(
    rewards,
    values,
    masks,
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None
):
    assert values.shape[-1] == rewards.shape[-1]
    use_accelerated = default(use_accelerated, rewards.is_cuda)

    values_padded = F.pad(values, (0, 1), value = 0.)
    values, values_next = values_padded[:-1], values_padded[1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)
    gae = scan(gates, delta)

    return gae + values

# evaluation

def evaluate_agent(
    env_name = 'LunarLander-v3',
    model = None,
    num_episodes = 20,
    video_folder = 'eval_videos',
    record_every = 5,
    device = 'cpu',
    seed = 42,
    store_buffer = False,
    buffer_folder = './discovery_buffer',
    render_videos = True,
    forward_has_cache = False,
    filter_top_percentile = 0.5,
    quiet = False,
    vectorization_mode = None,
    temperature = 1.,
    high_temperature = 1.,
    max_timesteps = 1000,
    desc = 'Evaluating Agent',
    force_drop_condition = False,
    force_drop_state = False,
    cond_scale = 1.,
    state_scale = 1.
):
    shutil.rmtree(video_folder, ignore_errors = True)

    original_device = next(model.parameters()).device
    model = model.to(device)

    env = gym.make(env_name, render_mode = 'rgb_array' if render_videos else None)

    if render_videos:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder = video_folder,
            episode_trigger = lambda x: x % record_every == 0 if record_every > 0 else False,
            disable_logger = True
        )

    buffer = None

    if store_buffer:
        shutil.rmtree(buffer_folder, ignore_errors = True)
        Path(buffer_folder).mkdir(parents = True, exist_ok = True)

        buffer = ReplayBuffer(
            folder = buffer_folder,
            max_episodes = num_episodes,
            max_timesteps = max_timesteps,
            fields = dict(
                state = ('float', (8,)),
                action = ('int', ())
            )
        )

    unwrapped_model = model.model if hasattr(model, 'model') else model
    unwrapped_model.eval()

    if exists(vectorization_mode):
        assert not store_buffer, "store_buffer is not supported in vectorized mode"
        assert not render_videos, "render_videos is not supported in vectorized mode"

        env = gym.make_vec(env_name, num_envs=num_episodes, vectorization_mode=vectorization_mode)
        state, _ = env.reset(seed=seed)
        done = np.zeros(num_episodes, dtype=bool)
        step_count = 0
        ep_rewards = np.zeros(num_episodes)

        cache = None
        with torch.no_grad():
            while not done.all() and step_count < max_timesteps:
                state_tensor = tensor(state, dtype=torch.float32, device=device)

                kwargs = dict(cache = cache) if forward_has_cache else dict()
                if isinstance(unwrapped_model, Metacontroller):
                    kwargs['high_temperature'] = high_temperature
                    kwargs['force_drop_condition'] = force_drop_condition
                    kwargs['force_drop_state'] = force_drop_state
                    kwargs['cond_scale'] = cond_scale
                    kwargs['state_scale'] = state_scale

                action, *_, cache = unwrapped_model.get_action_and_value(state_tensor, **kwargs, temperature = temperature)

                a = action.cpu().numpy()
                next_state, reward, terminated, truncated, _ = env.step(a)

                ep_rewards += reward * (~done)
                done = done | terminated | truncated

                state = next_state
                step_count += 1

        model = model.to(original_device)
        return np.mean(ep_rewards)

    ep_rewards = []
    episodes_data = []
    pbar = tqdm(range(num_episodes), desc = desc, disable = quiet)

    for ep in pbar:
        ep_seed = seed + ep if exists(seed) else None
        state, _ = env.reset(seed = ep_seed)

        done = False
        step_count = 0
        ep_reward_sum = 0.

        ep_states = []
        ep_actions = []

        cache = None

        with torch.no_grad():
            while not done and step_count < max_timesteps:
                ep_states.append(state)
                state_tensor = rearrange(tensor(state, dtype = torch.float32, device = device), 'd -> 1 d')

                kwargs = dict(cache = cache) if forward_has_cache else dict()
                if isinstance(unwrapped_model, Metacontroller):
                    kwargs['high_temperature'] = high_temperature
                    kwargs['force_drop_condition'] = force_drop_condition
                    kwargs['force_drop_state'] = force_drop_state
                    kwargs['cond_scale'] = cond_scale
                    kwargs['state_scale'] = state_scale

                action, *_, cache = unwrapped_model.get_action_and_value(state_tensor, **kwargs, temperature = temperature)

                a = action.item()
                ep_actions.append(a)

                next_state, reward, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                state = next_state
                ep_reward_sum += reward
                step_count += 1

        ep_rewards.append(ep_reward_sum)
        avg_reward = np.mean(ep_rewards)
        pbar.set_postfix({'Avg Reward': f'{avg_reward:.2f}', 'Ep Reward': f'{ep_reward_sum:.2f}'})

        if store_buffer:
            episodes_data.append((ep_states, ep_actions, ep_reward_sum))

    avg_reward = np.mean(ep_rewards)
    if not quiet:
        print(f'\nEvaluation Complete! Average Reward over {num_episodes} episodes: {avg_reward:.2f}')

    if store_buffer:
        episodes_data.sort(key = lambda x: x[2], reverse = True)
        top_k = max(1, int(len(episodes_data) * filter_top_percentile))
        filtered_episodes = episodes_data[:top_k]

        filtered_avg_reward = np.mean([x[2] for x in filtered_episodes])
        if not quiet:
            print(f'Storing top {top_k} episodes with average reward: {filtered_avg_reward:.2f}')

        for ep_states, ep_actions, reward in filtered_episodes:
            with buffer.one_episode():
                for t in range(len(ep_states)):
                    buffer.store(state = ep_states[t], action = ep_actions[t])

    if not quiet:
        print(f'Videos saved to: {Path(video_folder).absolute()}')

    env.close()
    model = model.to(original_device)

    if store_buffer:
        return avg_reward, buffer

    return avg_reward

# main training entrypoint

def train_metacontroller(
    total_episodes = 750,
    lr = 3e-4,
    gamma = 0.99,
    gae_lambda = 0.95,
    clip_coef = 0.2,
    entropy_coef = 0.01,
    value_coef = 0.5,
    epochs = 4,
    batch_size = 64,
    rolling_window = 20,
    seed = 42,
    cpu = False,
    record_video_every = 50,
    use_wandb = False,
    target_avg_cum_reward = -50.0,
    ppo_joint_target_avg_cum_reward = 75.0,
    ppo_joint_total_episodes = 5000,
    ppo_joint_update_every = 25,
    ppo_joint_epochs = 2,
    margin_of_error = 5.0,
    evaluate = False,
    train_discovery = False,
    train_joint_metacontroller = False,
    condition_on_past_actions = False,
    train_evo_strat = False,
    train_evo_strat_joint = False,
    train_ppo_joint = False,
    evo_strat_target = 'inner',
    evo_strat_joint_target = 'higher',
    evo_strat_generations = 100,
    evo_strat_population_size = 64,
    evo_strat_learning_rate = 1e-3,
    evo_strat_noise_scale = 0.01,
    skip_ppo_eval = False,
    skip_discovery_eval = False,
    max_discovery_steps = 1000,
    max_timesteps = 1000,
    load_path = './metacontroller_agent.pt',
    eval_episodes = 20,
    filter_top_percentile = 0.5,
    discrete_high_actions = False,
    vq_codebook_size = 256,
    vq_decay = 0.8,
    vq_commitment_weight = 1.
):

    discovery_pt_name = 'discovery_discrete.pt' if discrete_high_actions else 'discovery_continuous.pt'
    discovery_evo_pt_name = 'discovery_evo_strat_discrete.pt' if discrete_high_actions else 'discovery_evo_strat_continuous.pt'

    torch.manual_seed(seed)
    np.random.seed(seed)

    shutil.rmtree('videos', ignore_errors = True)

    env = gym.make('LunarLander-v3', render_mode = 'rgb_array')
    if not train_evo_strat and not train_evo_strat_joint:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder = 'videos',
            episode_trigger = lambda x: x % record_video_every == 0 if record_video_every > 0 else False,
            disable_logger = True
        )

    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device

    if accelerator.num_processes > 1:
        import builtins
        _original_print = builtins.print
        builtins.print = lambda *args, **kwargs: _original_print(f'[rank{accelerator.process_index}]', *args, **kwargs)

        if cpu:
            import torch.distributed as dist
            orig_barrier = dist.barrier
            dist.barrier = lambda *args, device_ids = None, **kwargs: orig_barrier(*args, **kwargs)

    print(f'Using device: {device}')

    use_wandb = use_wandb and accelerator.is_main_process

    if use_wandb:
        if train_discovery:
            run_name = 'behavior_clone_phase'
        elif train_joint_metacontroller:
            run_name = 'joint_metacontroller_phase'
        elif train_evo_strat:
            run_name = 'evolutionary_strategy_phase'
        elif train_evo_strat_joint:
            run_name = 'joint_evolutionary_strategy_phase'
        elif train_ppo_joint:
            run_name = 'joint_ppo_phase'
        else:
            run_name = 'initial_policy_optimization'
        wandb.init(project = 'h-net-dynamic-chunking', name = run_name, config = dict(
            lr = lr, gamma = gamma, gae_lambda = gae_lambda,
            clip_coef = clip_coef, entropy_coef = entropy_coef, value_coef = value_coef,
            epochs = epochs, batch_size = batch_size
        ))

    model = ActorCritic(state_dim = 8, action_dim = 4).to(device)
    optimizer = Adam(model.parameters(), lr = lr)
    model, optimizer = accelerator.prepare(model, optimizer)

    # evaluation mode

    if evaluate:
        print(f'Loading agent weights from {load_path}...')
        model.load_state_dict(torch.load(load_path, map_location = 'cpu', weights_only = False))

        evaluate_agent(
            model = model,
            num_episodes = eval_episodes,
            video_folder = 'eval_videos',
            record_every = max(1, eval_episodes // 4),
            device = device,
            seed = seed
        )
        return

    # discovery distillation mode

    if train_discovery:
        print(f'Loading agent weights from {load_path}...')
        model.load_state_dict(torch.load(load_path, map_location = 'cpu', weights_only = False))

        if not skip_ppo_eval:
            avg_reward = evaluate_agent(
                model = model,
                num_episodes = 20,
                video_folder = 'initial_verification_videos',
                record_every = 0,
                device = device,
                seed = seed,
                store_buffer = False
            )

            print(f'Initial Verification Reward: {avg_reward:.2f}')
            expected_reward = target_avg_cum_reward - margin_of_error
            assert avg_reward >= expected_reward, f'Expected initial PPO reward >= {expected_reward:.2f}, got {avg_reward:.2f}. Please train PPO longer.'
        else:
            print('Skipping PPO initial evaluation as requested.')

        discovery_mod = DiscoveryModule(
            discrete_high_actions = discrete_high_actions,
            vq_codebook_size = vq_codebook_size,
            vq_decay = vq_decay,
            vq_commitment_weight = vq_commitment_weight
        ).to(device)

        if Path(discovery_pt_name).exists():
            print(f'Loading existing {discovery_pt_name}...')
            discovery_mod.load_state_dict(torch.load(discovery_pt_name, map_location = 'cpu', weights_only = False))

        disc_optimizer = Adam(discovery_mod.parameters(), lr = 1e-4)
        discovery_mod, disc_optimizer = accelerator.prepare(discovery_mod, disc_optimizer)

        best_eval_reward = -float('inf')
        loop_count = 0
        total_steps = 0
        discovery_target = target_avg_cum_reward + margin_of_error

        while best_eval_reward < discovery_target and total_steps < max_discovery_steps:
            loop_count += 1
            print(f'\n--- Discovery Training Loop {loop_count} ---')

            # gather expert trajectories

            gather_episodes = int(batch_size / filter_top_percentile)

            _, buffer = evaluate_agent(
                model = model,
                num_episodes = gather_episodes,
                video_folder = 'discovery_gather_videos',
                record_every = 0,
                device = device,
                seed = seed + loop_count * 1000,
                store_buffer = True,
                buffer_folder = './discovery_buffer',
                render_videos = False,
                filter_top_percentile = filter_top_percentile,
                desc = 'Gathering Expert Trajectories'
            )

            dataloader = buffer.dataloader(batch_size = batch_size, fields = ('state', 'action'))
            dataloader = accelerator.prepare(dataloader)

            # train discovery module

            discovery_mod.train()
            pbar_disc = tqdm(range(epochs), desc = 'Training DiscoveryModule')

            for ep in pbar_disc:
                total_loss_sum = 0.
                num_batches = 0

                for batch in dataloader:
                    if total_steps >= max_discovery_steps:
                        break

                    states, actions, lens = [batch[k].to(device) for k in ('state', 'action', '_lens')]

                    loss, breakdown = discovery_mod(states, actions, episode_lens = lens, return_loss_breakdown = True)

                    assert not torch.isnan(loss), 'NaN loss detected during discovery training!'

                    disc_optimizer.zero_grad()
                    accelerator.backward(loss)
                    nn.utils.clip_grad_norm_(discovery_mod.parameters(), 0.5)
                    disc_optimizer.step()
                    total_steps += 1

                    total_loss_sum += loss.item()
                    num_batches += 1

                    if use_wandb:
                        wandb.log(dict(discovery_loss = loss.item(), **breakdown))

                avg_loss = total_loss_sum / max(num_batches, 1)
                pbar_disc.set_postfix({'Loss': f'{avg_loss:.4f}'})

            unwrapped_mod = accelerator.unwrap_model(discovery_mod)

            # evaluate discovery policy

            print('Evaluating trained DiscoveryPolicy...')

            eval_reward = evaluate_agent(
                model = unwrapped_mod,
                num_episodes = eval_episodes,
                video_folder = 'discovery_eval_videos',
                record_every = 5,
                device = device,
                seed = seed + loop_count * 1000,
                store_buffer = False,
                forward_has_cache = True
            )

            if use_wandb:
                wandb.log(dict(
                    discovery_eval_reward = eval_reward,
                    discovery_loop = loop_count
                ))

                video_files = list(Path('discovery_eval_videos').glob('*.mp4'))

                if len(video_files) > 0:
                    latest_video = max(video_files, key = os.path.getmtime)
                    wandb.log({'discovery_eval_video': wandb.Video(str(latest_video))})

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                print(f'New best eval reward: {best_eval_reward:.2f}')

                torch.save(unwrapped_mod.state_dict(), discovery_pt_name)
                print(f'Saved new best {discovery_pt_name}!')

            if best_eval_reward >= discovery_target:
                print(f'DiscoveryModule reached target reward >= {discovery_target:.2f}! (best: {best_eval_reward:.2f})')
                break
            else:
                print('Behavior cloning target not yet met. Re-gathering data and continuing training...')

        return

    # joint metacontroller mode

    if train_joint_metacontroller:
        discovery_mod = DiscoveryModule(
            discrete_high_actions = discrete_high_actions,
            vq_codebook_size = vq_codebook_size,
            vq_decay = vq_decay,
            vq_commitment_weight = vq_commitment_weight
        ).to(device)

        if Path(discovery_evo_pt_name).exists():
            print(f'Loading existing {discovery_evo_pt_name} for Joint Metacontroller optimization...')
            discovery_mod.load_state_dict(torch.load(discovery_evo_pt_name, map_location = 'cpu', weights_only = False))
        elif Path(discovery_pt_name).exists():
            print(f'Loading existing {discovery_pt_name} for Joint Metacontroller optimization...')
            discovery_mod.load_state_dict(torch.load(discovery_pt_name, map_location = 'cpu', weights_only = False))

        ppo_agent = ActorCritic(state_dim = 8, action_dim = 4).to(device)
        ppo_expert_exists = Path(load_path).exists()
        if ppo_expert_exists:
            print(f'Loading PPO agent from {load_path} for Joint Metacontroller rollouts...')
            ppo_agent.load_state_dict(torch.load(load_path, map_location = 'cpu', weights_only = False))
            rollout_model = ppo_agent
            rollout_forward_has_cache = False
        else:
            print('PPO agent not found. Falling back to Discovery Module for rollouts.')
            rollout_model = discovery_mod
            rollout_forward_has_cache = True

        metacontroller = Metacontroller(discovery_mod = discovery_mod, condition_on_past_actions = condition_on_past_actions).to(device)
        meta_optimizer = Adam(metacontroller.parameters(), lr = 3e-4)

        metacontroller, meta_optimizer = accelerator.prepare(metacontroller, meta_optimizer)

        best_eval_reward = -float('inf')
        loop_count = 0
        total_steps = 0
        metacontroller_target = target_avg_cum_reward + margin_of_error

        while best_eval_reward < metacontroller_target and total_steps < max_discovery_steps:
            loop_count += 1
            print(f'\n--- Joint Metacontroller Training Loop {loop_count} ---')

            # gather expert trajectories
            if skip_discovery_eval and loop_count == 1:
                print("Skipping discovery eval and loading buffer...")
                buffer = ReplayBuffer.from_folder('./joint_metacontroller_buffer')
            else:
                gather_episodes = int(batch_size / filter_top_percentile)

                _, buffer = evaluate_agent(
                    model = rollout_model,
                    num_episodes = gather_episodes,
                    video_folder = 'joint_gather_videos',
                    record_every = 0,
                    device = device,
                    seed = seed + loop_count * 1000,
                    store_buffer = True,
                    buffer_folder = './joint_metacontroller_buffer',
                    render_videos = False,
                    filter_top_percentile = filter_top_percentile,
                    forward_has_cache = rollout_forward_has_cache,
                    max_timesteps = max_timesteps,
                    desc = 'Gathering Expert Trajectories'
                )

            dataloader = buffer.dataloader(batch_size = batch_size, fields = ('state', 'action'))
            dataloader = accelerator.prepare(dataloader)

            # train joint metacontroller
            pbar_meta = tqdm(range(epochs), desc = 'Training Joint Metacontroller')

            for ep in pbar_meta:
                total_loss_sum = 0.
                num_batches = 0

                metacontroller.train()

                for batch in dataloader:
                    if total_steps >= max_discovery_steps:
                        break

                    states, actions, lens = [batch[k].to(device) for k in ('state', 'action', '_lens')]

                    loss, breakdown = metacontroller(states, actions, episode_lens = lens, return_loss_breakdown = True)

                    meta_optimizer.zero_grad()
                    accelerator.backward(loss)
                    nn.utils.clip_grad_norm_(metacontroller.parameters(), 1.0)
                    meta_optimizer.step()

                    total_steps += 1
                    total_loss_sum += loss.item()
                    num_batches += 1

                    if use_wandb:
                        wandb.log(dict(metacontroller_loss = loss.item(), **breakdown))

                avg_loss = total_loss_sum / max(num_batches, 1)
                pbar_meta.set_postfix({'Loss': f'{avg_loss:.4f}'})

            unwrapped_mod = accelerator.unwrap_model(metacontroller)
            unwrapped_mod.eval()

            print('\nEvaluating Metacontroller...')
            eval_reward = evaluate_agent(
                model = unwrapped_mod,
                num_episodes = eval_episodes,
                video_folder = 'metacontroller_eval_videos',
                record_every = 5,
                device = device,
                seed = seed + loop_count * 1000,
                store_buffer = False,
                forward_has_cache = True,
                quiet = False
            )

            print('\nEvaluating Metacontroller (Reflexive Only)...')
            reflexive_eval_reward = evaluate_agent(
                model = unwrapped_mod,
                num_episodes = eval_episodes,
                video_folder = 'metacontroller_reflexive_eval_videos',
                record_every = 5,
                device = device,
                seed = seed + loop_count * 1000 + 1,
                store_buffer = False,
                forward_has_cache = True,
                quiet = False,
                force_drop_condition = True
            )

            print('\nEvaluating Metacontroller (High Action Only)...')
            high_action_only_eval_reward = evaluate_agent(
                model = unwrapped_mod,
                num_episodes = eval_episodes,
                video_folder = 'metacontroller_high_action_only_eval_videos',
                record_every = 5,
                device = device,
                seed = seed + loop_count * 1000 + 2,
                store_buffer = False,
                forward_has_cache = True,
                quiet = False,
                force_drop_state = True
            )

            if use_wandb:
                wandb.log(dict(
                    metacontroller_eval_reward = eval_reward,
                    reflexive_eval_reward = reflexive_eval_reward,
                    high_action_only_eval_reward = high_action_only_eval_reward
                ))

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                print(f'New best Metacontroller reward: {best_eval_reward:.2f}')
                unwrapped_mod.save('metacontroller_joint.pt')

            if best_eval_reward >= metacontroller_target:
                print('Metacontroller convergence target reached!')
                break

        print(f'Joint Metacontroller optimization finished! Best reward: {best_eval_reward:.2f}')

        return

    # evolutionary strategies mode

    if train_evo_strat:
        torch.set_num_threads(1)

        discovery_path = Path(discovery_pt_name)
        assert discovery_path.exists(), f"Error: {discovery_pt_name} not found. Please train the discovery module first."

        print(f'Loading existing {discovery_pt_name} for Evolutionary Strategies optimization...')
        discovery_mod = DiscoveryModule(
            discrete_high_actions = discrete_high_actions,
            vq_codebook_size = vq_codebook_size,
            vq_decay = vq_decay,
            vq_commitment_weight = vq_commitment_weight
        ).to(device)
        discovery_mod.load_state_dict(torch.load(discovery_pt_name, map_location = 'cpu', weights_only = False))

        if not skip_discovery_eval and accelerator.is_main_process:
            print('Running initial validation of discovery.pt...')
            initial_eval_reward = evaluate_agent(
                model = discovery_mod,
                num_episodes = 20,
                video_folder = 'evo_strat_initial_eval_videos',
                record_every = 0,
                device = device,
                seed = seed,
                render_videos = False,
                forward_has_cache = True,
                vectorization_mode = 'sync'
            )
            print(f'Initial DiscoveryModule Reward: {initial_eval_reward:.2f}')

        evo_strat_environment = lambda m: evaluate_agent(
            model = m, num_episodes = 1, video_folder = '', record_every = 0,
            device = device, seed = None, render_videos = False,
            forward_has_cache = True, quiet = True
        )

        target_to_params = dict(
            inner = [discovery_mod.hnet.network],
            outer = [discovery_mod.decoder1, discovery_mod.decoder2],
            both  = [discovery_mod]
        )
        assert evo_strat_target in target_to_params, f'Unknown evo_strat_target: {evo_strat_target}'
        params_to_optimize = target_to_params[evo_strat_target]

        evo = EvoStrategy(
            model = discovery_mod,
            environment = evo_strat_environment,
            num_generations = 1,
            noise_population_size = evo_strat_population_size,
            learning_rate = evo_strat_learning_rate,
            noise_scale = evo_strat_noise_scale,
            optimizer_klass = Adam,
            fitness_to_weighted_factor = 'centered_rank',
            params_to_optimize = params_to_optimize,
            cpu = cpu
        )

        best_eval_reward = -float('inf')

        for gen in range(evo_strat_generations):
            if accelerator.is_main_process:
                print(f'\n--- Evo Strat Generation {gen+1}/{evo_strat_generations} ---')
            evo()

            if accelerator.is_main_process:
                print('Evaluating deterministic performance...')
                eval_reward = evaluate_agent(
                    model = discovery_mod,
                    num_episodes = 20,
                    video_folder = 'evo_strat_eval_videos',
                    record_every = 5,
                    device = device,
                    seed = seed + gen * 1000,
                    store_buffer = False,
                    forward_has_cache = True,
                    render_videos = True,
                    quiet = False
                )

                if use_wandb:
                    wandb.log(dict(
                        evo_strat_eval_reward = eval_reward,
                        evo_strat_generation = gen
                    ))

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    print(f'New best Evo Strat reward: {best_eval_reward:.2f}')
                    torch.save(discovery_mod.state_dict(), discovery_evo_pt_name)

                if best_eval_reward >= 0:
                    print(f'Evo Strat reached target reward >= 0! (best: {best_eval_reward:.2f})')

            accelerator.wait_for_everyone()

        return

    # joint metacontroller evolutionary strategies mode

    if train_evo_strat_joint:
        torch.set_num_threads(1)

        joint_pt_name = 'metacontroller_joint.pt'
        assert Path(joint_pt_name).exists(), f"Error: {joint_pt_name} not found. Please train the joint metacontroller first."

        print(f'Loading existing {joint_pt_name} for Joint ES optimization...')
        metacontroller = Metacontroller.init_and_load(joint_pt_name).to(device)

        if accelerator.is_main_process:
            print('Running initial validation of metacontroller_joint.pt...')
            initial_eval_reward = evaluate_agent(
                model = metacontroller,
                num_episodes = 20,
                video_folder = 'evo_strat_joint_initial_eval_videos',
                record_every = 0,
                device = device,
                seed = seed,
                render_videos = False,
                forward_has_cache = True,
                vectorization_mode = 'sync'
            )
            print(f'Initial Joint Metacontroller Reward: {initial_eval_reward:.2f}')

        evo_strat_joint_environment = lambda m: evaluate_agent(
            model = m, num_episodes = 1, video_folder = '', record_every = 0,
            device = device, seed = None, render_videos = False,
            forward_has_cache = True, quiet = True
        )

        joint_target_to_params = dict(
            higher = [metacontroller.metacontroller],
            lower  = [metacontroller.lower_controller],
            both   = [metacontroller]
        )
        assert evo_strat_joint_target in joint_target_to_params, f'Unknown evo_strat_joint_target: {evo_strat_joint_target}'
        joint_params_to_optimize = joint_target_to_params[evo_strat_joint_target]

        evo_joint = EvoStrategy(
            model = metacontroller,
            environment = evo_strat_joint_environment,
            num_generations = 1,
            noise_population_size = evo_strat_population_size,
            learning_rate = evo_strat_learning_rate,
            noise_scale = evo_strat_noise_scale,
            optimizer_klass = Adam,
            fitness_to_weighted_factor = 'centered_rank',
            params_to_optimize = joint_params_to_optimize,
            cpu = cpu
        )

        best_eval_reward = -float('inf')
        joint_evo_pt_name = 'metacontroller_joint_evo_strat.pt'

        for gen in range(evo_strat_generations):
            if accelerator.is_main_process:
                print(f'\n--- Joint Evo Strat Generation {gen+1}/{evo_strat_generations} ---')
            evo_joint()

            if accelerator.is_main_process:
                print('Evaluating deterministic performance...')
                eval_reward = evaluate_agent(
                    model = metacontroller,
                    num_episodes = 20,
                    video_folder = 'evo_strat_joint_eval_videos',
                    record_every = 5,
                    device = device,
                    seed = seed + gen * 1000,
                    store_buffer = False,
                    forward_has_cache = True,
                    render_videos = True,
                    quiet = False
                )

                if use_wandb:
                    wandb.log(dict(
                        evo_strat_joint_eval_reward = eval_reward,
                        evo_strat_joint_generation = gen
                    ))

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    print(f'New best Joint Evo Strat reward: {best_eval_reward:.2f}')
                    torch.save(metacontroller.state_dict(), joint_evo_pt_name)

                if best_eval_reward >= target_avg_cum_reward + margin_of_error:
                    print(f'Joint Evo Strat reached target reward >= {target_avg_cum_reward + margin_of_error:.2f}! (best: {best_eval_reward:.2f})')
                    break

            accelerator.wait_for_everyone()

        return

    # joint ppo training mode

    if train_ppo_joint:
        model.load_state_dict(torch.load(load_path, map_location = 'cpu', weights_only = False))

        joint_pt_name = 'metacontroller_joint.pt'

        if Path(joint_pt_name).exists():
            metacontroller = Metacontroller.init_and_load(joint_pt_name, strict = False).to(device)
        else:
            discovery_mod = DiscoveryModule(
                discrete_high_actions = discrete_high_actions,
                vq_codebook_size = vq_codebook_size,
                vq_decay = vq_decay,
                vq_commitment_weight = vq_commitment_weight
            ).to(device)

            if Path(discovery_pt_name).exists():
                discovery_mod.load_state_dict(torch.load(discovery_pt_name, map_location = 'cpu', weights_only = False))

            metacontroller = Metacontroller(
                discovery_mod,
                condition_on_past_actions = condition_on_past_actions
            ).to(device)

        meta_optimizer = Adam(metacontroller.parameters(), lr = lr)
        metacontroller, meta_optimizer = accelerator.prepare(metacontroller, meta_optimizer)

        recent_rewards = deque(maxlen = rolling_window)
        pbar = tqdm(range(ppo_joint_total_episodes), desc = 'Joint PPO')

        buffer_dir = Path('./ppo_joint_buffer')
        buffer_dir.mkdir(exist_ok = True)

        buffer = ReplayBuffer(
            folder = buffer_dir,
            max_episodes = ppo_joint_update_every * 2,
            max_timesteps = max_timesteps,
            circular = True,
            fields = dict(
                state = ('float', (8,)),
                action = ('int', ()),
                log_prob = ('float', ()),
                advantage = ('float', ()),
                return_ = ('float', ())
            )
        )

        for ep in pbar:
            state, _ = env.reset(seed = seed + ep)

            ep_states, ep_actions, ep_log_probs = [], [], []
            ep_rewards, ep_values, ep_dones = [], [], []

            ep_reward_sum = 0.
            done = False
            cache = None

            metacontroller.eval()

            with torch.no_grad():
                for step in range(max_timesteps):
                    if done:
                        break

                    state_tensor = rearrange(tensor(state, dtype = torch.float32, device = device), 'd -> 1 d')

                    fine_action, _, _, cache, high_action, high_log_prob, value = metacontroller.get_action_and_value(
                        state_tensor, cache = cache
                    )

                    next_state, reward, terminated, truncated, _ = env.step(fine_action.item())
                    done = terminated or truncated

                    ep_states.append(state)
                    ep_actions.append(high_action.item())
                    ep_log_probs.append(high_log_prob.item())
                    ep_rewards.append(reward)
                    ep_values.append(value.item())
                    ep_dones.append(done)

                    state = next_state
                    ep_reward_sum += reward

            step_count = len(ep_states)

            recent_rewards.append(ep_reward_sum)
            avg_reward = np.mean(recent_rewards)
            pbar.set_postfix({'avg': f'{avg_reward:.2f}', 'ep': f'{ep_reward_sum:.2f}'})

            if use_wandb:
                log_dict = dict(
                    episode = ep,
                    reward = ep_reward_sum,
                    avg_reward_last_20 = avg_reward,
                    step_count = step_count
                )

                if record_video_every > 0 and ep % record_video_every == 0:
                    video_path = f'videos/rl-video-episode-{ep}.mp4'
                    if os.path.exists(video_path):
                        log_dict['video'] = wandb.Video(video_path)

                wandb.log(log_dict)

            if avg_reward >= ppo_joint_target_avg_cum_reward and len(recent_rewards) == rolling_window:
                save_path = 'metacontroller_joint_ppo.pt'
                torch.save(metacontroller.state_dict(), save_path)
                print(f'target reward reached ({ppo_joint_target_avg_cum_reward}). saved to {save_path}')
                break

            # gae

            rewards_t = tensor(ep_rewards, dtype = torch.float32, device = device)
            values_t = tensor(ep_values, dtype = torch.float32, device = device)
            masks_t = 1. - tensor(ep_dones, dtype = torch.float32, device = device)

            returns_t = calc_gae(rewards_t, values_t, masks_t, gamma, gae_lambda)
            advantages_t = returns_t - values_t

            with buffer.one_episode():
                for t in range(step_count):
                    buffer.store(
                        state = ep_states[t],
                        action = ep_actions[t],
                        log_prob = ep_log_probs[t],
                        advantage = advantages_t[t].item(),
                        return_ = returns_t[t].item()
                    )

            # ppo update

            if (ep + 1) % ppo_joint_update_every != 0:
                continue

            metacontroller.train()

            dataloader = buffer.dataloader(
                batch_size = batch_size,
                fields = ('state', 'action', 'log_prob', 'advantage', 'return_')
            )

            for _ in range(ppo_joint_epochs):
                for batch in dataloader:
                    states, actions, old_log_probs, advantages, returns = [
                        batch[k].to(device) for k in ('state', 'action', 'log_prob', 'advantage', 'return_')
                    ]

                    # flatten trajectory-level to timestep-level, masking padding

                    if states.ndim == 3:
                        lens = batch.get('_lens', None)
                        b, s, d = states.shape

                        flat_mask = None
                        if exists(lens):
                            mask = lens_to_mask(lens.to(device), max_len = s)
                            flat_mask = rearrange(mask, 'b s -> (b s)')

                        states, actions, old_log_probs, advantages, returns = [
                            rearrange(t, 'b s ... -> (b s) ...') for t in (states, actions, old_log_probs, advantages, returns)
                        ]

                        if exists(flat_mask):
                            states, actions, old_log_probs, advantages, returns = [
                                t[flat_mask] for t in (states, actions, old_log_probs, advantages, returns)
                            ]

                    if advantages.numel() > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # fresh high-level logits from state projection (no KV cache)

                    state_repr = metacontroller.state_proj(states)
                    dec_out = metacontroller.metacontroller(rearrange(state_repr, 'n d -> n 1 d'))
                    dec_out = rearrange(dec_out, 'n 1 d -> n d')

                    logits = metacontroller.high_action_readout(dec_out)
                    probs = torch.distributions.Categorical(logits = logits)
                    new_log_probs = probs.log_prob(actions.long())
                    entropy = probs.entropy().mean()

                    critic_logits = metacontroller.critic_mlp(dec_out)

                    delight_gate = (-new_log_probs * advantages).sigmoid().detach()
                    gated_advantages = advantages * delight_gate

                    ratio = (new_log_probs - old_log_probs).exp()
                    policy_loss = torch.max(
                        -gated_advantages * ratio,
                        -gated_advantages * ratio.clamp(1 - clip_coef, 1 + clip_coef)
                    ).mean()

                    value_loss = metacontroller.hl_gauss_loss(critic_logits, returns)
                    loss = policy_loss - entropy_coef * entropy + value_coef * value_loss

                    meta_optimizer.zero_grad()
                    accelerator.backward(loss)
                    nn.utils.clip_grad_norm_(metacontroller.parameters(), 0.5)
                    meta_optimizer.step()

                    if use_wandb:
                        wandb.log(dict(
                            ppo_joint_loss = loss.item(),
                            ppo_joint_policy_loss = policy_loss.item(),
                            ppo_joint_value_loss = value_loss.item(),
                            ppo_joint_entropy = entropy.item()
                        ))

        return

    # ppo training mode

    recent_rewards = deque(maxlen = rolling_window)
    pbar = tqdm(range(total_episodes), desc = 'Training PPO')

    buffer_dir = Path('./ppo_buffer')
    buffer_dir.mkdir(exist_ok = True)

    buffer = ReplayBuffer(
        folder = buffer_dir,
        max_episodes = 25,
        max_timesteps = 1000,
        circular = True,
        fields = dict(
            state = ('float', (8,)),
            action = ('int', ()),
            log_prob = ('float', ()),
            advantage = ('float', ()),
            return_ = ('float', ())
        )
    )

    for ep in pbar:
        state, _ = env.reset(seed = seed + ep)

        ep_states = []
        ep_actions = []
        ep_log_probs = []
        ep_rewards = []
        ep_values = []
        ep_dones = []

        ep_reward_sum = 0.
        done = False
        step_count = 0

        model.eval()

        with torch.no_grad():
            while not done and step_count < 1000:
                state_tensor = rearrange(tensor(state, dtype = torch.float32, device = device), 'd -> 1 d')
                action, log_prob, _, value, _ = model.get_action_and_value(state_tensor)

                action_item = action.item()
                next_state, reward, terminated, truncated, _ = env.step(action_item)
                done = terminated or truncated

                ep_states.append(state)
                ep_actions.append(action_item)
                ep_log_probs.append(log_prob.item())
                ep_rewards.append(reward)
                ep_values.append(value.item())
                ep_dones.append(done)

                state = next_state
                ep_reward_sum += reward
                step_count += 1

        recent_rewards.append(ep_reward_sum)
        avg_reward = np.mean(recent_rewards)
        pbar.set_postfix({'Avg Reward': f'{avg_reward:.2f}', 'Ep Reward': f'{ep_reward_sum:.2f}'})

        if use_wandb:
            log_dict = dict(
                episode = ep,
                reward = ep_reward_sum,
                avg_reward_last_20 = avg_reward,
                step_count = step_count
            )

            if record_video_every > 0 and ep % record_video_every == 0:
                video_path = f'videos/rl-video-episode-{ep}.mp4'
                if os.path.exists(video_path):
                    log_dict['video'] = wandb.Video(video_path)

            wandb.log(log_dict)

        if avg_reward >= target_avg_cum_reward and len(recent_rewards) == rolling_window:
            save_path = Path('./metacontroller_agent.pt')
            torch.save(model.state_dict(), save_path)
            print(f'Target average cumulative reward ({target_avg_cum_reward}) reached! Model saved to {save_path.absolute()}')
            break

        # compute gae and store episode

        rewards_t = tensor(ep_rewards, dtype = torch.float32, device = device)
        values_t = tensor(ep_values, dtype = torch.float32, device = device)
        masks_t = 1.0 - tensor(ep_dones, dtype = torch.float32, device = device)

        returns_t = calc_gae(rewards_t, values_t, masks_t, gamma, gae_lambda)
        advantages_t = returns_t - values_t

        with buffer.one_episode():
            for t in range(step_count):
                buffer.store(
                    state = ep_states[t],
                    action = ep_actions[t],
                    log_prob = ep_log_probs[t],
                    advantage = advantages_t[t].item(),
                    return_ = returns_t[t].item()
                )

        # ppo update

        model.train()

        dataloader = buffer.dataloader(
            batch_size = batch_size,
            fields = ('state', 'action', 'log_prob', 'advantage', 'return_')
        )

        for _ in range(epochs):
            for batch in dataloader:
                states, actions, old_log_probs, batch_advantages, batch_returns = [
                    batch[k].to(device) for k in ('state', 'action', 'log_prob', 'advantage', 'return_')
                ]

                if batch_advantages.numel() > 1:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                _, new_log_probs, entropy, _, critic_logits = model.get_action_and_value(states, actions)

                ratio = (new_log_probs - old_log_probs).exp()

                policy_loss1 = -batch_advantages * ratio
                policy_loss2 = -batch_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                value_loss = model.hl_gauss_loss(critic_logits, batch_returns)
                entropy_loss = entropy.mean()

                loss = policy_loss - entropy_coef * entropy_loss + value_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

if __name__ == '__main__':
    fire.Fire(train_metacontroller)
