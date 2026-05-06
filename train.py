# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "einops",
#     "x-transformers",
#     "accelerate",
#     "tqdm",
#     "numpy",
#     "vector-quantize-pytorch",
#     "torch-einops-utils",
#     "fire",
#     "assoc-scan",
# ]
# ///

import fire

import argparse
import math
import gzip
import random
import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator

from einops import rearrange

from h_net_dynamic_chunking import HNet
from x_transformers import Decoder

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

class HNetLM(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        dim_inner,
        enc_depth,
        depth,
        dec_depth,
        max_seq_len = 512,
        dim_head = 64,
        heads = 8,
        transformer_kwargs = dict()
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        encoder = Decoder(
            dim = dim,
            depth = enc_depth,
            attn_dim_head = dim_head,
            heads = heads,
            **transformer_kwargs
        )

        network = Decoder(
            dim = dim_inner,
            depth = depth,
            attn_dim_head = dim_head,
            heads = heads,
            **transformer_kwargs
        )

        decoder = Decoder(
            dim = dim,
            depth = dec_depth,
            attn_dim_head = dim_head,
            heads = heads,
            **transformer_kwargs
        )

        self.hnet = HNet(
            encoder,
            network,
            decoder,
            dim = dim,
            dim_inner = dim_inner
        )

        self.to_logits = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

    def sample(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.,
        filter_thres = 0.9,
    ):
        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        cache = None

        for _ in range(sample_num_times):
            if not exists(cache):
                logits, cache = self.forward(out, return_hiddens = True)
            else:
                logits, cache = self.forward(out[:, -1:], return_hiddens = True, cache = cache, start_pos = out.shape[-1] - 1)

            logits = logits[:, -1]

            logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(logits, temperature = temperature, dim = -1)

            out = torch.cat((out, sample), dim = -1)

        return out[..., prompt_seq_len:]

    def forward(self, x, return_loss = False, return_hiddens = False, cache = None, start_pos = 0):

        if return_loss:
            x, target = x[:, :-1], x[:, 1:]

        seq_len, device = x.shape[-1], x.device

        tokens = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(start_pos, start_pos + seq_len, device = device))

        tokens = tokens + pos_emb

        out = self.hnet(tokens, return_hiddens = return_hiddens, cache = cache)

        embed = out.output
        aux_loss = out.loss

        logits = self.to_logits(embed)

        if not return_loss:
            if return_hiddens:
                return logits, out.next_cache

            return logits

        ar_loss =  F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            target
        )

        return ar_loss + aux_loss, (ar_loss, aux_loss)

def main(
    cpu = False,
    num_batches = int(1e5),
    batch_size = 4,
    grad_accum_every = 4,
    learning_rate = 1e-4,
    validate_every = 100,
    prime_length = 32,
    generate_every = 500,
    generate_length = 256,
    seq_len = 512
):
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum_every,
        cpu=cpu
    )
    device = accelerator.device

    model = HNetLM(
        num_tokens = 256,
        dim = 256,
        dim_inner = 512,
        enc_depth = 3,
        depth = 2,
        dec_depth = 3,
        max_seq_len = max(seq_len, generate_length),
    )

    # prepare enwik8 data

    with gzip.open("./data/enwik8.gz") as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        np_train, np_valid = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data = data
            self.seq_len = seq_len

        def __len__(self):
            return self.data.size(0) // self.seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
            full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
            return full_seq

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset = TextSamplerDataset(data_val, seq_len)
    train_loader = DataLoader(train_dataset, batch_size = batch_size)
    val_loader = DataLoader(val_dataset, batch_size = batch_size)

    # optimizer

    optim = Adam(model.parameters(), lr = learning_rate)

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    train_loader = cycle(train_loader)
    val_loader = cycle(val_loader)

    # training

    for i in tqdm.tqdm(range(num_batches), mininterval = 10.0, desc = "training"):
        model.train()

        for _ in range(grad_accum_every):
            with accelerator.accumulate(model):
                data = next(train_loader)

                loss, (ar_loss, aux_loss) = model(data, return_loss = True)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 0.5)

                optim.step()
                optim.zero_grad()

        accelerator.print(f"training loss: {ar_loss.item():.3f}")

        if divisible_by(i, validate_every):
            model.eval()
            with torch.no_grad():
                valid_data = next(val_loader)

                loss, (ar_loss, aux_loss) = model(valid_data, return_loss = True)
                accelerator.print(f"validation loss: {ar_loss.item():.3f}")

        if divisible_by(i, generate_every):
            model.eval()

            inp = random.choice(val_dataset)[:prime_length]
            inp = inp.to(device)

            prime = decode_tokens(inp)
            accelerator.print(f"\n\nINPUT: {prime}")

            prompt = inp[None, ...]

            # sample from unwrapped model since it has the .sample method
            sampled = accelerator.unwrap_model(model).sample(prompt, generate_length)

            base_decode_output = decode_tokens(sampled[0])

            accelerator.print(f"\nOUTPUT: {base_decode_output}\n\n")

if __name__ == '__main__':
    fire.Fire(main)
