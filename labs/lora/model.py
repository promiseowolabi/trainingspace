"""Lab 12 — the model (given; do not edit). The same GPT as lab 07.

Lab 06's GPT-2 architecture in PyTorch: learned positions, pre-norm
LayerNorm, causal multi-head attention, a GELU MLP and tied embeddings.
PyTorch's autograd replaces your lab 02 engine; the forward pass is the one
you wrote in numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n_head = c.n_head
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd)
        self.fc_proj = nn.Linear(4 * c.n_embd, c.n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.fc_proj(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.config = c
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2: scale residual projections (O1.L1)
            if name.endswith(("proj.weight", "fc_proj.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.ln_f(x) @ self.wte.weight.T  # tied output layer (M1.L2)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0, generator=None):
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.config.block_size:])
            probs = F.softmax(logits[:, -1].float() / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1, generator=generator)], dim=1)
        return idx
