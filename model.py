import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from dataclasses import dataclass


@dataclass
class ModelArgs:
    vocab_size: int = -1 
    embedding_dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    block_size: int = 256
    dropout: float = 0.2

    device: str = None


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.embedding_dim = args.embedding_dim
        self.n_heads = args.n_heads
        self.attn = nn.Linear(self.embedding_dim, 3 * self.embedding_dim, bias=False)
        self.proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.attn_dropout = args.dropout
        self.resid_dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.attn(x).split(self.embedding_dim, dim=-1)

        q = q.view(B, T, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)
        k = k.view(B, T, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)
        v = v.view(B, T, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)

        dropout_p = self.attn_dropout if self.training else 0.0

        y = F.scaled_dot_product_attention(
            query=q, key=k, value=v, is_causal=True, dropout_p=dropout_p
        )

        y = y.transpose(1, 2).contiguous().view(B, T, self.embedding_dim)
        y = self.resid_dropout(self.proj(y))

        return y


class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.ln1 = nn.LayerNorm(args.embedding_dim)
        self.ln2 = nn.LayerNorm(args.embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(args.embedding_dim, 4 * args.embedding_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * args.embedding_dim, args.embedding_dim, bias=False),
            nn.Dropout(args.dropout),
        )

        self.attn = Attention(args=args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        args: ModelArgs = None,
    ):
        super().__init__()

        self.args = args

        self.embed = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.pos_embed = nn.Embedding(args.block_size, args.embedding_dim)
        self.block_size = args.block_size

        self.layers = nn.ModuleList(
            [DecoderBlock(args=args) for _ in range(args.n_layers)]
        )

        self.ln_f = nn.LayerNorm(args.embedding_dim)
        self.lm_head = nn.Linear(args.embedding_dim, args.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, target=None) -> Tuple[torch.Tensor, float]:
        B, T = tokens.shape

        tokens_embedding = self.embed(tokens)
        tokens_pos_embedding = self.pos_embed(torch.arange(T, device=tokens.device))

        x = tokens_embedding + tokens_pos_embedding

        for layer in self.layers:
            x = layer(x)

        logits = self.lm_head(self.ln_f(x))  # norm -> mlp

        loss = None
        if target is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = target.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return (logits, loss)

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = None,
        do_sample: bool = False,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            tokens_crop = (
                tokens
                if tokens.size(1) <= self.block_size
                else tokens[:, -self.block_size :]
            )

            logits, _ = self(tokens_crop)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -torch.inf

            probs = F.softmax(logits, dim=-1)

            if do_sample:
                token_next = torch.multinomial(probs, num_samples=1)
            else:
                _, token_next = torch.topk(probs, k=1, dim=-1)

            tokens = torch.cat((tokens, token_next), dim=1)

        return tokens
