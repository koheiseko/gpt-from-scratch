import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(
        self, embedding_dim: int, block_size: int, n_heads: int, dropout: float = 0.1
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.attn = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x: torch.tensor):
        B, T, C = x.shape

        q, k, v = self.attn(x).split(self.embedding_dim, dim=-1)

        q = q.view(B, T, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)
        k = k.view(B, T, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)
        v = v.view(B, T, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)

        y = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, self.embedding_dim)
        y = self.resid_dropout(self.proj(y))

        return y


class DecoderBlock(nn.Module):
    def __init__(
        self, embedding_dim: int, block_size, n_heads: int, dropout: int = 0.1
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

        self.attn = CausalSelfAttention(
            embedding_dim=embedding_dim,
            block_size=block_size,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        embedding_dim: int,
        n_heads: int,
        block_size: int,
        dropout: int = 0.1,
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(block_size, embedding_dim)
        self.block_size = block_size

        self.blocks = nn.Sequential(
            *[
                DecoderBlock(
                    embedding_dim=embedding_dim,
                    block_size=block_size,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        B, T = idx.shape

        idx_embedding = self.embed(idx)
        idx_pos_embedding = self.pos_embed(torch.arange(T, device=idx.device))

        x = idx_embedding + idx_pos_embedding

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if target is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = target.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature=0.8, top_k=None, do_sample=False
    ):
        for _ in range(max_new_tokens):
            idx_crop = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )

            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -torch.inf

            probs = F.softmax(logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            idx = torch.concat((idx, idx_next), dim=1)

        return idx
