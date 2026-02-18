"""Gated Transformer-XL (GTrXL) for Reinforcement Learning.

Based on "Stabilizing Transformers for Reinforcement Learning" (Parisotto et al., 2020).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_ff_activation(name):
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported ff_activation '{name}'. Expected one of: relu, gelu")


class GRUGate(nn.Module):
    """GRU-style gating mechanism for residual connections."""

    def __init__(self, d_model, gate_bias=2.0):
        super().__init__()
        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.U_r = nn.Linear(d_model, d_model, bias=False)
        self.W_z = nn.Linear(d_model, d_model, bias=False)
        self.U_z = nn.Linear(d_model, d_model, bias=False)
        self.W_g = nn.Linear(d_model, d_model, bias=False)
        self.U_g = nn.Linear(d_model, d_model, bias=False)
        self.bias = nn.Parameter(torch.full((d_model,), gate_bias))

    def forward(self, x, y):
        r = torch.sigmoid(self.W_r(y) + self.U_r(x))
        z = torch.sigmoid(self.W_z(y) + self.U_z(x) - self.bias)
        h_hat = torch.tanh(self.W_g(y) + self.U_g(r * x))
        return (1.0 - z) * x + z * h_hat


class RelativePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used for relative attention."""

    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, length):
        if length > self.pe.shape[0]:
            raise ValueError(
                f"Requested relative length {length}, but max_len is {self.pe.shape[0]}"
            )
        return self.pe[:length]


class RelativeMultiHeadAttention(nn.Module):
    """Transformer-XL relative multi-head attention.

    Implements Eq. (16)-(20) from Appendix C.2 in the GTrXL paper.
    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.r_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Global content and positional biases from Transformer-XL.
        self.u_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.v_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, memory=None, rel_pos=None, attn_mask=None):
        """
        Args:
            x: Current segment states, shape (seq_len, batch, d_model)
            memory: Previous segment states, shape (mem_len, batch, d_model) or None
            rel_pos: Relative positional encodings, shape (mem_len + seq_len, d_model)
            attn_mask: Bool mask, shape (batch, seq_len, mem_len + seq_len), where True means masked
        """
        seq_len, batch_size, _ = x.shape

        if memory is not None:
            # StopGrad(M^{(l-1)}) in Eq. (23).
            mem = memory.detach()
            kv_input = torch.cat([mem, x], dim=0)
            mem_len = mem.shape[0]
        else:
            kv_input = x
            mem_len = 0

        key_len = kv_input.shape[0]
        if rel_pos is None:
            raise ValueError("rel_pos is required for relative attention")
        if rel_pos.shape[0] < key_len:
            raise ValueError(
                f"rel_pos length {rel_pos.shape[0]} must be >= key_len {key_len}"
            )

        # Project Q, K, V (Eq. 16).
        q = self.q_proj(x)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        q = q.view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(key_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(key_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # Relative keys R = W_R Phi (Eq. 17).
        r = self.r_proj(rel_pos[:key_len])
        r = r.view(key_len, self.num_heads, self.head_dim)

        # Relative index for each (query_t, key_m): distance = (mem_len + t) - m.
        q_pos = torch.arange(seq_len, device=x.device).unsqueeze(1) + mem_len
        k_pos = torch.arange(key_len, device=x.device).unsqueeze(0)
        rel_idx = (q_pos - k_pos).clamp(min=0, max=key_len - 1)
        r_qk = r[rel_idx]  # (seq_len, key_len, num_heads, head_dim)

        # Eq. (18): QK + QR + uK + vR
        qk_term = torch.einsum("bhqd,bhkd->bhqk", q, k)
        qr_term = torch.einsum("bhqd,qkhd->bhqk", q, r_qk)
        uk_term = torch.einsum("hd,bhkd->bhk", self.u_bias, k).unsqueeze(2)
        vr_term = torch.einsum("hd,qkhd->hqk", self.v_bias, r_qk).unsqueeze(0)

        scores = (qk_term + qr_term + uk_term + vr_term) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                mask = attn_mask
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                if mask.dim() != 3:
                    raise ValueError("attn_mask must have shape (batch, seq, key) or (seq, key)")
                scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
            else:
                # Optional additive mask path for compatibility with float masks.
                add_mask = attn_mask
                if add_mask.dim() == 2:
                    add_mask = add_mask.unsqueeze(0)
                if add_mask.dim() != 3:
                    raise ValueError("attn_mask must have shape (batch, seq, key) or (seq, key)")
                scores = scores + add_mask.unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        y = torch.einsum("bhqk,bhkd->bhqd", weights, v)
        y = y.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size, self.d_model)
        y = self.o_proj(y)
        y = self.out_dropout(y)
        return y


class GTrXLLayer(nn.Module):
    """Single GTrXL layer with pre-LN, relative attention, and GRU gating."""

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff=None,
        dropout=0.0,
        gate_bias=2.0,
        ff_activation="relu",
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)

        self.attn = RelativeMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            _build_ff_activation(ff_activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.gate_attn = GRUGate(d_model, gate_bias)
        self.gate_ff = GRUGate(d_model, gate_bias)

    def forward(self, x, memory=None, rel_pos=None, attn_mask=None):
        # Eq. (23): relative attention on LayerNorm([StopGrad(M), E])
        x_norm = self.norm_attn(x)
        mem_norm = self.norm_attn(memory) if memory is not None else None
        attn_out = self.attn(x_norm, memory=mem_norm, rel_pos=rel_pos, attn_mask=attn_mask)

        # Eq. (24): ReLU before residual/gate path.
        h = self.gate_attn(x, F.relu(attn_out))

        # Eq. (25)-(26): FF on LayerNorm, then ReLU before residual/gate path.
        ff_out = self.ff(self.norm_ff(h))
        out = self.gate_ff(h, F.relu(ff_out))
        return out


class GTrXL(nn.Module):
    """Gated Transformer-XL (GTrXL) module."""

    def __init__(
        self,
        input_dim,
        d_model=256,
        num_layers=3,
        num_heads=8,
        d_ff=None,
        memory_length=64,
        dropout=0.0,
        gate_bias=2.0,
        ff_activation="relu",
        max_seq_len=2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.memory_length = memory_length

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = RelativePositionalEncoding(d_model, max_len=max_seq_len)

        self.layers = nn.ModuleList([
            GTrXLLayer(d_model, num_heads, d_ff, dropout, gate_bias, ff_activation)
            for _ in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for layer in self.layers:
            for proj in [
                layer.attn.q_proj,
                layer.attn.k_proj,
                layer.attn.v_proj,
                layer.attn.r_proj,
                layer.attn.o_proj,
            ]:
                nn.init.xavier_uniform_(proj.weight)

            for module in layer.ff:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def init_memory(self, batch_size, device=None):
        return [
            torch.zeros(self.memory_length, batch_size, self.d_model, device=device)
            for _ in range(self.num_layers)
        ]

    def _build_attention_mask(self, seq_len, memory, done_masks):
        """Build per-batch attention masks with done-boundary isolation.

        Returns:
            Bool mask of shape (batch, seq_len, mem_len + seq_len), where True means masked.
        """
        mem_len, batch_size = memory.shape[0], memory.shape[1]
        device = memory.device

        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

        if done_masks is None:
            allow_curr = causal.unsqueeze(-1).expand(seq_len, seq_len, batch_size)
            if mem_len > 0:
                allow_mem = torch.ones(seq_len, mem_len, batch_size, dtype=torch.bool, device=device)
            else:
                allow_mem = torch.zeros(seq_len, 0, batch_size, dtype=torch.bool, device=device)
        else:
            done = done_masks.to(device).bool()
            if done.dim() != 2:
                raise ValueError("done_masks must have shape (seq_len, batch)")

            # Episode id before each step's transition.
            episode_id = done.cumsum(dim=0) - done.to(torch.int64)

            # Current segment: attend only within same episode and causally.
            same_episode = episode_id.unsqueeze(1) == episode_id.unsqueeze(0)  # (seq, seq, batch)
            allow_curr = same_episode & causal.unsqueeze(-1)

            # Memory is only available before the first done in this segment.
            if mem_len > 0:
                allow_mem = (episode_id == 0).unsqueeze(1).expand(seq_len, mem_len, batch_size)
            else:
                allow_mem = torch.zeros(seq_len, 0, batch_size, dtype=torch.bool, device=device)

        allow = torch.cat([allow_mem, allow_curr], dim=1)  # (seq, key, batch)
        return ~allow.permute(2, 0, 1).contiguous()  # (batch, seq, key)

    def _update_memory(self, memory, layer_input, done_masks):
        """Update memory while dropping context before the latest done per env."""
        combined = torch.cat([memory, layer_input], dim=0)

        if done_masks is not None:
            done = done_masks.to(combined.device).bool()
            if done.dim() != 2:
                raise ValueError("done_masks must have shape (seq_len, batch)")

            if done.any():
                seq_len, batch_size = done.shape
                idx = torch.arange(seq_len, device=combined.device).unsqueeze(1).expand(seq_len, batch_size)
                last_done = torch.where(done, idx, torch.full_like(idx, -1)).max(dim=0).values

                # Keep only positions strictly after the latest done step.
                keep_from = memory.shape[0] + last_done + 1  # (batch,)
                pos = torch.arange(combined.shape[0], device=combined.device).unsqueeze(1)
                keep = pos >= keep_from.unsqueeze(0)
                combined = combined * keep.unsqueeze(-1)

        return combined[-self.memory_length :].detach()

    def forward(self, x, memory=None, done_masks=None):
        """
        Args:
            x: input tensor, shape (seq_len, batch, input_dim)
            memory: list of num_layers tensors each (mem_len, batch, d_model), or None
            done_masks: episode termination masks, shape (seq_len, batch) or None
        Returns:
            output: shape (seq_len, batch, d_model)
            new_memory: list of num_layers tensors each (mem_len, batch, d_model)
        """
        seq_len, batch_size = x.shape[0], x.shape[1]

        if memory is None:
            memory = self.init_memory(batch_size, device=x.device)

        mem_len = memory[0].shape[0]
        rel_pos = self.pos_enc(seq_len + mem_len)

        h = self.input_proj(x)

        new_memory = []
        for layer_idx, layer in enumerate(self.layers):
            layer_mem = memory[layer_idx]
            layer_input = h

            attn_mask = self._build_attention_mask(seq_len, layer_mem, done_masks)
            h = layer(
                layer_input,
                memory=layer_mem,
                rel_pos=rel_pos,
                attn_mask=attn_mask,
            )

            new_memory.append(self._update_memory(layer_mem, layer_input, done_masks))

        return h, new_memory
