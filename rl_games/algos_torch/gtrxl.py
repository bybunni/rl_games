"""
Gated Transformer-XL (GTrXL) for Reinforcement Learning.

Based on "Stabilizing Transformers for Reinforcement Learning" (Parisotto et al., 2020).
Key innovations over vanilla Transformer:
  - Pre-layer normalization (Identity Map Reordering)
  - GRU-style gating on residual connections with identity initialization
  - Segment-level recurrence from Transformer-XL for extended context
  - Relative positional encodings
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUGate(nn.Module):
    """GRU-style gating mechanism for residual connections.

    Replaces standard residual connection (y = x + sublayer(x)) with a gated version.
    The gate bias is initialized to a positive value (default 2.0) so the gate starts
    near-closed, making the network approximately identity/Markovian at initialization.

    Gate equations:
        r = sigmoid(W_r * y + U_r * x)
        z = sigmoid(W_z * y + U_z * x - b_g)
        h_hat = tanh(W_g * y + U_g * (r * x))
        output = (1 - z) * x + z * h_hat

    Where x is the residual input and y is the sublayer output.
    """

    def __init__(self, d_model, gate_bias=2.0):
        super().__init__()
        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.U_r = nn.Linear(d_model, d_model, bias=False)
        self.W_z = nn.Linear(d_model, d_model, bias=False)
        self.U_z = nn.Linear(d_model, d_model, bias=False)
        self.W_g = nn.Linear(d_model, d_model, bias=False)
        self.U_g = nn.Linear(d_model, d_model, bias=False)

        # Gate bias initialized positive so z starts near 0 => output ≈ x (identity)
        self.bias = nn.Parameter(torch.full((d_model,), gate_bias))

    def forward(self, x, y):
        """
        Args:
            x: residual input, shape (..., d_model)
            y: sublayer output, shape (..., d_model)
        Returns:
            gated output, shape (..., d_model)
        """
        r = torch.sigmoid(self.W_r(y) + self.U_r(x))
        z = torch.sigmoid(self.W_z(y) + self.U_z(x) - self.bias)
        h_hat = torch.tanh(self.W_g(y) + self.U_g(r * x))
        return (1 - z) * x + z * h_hat


class RelativePositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding from Transformer-XL.

    Generates position encodings for relative positions, enabling
    generalization beyond training sequence lengths.
    """

    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        """Return positional encodings for positions 0..seq_len-1.

        Args:
            seq_len: number of positions
        Returns:
            Tensor of shape (seq_len, 1, d_model)
        """
        return self.pe[:seq_len].unsqueeze(1)


class GTrXLLayer(nn.Module):
    """Single GTrXL Transformer layer with pre-LN and GRU gating.

    Architecture (pre-layer normalization):
        x -> LayerNorm -> MultiHeadAttention -> GRUGate(x, attn_out) -> residual_1
        residual_1 -> LayerNorm -> FFN -> GRUGate(residual_1, ffn_out) -> output
    """

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.0, gate_bias=2.0):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.gate_attn = GRUGate(d_model, gate_bias)
        self.gate_ff = GRUGate(d_model, gate_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory=None, pos_enc=None, attn_mask=None):
        """
        Args:
            x: input tensor, shape (seq_len, batch, d_model)
            memory: cached states from previous segment, shape (mem_len, batch, d_model) or None
            pos_enc: relative positional encodings (unused by nn.MultiheadAttention, included for API)
            attn_mask: causal attention mask, shape (seq_len, full_len) or None
        Returns:
            output: shape (seq_len, batch, d_model)
        """
        # Pre-LN for attention
        x_norm = self.norm_attn(x)

        # Concatenate memory with current input for key/value
        if memory is not None:
            mem_norm = self.norm_attn(memory)
            kv = torch.cat([mem_norm, x_norm], dim=0)
        else:
            kv = x_norm

        # Multi-head self-attention
        attn_out, _ = self.attn(
            query=x_norm,
            key=kv,
            value=kv,
            attn_mask=attn_mask,
            need_weights=False,
        )
        attn_out = self.dropout(attn_out)

        # GRU-gated residual for attention
        h = self.gate_attn(x, attn_out)

        # Pre-LN for feed-forward
        h_norm = self.norm_ff(h)
        ff_out = self.ff(h_norm)

        # GRU-gated residual for feed-forward
        out = self.gate_ff(h, ff_out)

        return out


class GTrXL(nn.Module):
    """Gated Transformer-XL (GTrXL) module.

    Stacks multiple GTrXLLayers with segment-level recurrence (memory).
    Each layer caches its output from the previous forward pass as memory
    for the next segment.

    Args:
        input_dim: dimension of input features (observation size after projection)
        d_model: transformer embedding dimension
        num_layers: number of GTrXL layers
        num_heads: number of attention heads
        d_ff: feed-forward hidden dimension (default: 4 * d_model)
        memory_length: number of cached timesteps from previous segment
        dropout: dropout rate (default: 0.0 for RL)
        gate_bias: GRU gate bias for identity initialization (default: 2.0)
        max_seq_len: maximum sequence length for positional encoding
    """

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
        max_seq_len=2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.memory_length = memory_length

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_enc = RelativePositionalEncoding(d_model, max_len=max_seq_len)

        # Transformer layers
        self.layers = nn.ModuleList([
            GTrXLLayer(d_model, num_heads, d_ff, dropout, gate_bias)
            for _ in range(num_layers)
        ])

        # Final layer norm (post-transformer)
        self.final_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for projection weights."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for layer in self.layers:
            # Initialize attention projections
            for p in layer.attn.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            # Initialize FF projections
            for module in layer.ff:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def init_memory(self, batch_size, device=None):
        """Initialize empty memory state.

        Returns:
            List of num_layers tensors, each shape (memory_length, batch_size, d_model),
            initialized to zeros.
        """
        return [
            torch.zeros(self.memory_length, batch_size, self.d_model, device=device)
            for _ in range(self.num_layers)
        ]

    def _build_causal_mask(self, seq_len, mem_len, device):
        """Build causal attention mask.

        The mask allows each query position to attend to:
        - All memory positions (no masking)
        - Current and past positions in the current segment (causal)

        Returns:
            Float mask of shape (seq_len, mem_len + seq_len) where True/inf = masked.
        """
        total_len = mem_len + seq_len
        # Create causal mask for the current segment part
        mask = torch.ones(seq_len, total_len, device=device, dtype=torch.bool)
        for i in range(seq_len):
            # Can attend to all memory + positions 0..i in current segment
            mask[i, :mem_len + i + 1] = False
        # Convert bool mask to float mask (True -> -inf for nn.MultiheadAttention)
        return mask.float().masked_fill(mask, float('-inf')).masked_fill(~mask, 0.0)

    def forward(self, x, memory=None, done_masks=None):
        """
        Args:
            x: input tensor, shape (seq_len, batch, input_dim)
            memory: list of num_layers tensors each (mem_len, batch, d_model), or None
            done_masks: episode termination masks, shape (seq_len, batch) or None.
                        1.0 = done (reset memory), 0.0 = not done.
        Returns:
            output: shape (seq_len, batch, d_model)
            new_memory: list of num_layers tensors each (mem_len, batch, d_model)
        """
        seq_len, batch_size = x.shape[0], x.shape[1]

        # Initialize memory if needed
        if memory is None:
            memory = self.init_memory(batch_size, device=x.device)

        mem_len = memory[0].shape[0]

        # Project input to d_model
        h = self.input_proj(x)

        # Add positional encoding to input
        pos = self.pos_enc(seq_len + mem_len)
        h = h + pos[mem_len:]  # Only add to current segment positions

        # Build causal mask
        attn_mask = self._build_causal_mask(seq_len, mem_len, x.device)

        # Process through layers, building new memory
        new_memory = []
        for i, layer in enumerate(self.layers):
            layer_mem = memory[i]

            # Zero out memory for done episodes at each timestep
            if done_masks is not None:
                layer_mem = self._apply_done_masks_to_memory(layer_mem, done_masks, batch_size)

            # Cache current layer input for next segment's memory
            # We cache the input to each layer (before the layer processes it)
            with torch.no_grad():
                new_mem = torch.cat([layer_mem, h], dim=0)
                new_memory.append(new_mem[-self.memory_length:].detach())

            h = layer(h, memory=layer_mem, pos_enc=pos, attn_mask=attn_mask)

        h = self.final_norm(h)

        return h, new_memory

    def _apply_done_masks_to_memory(self, memory, done_masks, batch_size):
        """Zero out memory for environments that had episode terminations.

        If any timestep in the current sequence has done=1 for an env,
        we zero the memory for that env (conservative approach matching LSTM behavior).
        """
        # done_masks: (seq_len, batch) - if any step is done, zero that env's memory
        if done_masks.dim() == 2:
            any_done = done_masks.any(dim=0)  # (batch,)
        else:
            any_done = done_masks

        if any_done.any():
            done_indices = any_done.nonzero(as_tuple=True)[0]
            memory = memory.clone()
            memory[:, done_indices, :] = 0.0
        return memory
