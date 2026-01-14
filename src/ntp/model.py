import numpy as np
import torch
import torch.distributions as distributions
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):   
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer (not a learnable parameter)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        # Add positional encoding to the input
        return x + self.pe[:, :x.size(1), :]
    


    import math

class TransformerNextToken(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers=2, num_heads=4, max_len=200, pad_value=-1):
        super().__init__()
        self.pad_value = pad_value
        self.embed_dim = embed_dim
        # 1. Embeddings
        # We need a standard embedding + a positional embedding (so it knows order)
        self.token_embedding = nn.Embedding(3, embed_dim) # 0, 1, pad
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len)
        self.layernorm = nn.LayerNorm(embed_dim)
        # 2. Transformer Block (Causal Decoder)
        # batch_first=True ensures it matches your input shape (batch, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
            dropout=0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Output Head
        self.output_head = nn.Linear(embed_dim, 1)

    def forward(self, x, lengths=None):
        device = x.device
        batch_size, seq_len = x.shape

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        key_padding_mask = (x == self.pad_value)

        x_in = x.clone()
        x_in[x_in == self.pad_value] = 2

            # Embed and Scale
            # Important: Multiply by sqrt(d_model) before adding PE (Standard practice)
        x_emb = self.token_embedding(x_in)

            # Add Fixed Position Info
        x_emb = self.pos_encoder(x_emb)

        x_emb = self.layernorm(x_emb)

        output = self.transformer(
                x_emb,
                mask=causal_mask,
                src_key_padding_mask=key_padding_mask,
                is_causal=True
            )

        logits = self.output_head(output).squeeze(-1)
        return logits