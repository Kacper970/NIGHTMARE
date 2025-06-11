import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MaskedSequenceEncoder(nn.Module):
    def __init__(
        self, input_dim=512, hidden_dim=768, num_layers=8, num_heads=12, dropout=0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.linear_in = nn.Linear(input_dim, hidden_dim)

        self.mas_token = nn.Parameter(torch.rand(1, 1, hidden_dim) * 0.01)
        self.cls_token = nn.Parameter(torch.rand(1, 1, hidden_dim) * 0.01)
        self.sep_token = nn.Parameter(torch.rand(1, 1, hidden_dim) * 0.01)

        self.positional_encoding = PositionalEncoding(hidden_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
            ),
            num_layers,
        )

    def forward(self, sequence, secondary_sequence=None, mask=None):
        b, s = sequence.shape[:2]
        tokens = self.linear_in(sequence)

        # Mask relevant token if provided
        if mask != None:
            tokens[mask] = self.mas_token

        # Add [CLS] to start of x_tokens
        cls_token = self.cls_token.expand(b, 1, self.hidden_dim)
        tokens = torch.cat((cls_token, tokens), dim=1)

        if secondary_sequence is not None:
            secondary_tokens = self.linear_in(secondary_sequence)

            if mask != None:
                secondary_tokens[mask] = self.mas_token

            # Add [SEP] between sequences
            sep_token = self.sep_token.expand(b, 1, self.hidden_dim)
            tokens = torch.cat((tokens, sep_token, secondary_tokens), dim=1)

        tokens = self.positional_encoding(tokens)
        encoding = self.encoder(tokens)

        return encoding
