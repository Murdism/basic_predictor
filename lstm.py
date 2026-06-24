#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


OUTPUT_DIMS = {
    "point": 2,
    "gaussian": 4,
    "bivariate_gaussian": 5,
}

OUTPUT_TYPE_ALIASES = {
    "point": "point",
    "xy": "point",
    "gaussian": "gaussian",
    "bivariate": "bivariate_gaussian",
    "bivariate_gaussian": "bivariate_gaussian",
}


class InputEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))


class OutputHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim=2,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=2,
        dropout=0.0,
        output_type="point",
        output_dim=None,
    ):
        super().__init__()

        output_type = self._normalize_output_type(output_type)
        expected_output_dim = OUTPUT_DIMS[output_type]

        if output_dim is None:
            output_dim = expected_output_dim
        elif output_dim != expected_output_dim:
            raise ValueError(
                f"output_dim={output_dim} does not match output_type='{output_type}' "
                f"(expected {expected_output_dim})."
            )

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_type = output_type
        self.output_dim = output_dim

        self.input_embedding = InputEmbedding(input_dim, embedding_dim)

        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_head = OutputHead(hidden_dim, output_dim)

    def _normalize_output_type(self, output_type):
        key = output_type.lower()
        if key not in OUTPUT_TYPE_ALIASES:
            valid = ", ".join(sorted(OUTPUT_DIMS))
            raise ValueError(f"Unknown LSTM output_type: {output_type}. Expected one of: {valid}.")
        return OUTPUT_TYPE_ALIASES[key]

    def encode(self, obs_traj):
        embedded = self.input_embedding(obs_traj)
        _, (h, c) = self.encoder(embedded)
        return h, c

    def decode(self, decoder_input, hidden, cell, pred_len):
        predictions = []

        for _ in range(pred_len):
            embedded = self.input_embedding(decoder_input).unsqueeze(1)
            output, (hidden, cell) = self.decoder(embedded, (hidden, cell))
            pred_step = self.output_head(output.squeeze(1))
            predictions.append(pred_step)
            decoder_input = pred_step[..., : self.input_dim]

        return torch.stack(predictions, dim=1)

    def forward(self, obs_traj, pred_len):
        decoder_input = obs_traj[:, -1, :]
        hidden, cell = self.encode(obs_traj)
        pred_traj = self.decode(decoder_input, hidden, cell, pred_len)
        return pred_traj
