from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tools.train_next.core import supervised_loss


class SequenceModel(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.fusion = config['fusion']
        self.separate_heads = config['separate_heads']
        if self.fusion == 'branches':
            if input_dim != 531:
                raise ValueError('Branch fusion requires CLAP512 + acoustic19')
            self.semantic = nn.Sequential(nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU())
            self.acoustic = nn.Sequential(nn.Linear(19, 64), nn.LayerNorm(64), nn.GELU())
            width = 192
        else:
            self.norm = nn.LayerNorm(input_dim)
            width = input_dim
        self.lstm = nn.LSTM(width, config['hidden'], num_layers=config['layers'],
                            dropout=config['dropout'] if config['layers'] > 1 else 0.,
                            batch_first=True, bidirectional=True)
        if self.separate_heads:
            self.heads = nn.ModuleList([nn.Sequential(nn.Dropout(config['dropout']),
                nn.Linear(config['hidden'] * 2, 64), nn.GELU(), nn.Linear(64, 1), nn.Tanh()) for _ in range(2)])
        else:
            self.head = nn.Sequential(nn.Dropout(config['dropout']), nn.Linear(config['hidden'] * 2, 2), nn.Tanh())

    def forward(self, x, lengths):
        if self.fusion == 'branches':
            x = torch.cat((self.semantic(x[..., :512]), self.acoustic(x[..., 512:])), dim=-1)
        else:
            x = self.norm(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=x.shape[1])
        return torch.cat([head(output) for head in self.heads], dim=-1) if self.separate_heads else self.head(output)


def derivative_loss(prediction, target, mask, native_seconds, hop_seconds=.5):
    terms = []
    for i in range(len(prediction)):
        for axis in range(2):
            indices = torch.nonzero(mask[i, :, axis], as_tuple=False).flatten()
            if len(indices) < 2:
                continue
            elapsed = (indices[1:] - indices[:-1]).to(prediction.dtype) * hop_seconds
            adjacent = torch.isclose(elapsed, elapsed.new_tensor(native_seconds[i]))
            if not torch.any(adjacent):
                continue
            left, right = indices[:-1][adjacent], indices[1:][adjacent]
            dt = elapsed[adjacent]
            p = (prediction[i, right, axis] - prediction[i, left, axis]) / dt
            y = (target[i, right, axis] - target[i, left, axis]) / dt
            terms.append(nn.functional.smooth_l1_loss(p, y, beta=.2))
    return torch.stack(terms).mean() if terms else prediction.sum() * 0


def loss(prediction, target, mask, native_seconds, config):
    value = supervised_loss(prediction, target, mask, ccc_weight=config['ccc_weight'])
    if config['derivative_weight']:
        value = value + config['derivative_weight'] * derivative_loss(prediction, target, mask, native_seconds)
    return value
