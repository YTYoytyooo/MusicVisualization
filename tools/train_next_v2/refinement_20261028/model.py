"""Two preregistered single-factor alternatives to the frozen B01 model."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tools.train_next_v2.model import SequenceModel, loss as reference_loss


class ToneResidualModel(SequenceModel):
    """Bounded offline prediction with explicit mean over valid FEATURE frames.

    The tone is not the mean of sparse annotation points. No label mask enters
    inference. Centering uses actual sequence lengths and excludes padding.
    Since a centered tanh residual lies in [-2, 2], half the available headroom
    guarantees the final prediction remains in [-1, 1] without clipping.
    """

    def __init__(self, input_dim, config):
        if config['fusion'] != 'branches' or config['separate_heads']:
            raise ValueError('Tone residual is defined only against the B01 shared-head control')
        super().__init__(input_dim, config)
        self.tone_head = nn.Sequential(nn.Dropout(config['dropout']),
                                       nn.Linear(config['hidden'] * 2, 2), nn.Tanh())

    def components(self, x, lengths):
        fused = torch.cat((self.semantic(x[..., :512]), self.acoustic(x[..., 512:])), dim=-1)
        packed = pack_padded_sequence(fused, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.lstm(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.shape[1])
        valid = (torch.arange(x.shape[1], device=x.device)[None, :] < lengths.to(x.device)[:, None])[..., None]
        count = lengths.to(device=x.device, dtype=x.dtype)[:, None, None]
        pooled = (encoded * valid).sum(dim=1) / count[:, 0]
        tone = self.tone_head(pooled)[:, None, :]
        raw = self.head(encoded)
        centered = (raw - (raw * valid).sum(dim=1, keepdim=True) / count) * valid
        residual = (1 - tone.abs()) * .5 * centered
        return tone, residual

    def forward(self, x, lengths):
        tone, residual = self.components(x, lengths)
        return tone + residual


def build_model(input_dim, config):
    if config['architecture'] == 'branch_lstm':
        return SequenceModel(input_dim, config)
    if config['architecture'] == 'tone_residual':
        return ToneResidualModel(input_dim, config)
    raise ValueError('Architecture outside the frozen comparison')


def observed_mean_loss(prediction, target, mask):
    terms = []
    for i in range(len(prediction)):
        for axis in range(2):
            valid = mask[i, :, axis]
            if valid.any():
                terms.append(nn.functional.smooth_l1_loss(prediction[i, valid, axis].mean(),
                                                          target[i, valid, axis].mean(), beta=.2))
    return torch.stack(terms).mean() if terms else prediction.sum() * 0


def objective(prediction, target, mask, native_seconds, config):
    result = reference_loss(prediction, target, mask, native_seconds, config)
    if config['mean_weight']:
        result = result + config['mean_weight'] * observed_mean_loss(prediction, target, mask)
    return result
