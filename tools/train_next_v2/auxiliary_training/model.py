"""The frozen B00 sequence backbone with a separate whole-clip classifier."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tools.train_next_v2.model import SequenceModel


class AuxiliaryModel(SequenceModel):
    def __init__(self, input_dim, settings):
        if settings['fusion'] != 'concatenate' or settings['separate_heads']:
            raise ValueError('D protocol requires the unchanged B00 backbone')
        super().__init__(input_dim, settings)
        self.quadrant = nn.Linear(settings['hidden'] * 2, 4)

    def encode(self, x, lengths):
        packed = pack_padded_sequence(self.norm(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=x.shape[1])
        return output

    def forward(self, x, lengths):
        # Existing continuous-VA evaluators see exactly the original interface.
        return self.head(self.encode(x, lengths))

    def classify(self, x, lengths):
        states = self.encode(x, lengths)
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths.to(x.device)[:, None]
        pooled = (states * valid.unsqueeze(-1)).sum(1) / lengths.to(x.device)[:, None]
        return self.quadrant(pooled)
