"""The unchanged B00 VA backbone plus a whole-track observed-tag head."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tools.train_next_v2.model import SequenceModel
from tools.train_next_v2.jamendo_data.contract import TAG_ORDER


class AuxiliaryModel(SequenceModel):
    def __init__(self, input_dim, settings):
        if settings['fusion'] != 'concatenate' or settings['separate_heads']:
            raise ValueError('Jamendo comparison retains the B00 backbone')
        super().__init__(input_dim, settings)
        self.tags = nn.Linear(settings['hidden']*2, len(TAG_ORDER))

    def encode(self, x, lengths):
        packed = pack_padded_sequence(self.norm(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        return pad_packed_sequence(output, batch_first=True, total_length=x.shape[1])[0]

    def forward(self, x, lengths):
        return self.head(self.encode(x, lengths))

    def classify(self, x, lengths):
        states = self.encode(x, lengths)
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths.to(x.device)[:, None]
        pooled = (states*valid.unsqueeze(-1)).sum(1)/lengths.to(x.device)[:, None]
        return self.tags(pooled)


def tag_loss(logits, observed_distribution):
    """Conditional observed-tag cross entropy, not missing-label binary BCE."""
    target = observed_distribution
    if logits.ndim != 2 or logits.shape[1] != len(TAG_ORDER) or target.shape != logits.shape:
        raise ValueError('Exactly one twelve-tag distribution is required per full track')
    if not torch.isfinite(target).all() or (target < 0).any() or not torch.allclose(target.sum(1), torch.ones_like(target[:, 0]), atol=1e-6, rtol=0):
        raise ValueError('Invalid observed-tag distribution')
    return -(target*F.log_softmax(logits, dim=1)).sum(1).mean()
