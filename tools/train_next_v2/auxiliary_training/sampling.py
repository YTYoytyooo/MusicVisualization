"""Disjoint deterministic main/auxiliary streams, completely restorable by step."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from tools.train_next_v2.data import StepSampler


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class PairedSampler:
    def __init__(self, main, auxiliary, settings):
        if not main or not auxiliary:
            raise ValueError('Both supervised pools must be nonempty')
        self.main = StepSampler(main, settings['sampling_alpha'], settings['seed'])
        self.seed = settings['seed']
        self.main_batch = settings['batch_size']
        self.aux_batch = settings['aux_batch_size']
        self.aux_size = len(auxiliary)
        self.definition = dict(schema='step-keyed-paired-sampler-v1', seed=self.seed,
            main_alpha=settings['sampling_alpha'], main_batch_size=self.main_batch,
            aux_batch_size=self.aux_batch, aux_stream_salt=440401,
            main_membership=digest([(s['song_id'], s['source_dataset']) for s in main]),
            auxiliary_membership=digest([(s['song_id'], s['class_index']) for s in auxiliary]))

    def indices(self, step):
        if step < 1:
            raise ValueError('Optimizer steps start at one')
        main = self.main.indices(step, self.main_batch)
        # Never draws from the main sampler or global NumPy/Torch RNG.
        rng = np.random.default_rng([self.seed, step, 440401])
        auxiliary = rng.integers(self.aux_size, size=self.aux_batch).tolist()
        return main, auxiliary

    def state(self, completed_step):
        return dict(**self.definition, next_step=completed_step + 1,
                    main_exposures=completed_step * self.main_batch,
                    auxiliary_forward_exposures=completed_step * self.aux_batch)

    def verify(self, saved, completed_step):
        if saved != self.state(completed_step):
            raise ValueError('Main/auxiliary sampler state or membership mismatch')


def collate_auxiliary(rows, mean, std):
    """One class per complete clip; no artificial timepoint V/A targets or masks."""
    lengths = torch.tensor([len(s['x']) for s in rows], dtype=torch.long)
    if not len(rows) or int(lengths.min()) < 1:
        raise ValueError('Empty auxiliary clip')
    if any('y' in s or 'mask' in s or s['class_index'] not in range(4) for s in rows):
        raise ValueError('Auxiliary clips may carry only whole-clip quadrant supervision')
    values = np.zeros((len(rows), int(lengths.max()), len(mean)), dtype=np.float32)
    for i, row in enumerate(rows):
        values[i, :lengths[i]] = (row['x'] - mean) / std
    return torch.from_numpy(values), torch.tensor([s['class_index'] for s in rows]), lengths
