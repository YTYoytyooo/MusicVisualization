"""Independent main draws and uniform artist-group/track auxiliary draws."""
from collections import defaultdict
import hashlib
import json

import numpy as np
import torch

from tools.train_next_v2.data import StepSampler
from tools.train_next_v2.jamendo_data.contract import observed_tag_distribution


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':')).encode()).hexdigest()


class PairedSampler:
    def __init__(self, main, auxiliary, settings):
        if not main or not auxiliary or len({r['song_id'] for r in auxiliary}) != len(auxiliary):
            raise ValueError('Nonempty unique supervised pools required')
        self.main=StepSampler(main,settings['sampling_alpha'],settings['seed'])
        self.seed=settings['seed'];self.main_batch=settings['batch_size'];self.aux_batch=settings['aux_batch_size']
        groups=defaultdict(list)
        for index,row in enumerate(auxiliary):
            observed_tag_distribution(row['observed_tags'])
            if not row['group_id']: raise ValueError('Missing artist group')
            groups[row['group_id']].append(index)
        self.names=sorted(groups);self.groups=[groups[name] for name in self.names]
        self.definition=dict(schema='step-keyed-artist-group-track-v1',seed=self.seed,aux_stream_salt=440402,
            main_alpha=settings['sampling_alpha'],main_batch_size=self.main_batch,aux_batch_size=self.aux_batch,
            main_membership=digest([(r['song_id'],r['source_dataset']) for r in main]),
            auxiliary_membership=digest([(r['song_id'],r['group_id'],r['observed_tags']) for r in auxiliary]),
            groups=digest([(name,indices) for name,indices in zip(self.names,self.groups)]))

    def indices(self, step):
        if step < 1: raise ValueError('Optimizer steps begin at one')
        main=self.main.indices(step,self.main_batch)
        rng=np.random.default_rng([self.seed,step,440402])
        group_ids=rng.integers(len(self.groups),size=self.aux_batch)
        auxiliary=[self.groups[g][rng.integers(len(self.groups[g]))] for g in group_ids]
        return main,auxiliary

    def state(self, completed_step):
        return dict(**self.definition,next_step=completed_step+1,main_exposures=completed_step*self.main_batch,
            auxiliary_forward_exposures=completed_step*self.aux_batch)

    def verify(self, saved, completed_step):
        if saved != self.state(completed_step):
            raise ValueError('Main/auxiliary sampler state or membership mismatch')


def collate_auxiliary(rows, mean, std):
    if not rows or any(len(r['x']) < 1 for r in rows): raise ValueError('Empty auxiliary track')
    if any(set(r) & {'y','mask','class_index','timepoint_targets'} for r in rows):
        raise ValueError('Only whole-track observed tags are allowed')
    lengths=torch.tensor([len(r['x']) for r in rows],dtype=torch.long)
    values=np.zeros((len(rows),int(lengths.max()),len(mean)),dtype=np.float32)
    targets=[]
    for i,row in enumerate(rows):
        values[i,:lengths[i]]=(row['x']-mean)/std
        targets.append(observed_tag_distribution(row['observed_tags']))
    return torch.from_numpy(values),torch.from_numpy(np.stack(targets)),lengths
