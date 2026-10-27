"""Per-song atomic data plus a hash-checked commit receipt."""
import json
from pathlib import Path
import re

import numpy as np

from tools.train_next.core import atomic_json, sha256
from tools.train_next_v2.common import ROOT, now


class ArtifactStore:
    def __init__(self, directory, contract):
        self.directory = Path(directory).resolve()
        if not self.directory.is_relative_to(ROOT):
            raise ValueError('Artifacts must stay inside Vm')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.contract = contract

    def paths(self, key):
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', key):
            raise ValueError('Unsafe artifact name')
        return (self.directory/(key+'.npz'), self.directory/(key+'.json'), self.directory/(key+'.tmp'))

    def read(self, key, song_id):
        path, receipt, temporary = self.paths(key)
        if temporary.exists():
            raise ValueError('Partial artifact requires inspection: '+str(temporary))
        if not receipt.exists():
            if path.exists():
                raise ValueError('Unreceipted artifact requires inspection: '+str(path))
            return None
        row = json.loads(receipt.read_text(encoding='utf-8'))
        if (row['contract_sha256'] != self.contract or row['song_id'] != song_id
                or row['file'] != path.name or not path.exists() or sha256(path) != row['sha256']):
            raise ValueError('Artifact provenance/hash changed: '+key)
        with np.load(path, allow_pickle=False) as saved:
            arrays = {k:saved[k].copy() for k in saved.files}
        return row, arrays

    def save(self, key, song_id, arrays, measurements):
        if self.read(key, song_id) is not None:
            raise ValueError('Preserve completed artifact: '+key)
        path, receipt, temporary = self.paths(key)
        with temporary.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        row = dict(song_id=song_id, contract_sha256=self.contract, file=path.name,
                   sha256=sha256(path), completed_at=now(), measurements=measurements)
        atomic_json(receipt, row)
        return row
