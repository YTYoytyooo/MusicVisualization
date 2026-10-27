"""Preserve each model's actual input, normalization and temporal contract."""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import sys

import numpy as np
import torch

from tools.train_next.core import SequenceVA, sha256
from tools.train_next.features import feature_spec, local_clap_snapshot
from tools.train_next_v2.common import ROOT, read_json, verify_hashes
from tools.train_next_v2.calibration.job_v2 import load_models
from tools.train_next_v2.research_inference.adapter import standardize, validate_clock


@lru_cache(maxsize=1)
def load_legacy_source():
    """Load unchanged V2 source without altering the app's import paths/files."""
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, ROOT/'versions/v2.0'/filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    features = load('_comparison_v2_features', 'feature_extraction.py')
    previous = sys.modules.get('feature_extraction')
    try:
        sys.modules['feature_extraction'] = features
        source = load('_comparison_v2_emotion', 'emotion_model.py')
    finally:
        if previous is None:
            sys.modules.pop('feature_extraction', None)
        else:
            sys.modules['feature_extraction'] = previous
    return features, source


def regular_clock(times, hop):
    times = np.asarray(times)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or not np.allclose(times, np.arange(len(times))*hop, rtol=0, atol=1e-7)):
        raise ValueError('Unexpected native model clock; shifting/stretching is forbidden')
    return times


def common_clock(sequence_times, legacy_times, mask):
    """Exact joins only; drop unsupported terminal labels equally for all models."""
    sequence_times = regular_clock(sequence_times, .5)
    legacy_times = regular_clock(legacy_times, .1)
    mask = np.asarray(mask)
    if mask.shape != (len(sequence_times),2) or mask.dtype != np.bool_:
        raise ValueError('Expected independent boolean V/A masks')
    indices = np.rint(sequence_times/.1).astype(np.int64)
    supported = indices < len(legacy_times)
    if not np.allclose(sequence_times[supported], legacy_times[indices[supported]], rtol=0, atol=1e-7):
        raise ValueError('Model clocks cannot be joined exactly')
    indices[~supported] = -1
    return indices, mask & supported[:, None]


def legacy_windows(embeddings, start, end):
    x = np.asarray(embeddings)
    if (x.ndim != 2 or x.shape[1] != 512 or not len(x) or x.dtype != np.float32
            or not np.isfinite(x).all() or not 0 <= start < end <= len(x)):
        raise ValueError('Expected finite raw float32 CLAP512 embeddings and valid bounds')
    out = np.zeros((end-start, 10, 512), np.float32)
    for local, index in enumerate(range(start,end)):
        left = max(0,index-9)
        out[local, -(index-left+1):] = x[left:index+1]
    return out


class V2Head:
    def __init__(self, model, device):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @classmethod
    def load(cls, protocol, device):
        verify_hashes(protocol['pinned_hashes'])
        _, source = load_legacy_source()
        if (source.WINDOW_SIZE, source.CLAP_DIM, source.N_EMOTIONS) != (10,512,5):
            raise ValueError('V2 model contract changed')
        interface = source.EmotionInterface.load(str(ROOT/protocol['models']['v2']['checkpoint']))
        return cls(interface.model, device)

    @torch.inference_mode()
    def predict(self, embeddings, batch_size=256):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Invalid inference batch size')
        parts=[]
        for start in range(0,len(embeddings),batch_size):
            x=legacy_windows(embeddings,start,min(start+batch_size,len(embeddings)))
            parts.append(self.model(torch.from_numpy(x).to(self.device)).cpu().numpy())
        if not parts:
            raise ValueError('Empty legacy embeddings')
        result=np.concatenate(parts)
        if result.shape!=(len(embeddings),5) or not np.isfinite(result).all() or np.any(np.abs(result)>1):
            raise ValueError('Invalid legacy output')
        return result


class V2Encoder:
    """Device port of V2 extraction, including its original random fusion flag."""
    def __init__(self, protocol, device='cuda'):
        self.protocol=protocol
        verify_hashes(protocol['pinned_hashes'])
        self.device=torch.device(device)
        self.model=self.processor=None

    def unload(self):
        self.model=self.processor=None
        if self.device.type=='cuda': torch.cuda.empty_cache()

    @torch.inference_mode()
    def extract(self, audio_22050, seed=42, progress=None):
        y=np.asarray(audio_22050)
        if y.ndim!=1 or y.dtype!=np.float32 or len(y)<2205 or not np.isfinite(y).all():
            raise ValueError('V2 audio must be finite mono float32 at 22050 Hz, >=0.1 s')
        if seed!=self.protocol['v2_features']['seed']:
            raise ValueError('Frozen legacy extraction seed changed')
        np.random.seed(seed % 2**32)
        if self.model is None:
            _, source=load_legacy_source()
            self.model,self.processor=source._load_clap(ROOT/self.protocol['v2_features']['snapshot'])
            self.model.to(self.device).eval()
        import librosa
        y48=librosa.resample(y,orig_sr=22050,target_sr=48000,res_type='soxr_hq').astype(np.float32)
        n=max(1,len(y)//2205)
        parts=[]
        for start in range(0,n,16):
            windows=[]
            for index in range(start,min(start+16,n)):
                offset=int(index*.1*48000)
                window=y48[offset:offset+96000]
                if len(window)<96000: window=np.pad(window,(0,96000-len(window)))
                windows.append(window)
            inputs=self.processor(audio=windows,return_tensors='pt',sampling_rate=48000,padding=True)
            inputs={k:v.to(self.device) for k,v in inputs.items()}
            output=self.model.audio_model(input_features=inputs.get('input_features'),is_longer=inputs.get('is_longer'))
            parts.append(self.model.audio_projection(output.pooler_output).cpu().numpy().astype(np.float32))
            if progress: progress(min(start+16,n),n)
        result=np.concatenate(parts)
        if result.shape!=(n,512) or not np.isfinite(result).all(): raise ValueError('Invalid V2 features')
        return np.arange(n,dtype=np.float64)*.1,result


class SequenceHeads:
    def __init__(self, protocol, kind, device='cuda'):
        if kind not in ('mixed','b01'): raise ValueError('Unknown comparison model')
        verify_hashes(protocol['pinned_hashes'])
        self.device=torch.device(device)
        self.key=protocol['sequence_features']['key']
        if feature_spec(local_clap_snapshot())!=protocol['sequence_features']['spec']:
            raise ValueError('Sequence feature contract changed')
        selected=protocol['models'][kind]
        if kind=='mixed':
            saved=torch.load(ROOT/selected['checkpoint'],map_location='cpu',weights_only=True)
            if (saved['architecture']!='whole-song-bilstm-va' or saved['label_order']!=['valence','arousal']
                    or saved['feature_spec']!=protocol['sequence_features']['spec']):
                raise ValueError('Mixed checkpoint contract changed')
            verify_hashes({'tools/train_next/'+k:v for k,v in saved['config']['code'].items()})
            model=SequenceVA(531,saved['config']['hidden']).to(self.device)
            model.load_state_dict(saved['state_dict'],strict=True)
            self.models=[model.eval()]
            self.scaler=(saved['mean'].numpy(),saved['std'].numpy())
        else:
            if [r['run_id'] for r in selected['members']]!=[f'b01-branch-fusion-s{s}' for s in (20261003,20261004,20261005)]:
                raise ValueError('B01 members changed')
            for member in selected['members']:
                cfg=read_json(ROOT/member['config'])
                if cfg['feature_key']!=self.key or cfg['protocol_sha256']!=protocol['B_protocol_sha256']:
                    raise ValueError('B01 feature/protocol mismatch')
                verify_hashes(cfg['code'])
            self.models,self.scaler=load_models(selected,self.device)

    @torch.inference_mode()
    def predict(self,times,features,key,duration_seconds):
        if key!=self.key: raise ValueError('Wrong feature cache')
        x=standardize(features,*self.scaler)
        validate_clock(times,len(x),duration_seconds)
        tensor=torch.from_numpy(x).unsqueeze(0).to(self.device)
        lengths=torch.tensor([len(x)],dtype=torch.int64)
        members=np.stack([m(tensor,lengths).cpu().numpy()[0] for m in self.models])
        if not np.isfinite(members).all() or np.any(np.abs(members)>1): raise ValueError('Invalid sequence predictions')
        return members,members.astype(np.float64).mean(axis=0)
