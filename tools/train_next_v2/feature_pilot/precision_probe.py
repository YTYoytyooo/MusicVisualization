"""Bounded FP32/TF32 diagnosis of a recorded MERT batch-parity failure."""
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import sys
import time
import traceback

import librosa
import numpy as np
import soundfile as sf
import torch

from tools.train_next.core import ROOT, atomic_json, sha256
from tools.train_next.runtime import run_lease
from tools.train_next_v2.common import GPU_LEASE, assert_gpu_available, now, process_identity, read_json, verify_hashes
from .encoders import FrozenEncoder


def main(launch_path):
    launch = read_json(launch_path)
    destination = Path(launch_path).parent
    if (destination / 'report.json').exists():
        raise FileExistsError('Immutable precision diagnostic already exists')
    verify_hashes(launch['code_hashes'])
    parent = read_json(launch['parent_launch'])
    verify_hashes(parent['protected_hashes'])
    for name, value in parent['environment'].items():
        if os.environ.get(name) != value:
            raise ValueError('Environment mismatch: ' + name)
    assert_gpu_available()
    report = dict(job_id=launch['job_id'], started_at=now(), status='running', cases=[],
                  parent_launch_sha256=sha256(launch['parent_launch']), labels_read=False)
    with run_lease(GPU_LEASE), run_lease(launch['job_id']), (destination / 'console.log').open('x', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            identity = process_identity()
            atomic_json(destination / 'process.json', identity)
            atomic_json(destination / 'status.json', dict(state='running', process_identity=identity, updated_at=now()))
            encoder = None
            try:
                torch.set_num_threads(8)
                torch.manual_seed(20261003)
                np.random.seed(20261003)
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark = False
                report['initial_precision'] = dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
                row = parent['songs'][0]
                audio, rate = sf.read(ROOT / row['audio_path'], dtype='float32', always_2d=True)
                waveform = librosa.resample(audio.mean(1), orig_sr=rate, target_sr=24000, res_type='soxr_hq')
                times = np.asarray(row['preflight_indices']) * .5
                encoder = FrozenEncoder(next(s for s in parent['encoders'] if s['name'] == 'mert95m'))
                arrays = {}
                for mode, conv_tf32 in [('original', report['initial_precision']['cudnn_allow_tf32']), ('ieee_fp32', False)]:
                    torch.backends.cudnn.allow_tf32 = conv_tf32
                    torch.backends.cuda.matmul.allow_tf32 = report['initial_precision']['matmul_allow_tf32'] if mode == 'original' else False
                    started = time.monotonic()
                    values, _, batches = encoder.encode(waveform, times, 10, 4)
                    repeated, _, _ = encoder.encode(waveform, times, 10, 4)
                    index = len(times) // 2
                    single, _, _ = encoder.encode(waveform, times[index:index+1], 10, 1)
                    item = dict(mode=mode, batches=batches, seconds=time.monotonic()-started,
                        repeated_max_diff=float(np.max(np.abs(values - repeated))),
                        batch_single_max_diff=float(np.max(np.abs(values[index] - single[0]))),
                        passed_original_tolerance=bool(np.allclose(values[index], single[0], **parent['batch_parity_tolerance'])))
                    arrays[mode+'_batch'] = values
                    arrays[mode+'_single'] = single
                    report['cases'].append(item)
                    print(json.dumps(item), flush=True)
                report['status'] = 'completed'
                with (destination / 'numeric-evidence.npz').open('xb') as stream:
                    np.savez_compressed(stream, times=times, **arrays)
                report['numeric_evidence_sha256'] = sha256(destination / 'numeric-evidence.npz')
            except Exception as error:
                report.update(status='failed', error=repr(error))
                traceback.print_exc()
            finally:
                if encoder is not None:
                    encoder.unload()
            report['completed_at'] = now()
            atomic_json(destination / 'report.json', report)
            atomic_json(destination / 'status.json', dict(state=report['status'], process_identity=identity, updated_at=now()))
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
