"""Preserve MERP averages without inventing audio timestamps for filtered data."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'data/datasets/.dependencies'))
import pyarrow.parquet as pq


def main():
    root = ROOT / 'data/datasets/merp'
    output = root / 'prepared/quarantined-v1'
    output.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    for row in pq.read_table(root / 'raw/annotations_filtered.parquet').to_pylist():
        groups[row['song_id']].append(row)
    records = []
    for song_id, rows in sorted(groups.items()):
        lengths = [[len(row[axis]) for axis in ('valence', 'arousal')] for row in rows]
        length = max(max(x) for x in lengths)
        sums = np.zeros((length, 2), dtype=np.float64)
        counts = np.zeros((length, 2), dtype=np.int32)
        for row in rows:
            for dim, axis in enumerate(('valence', 'arousal')):
                values = np.asarray(row[axis], dtype=np.float64)
                valid = np.isfinite(values)
                sums[:len(values), dim] += np.where(valid, values, 0)
                counts[:len(values), dim] += valid
        averages = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        # Store nominal index only. The public conversion does not establish t=0.
        np.savez_compressed(output / f'{song_id}.npz', sample_index=np.arange(length),
                            nominal_sample_period_seconds=.1, mean=averages, count=counts)
        audio = sf.info(root / 'raw/audio' / f'{song_id}.wav')
        records.append(dict(song_id=song_id, raters=len(rows), length_range=[min(map(min, lengths)), length],
                            audio_duration_seconds=audio.duration, audio_time_origin_seconds=None,
                            training_enabled=False, known_deam_anchor=song_id.startswith('deam_'),
                            reason='filtered_array_audio_origin_unresolved'))
    report = dict(status='quarantined', songs=len(records), known_deam_anchors=sum(r['known_deam_anchor'] for r in records),
                  source='https://huggingface.co/datasets/amaai-lab/MERP',
                  alignment_evidence=['https://github.com/dorienh/MERP/blob/master/processing/pruning.py',
                                      'https://github.com/dorienh/MERP/blob/master/amazon_Merged.html'],
                  reason='Public code crops excess samples from the front; paper describes end cropping; no audio-clock timestamps in filtered arrays.',
                  label_processing='elementwise mean with independent valid-rater counts; no truncation, clipping or time stretching',
                  records=records)
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='quarantined', songs=len(records), known_deam_anchors=report['known_deam_anchors'])))


if __name__ == '__main__':
    main()
