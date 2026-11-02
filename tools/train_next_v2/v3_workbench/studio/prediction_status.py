"""Read-only presentation of frozen prediction evidence; never fits a model.

This checks the imported record chain, not model accuracy or current promotion.
Missing or changed evidence cannot turn into an affirmative calibration claim.
"""
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from .paths import WORKSPACE

MANIFEST = WORKSPACE / 'models/next_va_v2/v3-preview-v1/manifest.json'


def prediction_statuses(path, project, records):
    from .store import read_json, digest, object_digest, _contained_file, load_raw

    def fallback(name, reason):
        return dict(evidence='unverified', research='not_recorded',
                    calibration='not_recorded', reason=reason,
                    kind='legacy' if name == 'v2' else 'unknown')

    result = {name: fallback(name, 'missing_evidence') for name in records}
    provenance = project.get('provenance', {})
    if not isinstance(provenance, dict) or provenance.get('kind') != 'frozen-v3-analysis':
        return result
    try:
        # Only the known local manifest is considered. Do not traverse arbitrary
        # model paths or invoke the training/test-data manifest validator here.
        manifest_hash = digest(MANIFEST)
        if manifest_hash != provenance['manifest_sha256']:
            raise ValueError('manifest identity')
        manifest = read_json(MANIFEST)
        if (manifest.get('schema_version') != 3
                or manifest.get('status') != 'research_preview_not_promoted'
                or manifest.get('promoted_model') is not False):
            raise ValueError('unsupported status')
        paths = [_contained_file(path, 'analysis/source-v3/'+relative) for relative in
                 ('project.json', 'analysis/result.json', 'analysis/receipt.json')]
        for file, key in zip(paths, ('source_project_sha256', 'source_result_sha256', 'source_receipt_sha256')):
            if digest(file) != provenance[key]:
                raise ValueError('archive identity')
        source, analysis, receipt = map(read_json, paths)
        if (any(obj.get('schema_version') != 3 or obj.get('manifest_sha256') != manifest_hash
                for obj in (source, analysis, receipt))
                or source.get('id') != analysis.get('project_id')
                or source.get('audio') != analysis.get('audio')
                or receipt.get('project_sha256') != provenance['source_project_sha256']
                or receipt.get('result_sha256') != provenance['source_result_sha256']
                or receipt.get('audio_sha256') != project.get('audio_sha256')
                or analysis['audio']['sha256'] != project.get('audio_sha256')
                or round(analysis['audio']['duration_seconds']*1e6) != project.get('duration_us')
                or analysis.get('provenance', {}).get('promoted_model') is not False):
            raise ValueError('analysis binding')
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
        return {name: fallback(name, 'evidence_mismatch') for name in records}

    for name, record in records.items():
        if name not in ('b01', 'b00', 'v2'):
            continue
        try:
            identity = manifest['legacy'] if name == 'v2' else manifest['systems'][name]
            if (Path(record.get('model_path', '')).resolve() != MANIFEST.resolve()
                    or record.get('model_sha256') != object_digest(identity)):
                raise ValueError('model binding')
            original = analysis['systems'][name]
            raw = load_raw(path, system=name)
            values = np.asarray(original['raw_va'], dtype=np.float64)
            clock = np.rint(np.asarray(original['times_seconds'])*1e6).astype(np.int64)
            if (not np.array_equal(raw.times_us, clock)
                    or not np.array_equal(raw.mask, np.isfinite(values))
                    or not np.array_equal(np.asarray(raw), values, equal_nan=True)):
                raise ValueError('raw values or native clock')
            if name == 'v2':
                if original.get('calibrated') is not False:
                    raise ValueError('legacy calibration')
                result[name] = dict(evidence='verified_frozen_record', kind='legacy',
                                    research='not_recorded', calibration='not_applied', reason=None)
            else:
                if (manifest.get('calibration', {}).get('status') != 'unavailable'
                        or identity.get('calibration') != 'unavailable'
                        or analysis.get('calibration', {}).get('status') != 'unavailable'
                        or analysis['calibration'].get('values') is not None):
                    raise ValueError('calibration binding')
                result[name] = dict(evidence='verified_frozen_record', kind='research',
                                    research='not_promoted', calibration='unavailable', reason=None)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, BadZipFile, EOFError):
            result[name] = fallback(name, 'evidence_mismatch')
    return result
