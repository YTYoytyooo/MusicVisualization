"""Freeze six previously audited heads and an existing-audio-only library."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.train_next.core import atomic_json, sha256
from tools.train_next_v2.common import ROOT, BASE, now, read_json, verify_hashes

MANIFEST = ROOT / 'models/next_va_v2/v3-preview-v1/manifest.json'
LIBRARY = ROOT / 'data/v3.0/library.json'
B_PROTOCOL = BASE / 'protocols/b-v1.json'
COMPARISON = BASE / 'protocols/f-v2-mixed-b01-validation-cpu-v2.json'
B00_MANIFEST = ROOT / 'models/next_va_v2/research-reference-v1/manifest.json'
SEEDS = (20261003, 20261004, 20261005)
B_PROTOCOL_HASH = '5ee479f2740f9fd2ddff0f11ed9bd381e70477ecaccb9493d686875f8bf09e6b'
PROTECTED = {
    'models/emotion_model.pth': 'f0c304aec8120ba09e21b2540cc5ae0de23a1536c449673edd0388b896e31667',
    'models/next_va/teacher-20261003-recovery/best.pth': '73aedc4c76b6add3eedbfc9b4c56ccec60fb01956865b5dd042b86173d20d67b',
    'models/next_va/mixed-20261003-recovery/best.pth': 'bcc942e4955b90125c75dd0a6f4f5fa927f2119cc07da7d5ecb54423288e337d',
}


def contained(path):
    value = (ROOT / path).resolve()
    if not value.is_relative_to(ROOT):
        raise ValueError('V3 files must stay inside Vm')
    return value


def relative(path):
    return contained(path).relative_to(ROOT).as_posix()


def array_digest(value):
    import numpy as np
    x = np.asarray(value)
    return hashlib.sha256(str(x.dtype).encode() + repr(x.shape).encode() + x.tobytes(order='C')).hexdigest()


def source_hashes():
    return {relative(p): sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def _source_metadata(b):
    source = contained(b['source_manifest'])
    if sha256(source) != b['manifest_sha256']:
        raise ValueError('Frozen source metadata changed')
    fields = ('song_id', 'audio_sha256', 'audio_path', 'split', 'group_id', 'source_dataset',
              'duration_seconds', 'sample_rate', 'channels', 'decoded_frames')
    # No dynamic label arrays or final-test audio are opened. Only identity and
    # geometry fields are retained from the prepared manifest.
    rows = []
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            rows.append({key: row.get(key) for key in fields})
    if sum(row['split'] == 'test' for row in rows) != 622:
        raise ValueError('Final-test metadata membership changed')
    return rows


def prepare_manifest(path=MANIFEST):
    """Create once; an existing manifest must verify, never silently refresh."""
    import numpy as np
    import torch
    from tools.train_next.features import feature_spec, local_clap_snapshot

    path = contained(path)
    if path.exists():
        validate_manifest(path)
        return path
    verify_hashes(PROTECTED)
    if sha256(B_PROTOCOL) != B_PROTOCOL_HASH:
        raise ValueError('Frozen B protocol changed')
    b, prior, reference = read_json(B_PROTOCOL), read_json(COMPARISON), read_json(B00_MANIFEST)
    if (prior['B_protocol_sha256'] != B_PROTOCOL_HASH or reference['B_protocol_sha256'] != B_PROTOCOL_HASH
            or prior.get('gpu_legacy_port_accepted') is not False):
        raise ValueError('Frozen independent inference provenance changed')
    snapshot = local_clap_snapshot()
    if feature_spec(snapshot) != b['feature_spec']:
        raise ValueError('Use the pinned independent Python and feature environment')
    pinned = {**PROTECTED, **source_hashes()}
    systems = {}
    for kind, source in [('b01', prior['models']['b01']), ('b00', reference)]:
        expected = [f'{"b01-branch-fusion" if kind == "b01" else "b00-reference"}-s{s}' for s in SEEDS]
        if [m['run_id'] for m in source['members']] != expected:
            raise ValueError('Frozen member identities changed')
        members = []
        scaler = None
        for member in source['members']:
            checkpoint, config_path = contained(member['checkpoint']), contained(member['config'])
            if sha256(checkpoint) != member['checkpoint_sha256']:
                raise ValueError('Previously audited checkpoint changed: ' + member['run_id'])
            expected_config = (prior if kind == 'b01' else reference)['pinned_hashes'][member['config']]
            if sha256(config_path) != expected_config:
                raise ValueError('Previously audited config changed')
            config = read_json(config_path)
            if config['protocol_sha256'] != B_PROTOCOL_HASH or config['feature_key'] != b['feature_key']:
                raise ValueError('Member trained on a different protocol or feature version')
            verify_hashes(config['code'])
            pinned.update(config['code'])
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if (saved['provenance'] != config or saved['settings'] != config['settings']
                    or saved['label_order'] != ['valence', 'arousal']
                    or saved['architecture'] != 'whole-song-va-experiment-v2'):
                raise ValueError('Checkpoint metadata differs from audited member')
            mean, std = saved['mean'].numpy(), saved['std'].numpy()
            if (mean.shape != (531,) or std.shape != (531,) or mean.dtype != np.float32
                    or std.dtype != np.float32 or not np.isfinite(mean).all()
                    or not np.isfinite(std).all() or np.any(std <= 0)):
                raise ValueError('Frozen training scaler is invalid')
            if scaler is not None and (not np.array_equal(scaler[0], mean) or not np.array_equal(scaler[1], std)):
                raise ValueError('Members disagree on their frozen training scaler')
            scaler = (mean.copy(), std.copy())
            members.append({**member, 'scaler_mean_sha256': array_digest(mean), 'scaler_std_sha256': array_digest(std)})
            pinned[relative(checkpoint)] = member['checkpoint_sha256']
            pinned[relative(config_path)] = expected_config
            del saved
        systems[kind] = dict(members=members, aggregation='arithmetic_mean_of_three_fixed_seeds',
                             disagreement='sample_standard_deviation_ddof_1_not_confidence',
                             calibration='unavailable')
    # Verify only code/model provenance, not the old protocol's label/584-audio pins.
    for origin in (prior['pinned_hashes'], reference['pinned_hashes']):
        for name, digest in origin.items():
            if name.endswith('.py') and not '/tests/' in name:
                pinned[name] = digest
    for item in [B_PROTOCOL, COMPARISON, B00_MANIFEST, contained(b['source_manifest'])]:
        pinned[relative(item)] = sha256(item)
    for item in sorted(snapshot.iterdir()):
        if item.is_file() and item.suffix in ('.json', '.txt', '.safetensors'):
            pinned[relative(item)] = sha256(item)
    metadata = _source_metadata(b)
    test_groups = {r['group_id'] for r in metadata if r['split'] == 'test'}
    excluded = {r['audio_sha256'] for r in metadata if r['split'] == 'test' or r['group_id'] in test_groups}
    verify_hashes(pinned)
    value = dict(schema_version=3, id='v3-preview-v1', created_at=now(),
        status='research_preview_not_promoted', training_enabled=False, promoted_model=False,
        B_protocol=relative(B_PROTOCOL), B_protocol_sha256=B_PROTOCOL_HASH,
        source_manifest=relative(b['source_manifest']), source_manifest_sha256=b['manifest_sha256'],
        label_order=['valence', 'arousal'], feature_key=b['feature_key'], feature_spec=b['feature_spec'],
        systems=systems, legacy=dict(checkpoint='models/emotion_model.pth', execution_device='cpu',
            snapshot=relative(snapshot), seed=42, batch_size=16, hop_seconds=.1,
            source='unchanged versions/v2.0 EmotionInterface, original CPU features',
            gpu_port_accepted=False),
        excluded_final_test_audio_sha256=sorted(excluded), final_test_songs=622,
        pinned_hashes=pinned, inference_scope='offline whole-song bidirectional; native zero; no time warp',
        calibration=dict(status='unavailable', reason='B00 audit failed; no B01 calibration transfer permitted'),
        tension=dict(status='unavailable', reason='No validated psychological supervision'),
        limitations=['B01 failed per-dataset/axis non-inferiority; this preview cannot authorize model replacement.',
                    'Seed disagreement is not calibrated accuracy or confidence.',
                    'No training, label fitting, final-test evaluation, or new dataset ingestion.'])
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)
    return path


def validate_manifest(path=MANIFEST):
    from tools.train_next.features import feature_spec, local_clap_snapshot
    path = contained(path)
    m = read_json(path)
    if (m.get('schema_version') != 3 or m.get('id') != 'v3-preview-v1'
            or m.get('training_enabled') is not False or m.get('promoted_model') is not False
            or m.get('label_order') != ['valence', 'arousal']
            or m.get('calibration', {}).get('status') != 'unavailable'
            or set(m.get('systems', {})) != {'b01', 'b00'}
            or m.get('legacy', {}).get('execution_device') != 'cpu'
            or m.get('legacy', {}).get('gpu_port_accepted') is not False
            or m.get('tension', {}).get('status') != 'unavailable'
            or m.get('final_test_songs') != 622):
        raise ValueError('Unsupported V3 manifest contract')
    verify_hashes(m['pinned_hashes'])
    # Hash validation alone does not protect a changed list of selected members,
    # aggregation rule or feature key inside an otherwise parseable manifest.
    b, prior, reference = read_json(B_PROTOCOL), read_json(COMPARISON), read_json(B00_MANIFEST)
    if (m['feature_key'] != b['feature_key'] or m['feature_spec'] != b['feature_spec']
            or m['B_protocol'] != relative(B_PROTOCOL)
            or m['source_manifest'] != relative(b['source_manifest'])
            or m['source_manifest_sha256'] != b['manifest_sha256']):
        raise ValueError('Manifest feature/source contract changed')
    mandatory = {**PROTECTED, **source_hashes()}
    for source_path in (B_PROTOCOL, COMPARISON, B00_MANIFEST, contained(b['source_manifest'])):
        mandatory[relative(source_path)] = sha256(source_path)
    for kind, original in [('b01', prior['models']['b01']), ('b00', reference)]:
        selected = m['systems'][kind]
        expected_members = original['members']
        if (selected.get('aggregation') != 'arithmetic_mean_of_three_fixed_seeds'
                or selected.get('disagreement') != 'sample_standard_deviation_ddof_1_not_confidence'
                or selected.get('calibration') != 'unavailable'
                or len(selected.get('members', [])) != 3):
            raise ValueError('Model aggregation/calibration contract changed')
        for member, expected in zip(selected['members'], expected_members):
            if any(member.get(key) != value for key, value in expected.items()):
                raise ValueError('Selected model member differs from frozen audited identity')
            for key in ('scaler_mean_sha256', 'scaler_std_sha256'):
                value = member.get(key)
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('Invalid frozen scaler digest')
            origin = prior if kind == 'b01' else reference
            mandatory[member['checkpoint']] = expected['checkpoint_sha256']
            mandatory[member['config']] = origin['pinned_hashes'][member['config']]
            mandatory.update(read_json(contained(member['config']))['code'])
    for origin in (prior['pinned_hashes'], reference['pinned_hashes']):
        mandatory.update({name:digest for name,digest in origin.items()
                          if name.endswith('.py') and '/tests/' not in name})
    snapshot = local_clap_snapshot()
    for item in sorted(snapshot.iterdir()):
        if item.is_file() and item.suffix in ('.json', '.txt', '.safetensors'):
            mandatory[relative(item)] = sha256(item)
    if m['pinned_hashes'] != mandatory:
        raise ValueError('Required immutable model/code inputs changed')
    legacy = m['legacy']
    if (legacy.get('checkpoint') != 'models/emotion_model.pth'
            or legacy.get('snapshot') != relative(snapshot) or legacy.get('seed') != 42
            or legacy.get('batch_size') != 16 or legacy.get('hop_seconds') != .1):
        raise ValueError('Original CPU V2 execution contract changed')
    metadata = _source_metadata(b)
    test_groups = {r['group_id'] for r in metadata if r['split'] == 'test'}
    excluded = sorted({r['audio_sha256'] for r in metadata
                       if r['split'] == 'test' or r['group_id'] in test_groups})
    if m['excluded_final_test_audio_sha256'] != excluded:
        raise ValueError('Sealed final-test identity exclusion changed')
    if feature_spec(local_clap_snapshot()) != m['feature_spec']:
        raise ValueError('Extractor/dependency environment changed')
    if m['B_protocol_sha256'] != B_PROTOCOL_HASH or sha256(B_PROTOCOL) != B_PROTOCOL_HASH:
        raise ValueError('Training protocol binding changed')
    return m


def build_inventory(output=LIBRARY, max_songs=24):
    """Pin up to 24 existing files: known 18 comparison songs plus asset coverage."""
    import soundfile as sf
    if type(max_songs) is not int or not 1 <= max_songs <= 24:
        raise ValueError('Inventory size must be between 1 and 24')
    if sha256(B_PROTOCOL) != B_PROTOCOL_HASH:
        raise ValueError('B metadata protocol changed')
    b = read_json(B_PROTOCOL)
    all_rows = _source_metadata(b)
    test_groups = {r['group_id'] for r in all_rows if r['split'] == 'test'}
    blocked = {r['audio_sha256'] for r in all_rows if r['split'] == 'test' or r['group_id'] in test_groups}
    prior = read_json(COMPARISON)
    reps = {r['song_id'] for r in prior['representatives']}
    chosen, seen = [], set()
    for row in prior['validation_songs']:
        if row['song_id'] not in reps:
            continue
        path, digest = contained(row['audio_path']), row['audio_sha256']
        if row['split'] != 'validation' or digest in blocked or digest in seen:
            continue
        if sha256(path) != digest:
            raise ValueError('Existing comparison audio changed')
        chosen.append(dict(id=digest[:24], path=relative(path), audio_path=relative(path),
            sha256=digest, audio_sha256=digest, name=path.name, song_id=row['song_id'],
            duration_seconds=row['duration_seconds'], sample_rate=row['sample_rate'], channels=row['channels'],
            selection='previously frozen illustrative validation song; not new accuracy evidence',
            source='existing_comparison', native_geometry_verified=False))
        seen.add(digest)
    candidates = []
    for directory in ('assets/audio/demo', 'assets/audio/test', 'assets/audio/showdemo', 'showdemo'):
        folder = ROOT / directory
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in ('.mp3', '.flac', '.wav', '.ogg'):
                continue
            digest = sha256(path)
            if digest in blocked or digest in seen:
                continue
            seen.add(digest)
            try:
                info = sf.info(path)
            except Exception:
                continue
            candidates.append(dict(id=digest[:24], path=relative(path), audio_path=relative(path),
                sha256=digest, audio_sha256=digest, name=path.name,
                duration_seconds=info.frames / info.samplerate, sample_rate=info.samplerate, channels=info.channels,
                selection='existing asset selected by format and metadata duration coverage',
                source='existing_asset', native_geometry_verified=False))
    chosen = sorted(chosen, key=lambda r: r['song_id'])[:max_songs]
    # First include the duration extremes of each existing format, then spread
    # remaining slots across duration. Never consult model scores or labels.
    ordered = sorted(candidates, key=lambda r: (r['duration_seconds'], r['sha256']))
    selected_assets = []
    for suffix in sorted({Path(r['path']).suffix.lower() for r in ordered}):
        group = [r for r in ordered if Path(r['path']).suffix.lower() == suffix]
        for row in (group[0], group[-1]):
            if row not in selected_assets:
                selected_assets.append(row)
    if ordered:
        for fraction in (.25, .5, .75):
            row = ordered[round((len(ordered) - 1) * fraction)]
            if row not in selected_assets:
                selected_assets.append(row)
    selected_assets.extend(r for r in ordered if r not in selected_assets)
    chosen += selected_assets[:max_songs - len(chosen)]
    result = dict(schema_version=3, id='v3-existing-audio-v1', created_at=now(), songs=chosen,
        source_manifest_sha256=b['manifest_sha256'], comparison_protocol_sha256=sha256(COMPARISON),
        final_test_songs_excluded=622, final_test_labels_loaded=False, new_annotation_audio_used=False,
        selection='Up to 18 already published illustrative validation songs plus existing assets by format/duration; engineering checks only.',
        duration_note='Metadata duration for selection only; analyze verifies actual native EOF before inference.')
    if output is not None:
        output = contained(output)
        if output.exists():
            old = read_json(output)
            # Do not silently change a frozen engineering cohort or its paths.
            if old.get('songs') != result['songs'] or old.get('source_manifest_sha256') != result['source_manifest_sha256']:
                raise ValueError('Existing V3 inventory changed; version it explicitly')
            return old
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, result)
    return result
