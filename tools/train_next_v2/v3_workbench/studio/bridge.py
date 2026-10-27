"""Copy verified existing analysis into mutable works without editing receipts.

No labels are loaded and no model training is performed. Frozen prediction
projects retain their original bytes and directory; works own portable audio,
native two-dimensional arrays, edits, revisions and outputs separately.
"""
from pathlib import Path
import shutil
import uuid
import numpy as np

from .paths import DEFAULT_PROJECTS, DATA_ROOT, WORKSPACE, migrated_path
from .store import (atomic_json, create_project, digest, load_project, load_raw,
                    load_revision, object_digest, read_json, save_revision, _metadata,
                    _contained_file)


def _publish(path):
    p = _metadata(path)
    target = Path(path).parent/p['id']
    if target.exists():
        raise ValueError('作品目标已存在，不覆盖')
    Path(path).rename(target)
    return target


def _snapshot_files(source, target):
    """Copy ordinary files only, reject links and verify every copied byte."""
    source, target = Path(source).resolve(), Path(target)
    hashes = {}
    for file in sorted(source.rglob('*')):
        if file.is_symlink() or not file.resolve().is_relative_to(source):
            raise ValueError('导入源包含链接或越界路径，请先整理为普通本地项目')
        if file.is_file() and file.name != '.write.lock':
            relative = file.relative_to(source)
            output = target/relative
            expected = digest(file)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, output)
            if digest(output) != expected or digest(file) != expected:
                raise ValueError('导入期间源文件改变：'+relative.as_posix())
            hashes[relative.as_posix()] = expected
    return hashes


def _resolve_audio(record):
    original = Path(record['path'])
    original = original if original.is_absolute() else WORKSPACE/original
    candidates = [original, migrated_path(original)]
    for candidate in candidates:
        if candidate is not None and candidate.is_file() and digest(candidate) == record['sha256']:
            return candidate.resolve()
    # Relocation is a separate verified mapping, never a rewrite of receipts.
    mapping = DATA_ROOT/'audio-relocations.json'
    if mapping.exists():
        entries = read_json(mapping)
        if not isinstance(entries, dict):
            raise ValueError('音频迁移记录格式错误')
        locations = entries.get(record['sha256'], [])
        if not isinstance(locations, list) or any(not isinstance(value, str) for value in locations):
            raise ValueError('音频迁移位置必须为路径列表')
        for value in locations:
            candidates.append(workspace_path(value))
    for work in (DATA_ROOT/'works').glob('p-*/project.json'):
        try:
            meta = read_json(work)
        except (ValueError, OSError):
            continue
        if meta.get('audio_sha256') == record['sha256']:
            candidates.append(_contained_file(work.parent, meta['audio_file']))
    for candidate in candidates:
        if candidate is not None and candidate.is_file() and digest(candidate) == record['sha256']:
            return candidate.resolve()
    raise ValueError('音频缺失或内容已变；需提供与凭证 SHA256 一致的音频位置')


def workspace_path(value):
    path = (WORKSPACE/value).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError('音频迁移目标必须保存在 Vm 内')
    return path


def _verified_analysis(directory):
    from tools.train_next_v2.v3_inference.manifest import MANIFEST, validate_manifest
    from tools.train_next_v2.v3_inference.pipeline import verify_result, validate_output_system
    directory = Path(directory).resolve()
    project, result, receipt = (read_json(directory/name) for name in
                               ('project.json', 'analysis/result.json', 'analysis/receipt.json'))
    manifest = validate_manifest(MANIFEST)
    mh = digest(MANIFEST)
    if (receipt.get('schema_version') != 3 or project.get('schema_version') != 3
            or result.get('schema_version') != 3 or project.get('id') != directory.name
            or result.get('project_id') != directory.name or project.get('audio') != result.get('audio')
            or receipt.get('project_sha256') != digest(directory/'project.json')
            or receipt.get('result_sha256') != digest(directory/'analysis/result.json')
            or any(v.get('manifest_sha256') != mh for v in (project, result, receipt))
            or receipt.get('audio_sha256') != result['audio']['sha256']):
        raise ValueError('冻结分析凭证、模型或项目身份不匹配')
    if result['audio']['sha256'] in manifest['excluded_final_test_audio_sha256']:
        raise ValueError('最终测试音频保持封存，不导入作品')
    audio = _resolve_audio(result['audio'])
    original = workspace_path(result['audio']['path'])
    if original.is_file() and digest(original) == result['audio']['sha256']:
        verify_result(directory, manifest_path=MANIFEST)
    else:
        # Frozen verifier intentionally requires the historical path. Verify
        # unchanged receipts plus the independently hash-matched replacement.
        if not {'b01', 'b00'}.issubset(result['systems']) or set(result['systems'])-{'b01', 'b00', 'v2'}:
            raise ValueError('冻结模型集合不匹配')
        for name, system in result['systems'].items():
            validate_output_system(system, result['audio']['duration_seconds'], name == 'v2')
        for field in ('calibration', 'tension'):
            if result[field].get('status') != 'unavailable' or any(result[field].get(k) is not None for k in ('values', 'intervals')):
                raise ValueError('未验证输出不能携带伪造数值')
        acoustic = result['acoustic']
        clock = result['systems']['b01']['times_seconds']
        if acoustic['times_seconds'] != clock:
            raise ValueError('声学时间轴不匹配')
        for field in ('rms_dbfs', 'rms_amplitude', 'spectral_centroid_hz'):
            values = np.asarray(acoustic[field])
            if values.shape != (len(clock),) or not np.isfinite(values).all() or (field != 'rms_dbfs' and np.any(values < 0)):
                raise ValueError('声学数值非法')
        if (acoustic.get('dbfs_floor_amplitude', 1e-8) != 1e-8
                or not np.allclose(acoustic['rms_dbfs'], 20*np.log10(np.maximum(acoustic['rms_amplitude'], 1e-8)), rtol=0, atol=1e-10)):
            raise ValueError('声学 dBFS 与 RMS 不匹配')
    return project, result, receipt, manifest, audio


def _work_rhythm(audio, native_duration):
    """Reusable acoustic tempo/beats, independent of frozen emotion outputs."""
    from importlib.metadata import version
    from tools.train_next_v2.v3_workbench.feature_extraction import load_audio, extract_global_info
    fingerprint = dict(schema_version=1, audio_sha256=digest(audio), native_duration=float(native_duration),
                       extractor_sha256=digest(Path(__file__).parent.parent/'feature_extraction.py'),
                       libraries={name: version(name) for name in ('numpy', 'librosa', 'soundfile')})
    key = object_digest(fingerprint)
    file = DATA_ROOT/'cache/workbench-rhythm'/(key+'.json')
    if file.exists():
        record = read_json(file)
        if (record.get('identity') != fingerprint or record.get('sha256') !=
                object_digest({k: v for k, v in record.items() if k != 'sha256'})):
            raise ValueError('节拍声学缓存校验失败')
    else:
        y, rate = load_audio(str(audio))
        global_info = extract_global_info(y, rate)
        tempo = float(global_info['tempo'])
        beats = np.asarray(global_info['beat_times'], dtype=np.float64)
        if (not np.isfinite(tempo) or tempo < 0 or beats.ndim != 1 or not np.isfinite(beats).all()
                or np.any(np.diff(beats) <= 0) or abs(len(y)/rate-native_duration) > 1/rate+1e-6):
            raise ValueError('节拍分析时钟或数据非法；不能拉伸原生音频时间')
        inside = (beats >= 0) & (beats < native_duration)
        record = dict(identity=fingerprint, tempo=tempo, beat_times=beats[inside].tolist(),
                      decoded_duration=len(y)/rate, discarded_outside_audio=int((~inside).sum()),
                      semantics='acoustic beat tracking for visual rhythm; not emotion labels',
                      model_inference_performed=False, training_performed=False)
        if digest(audio) != fingerprint['audio_sha256']:
            raise ValueError('节拍提取期间音频改变')
        record['sha256'] = object_digest(record)
        atomic_json(file, record)
    return (dict(duration=native_duration, tempo=record['tempo'], beat_times=record['beat_times']),
            dict(cache=str(file), sha256=digest(file), identity=fingerprint,
                 semantics=record['semantics']))


def import_analysis(analysis_dir, root=DEFAULT_PROJECTS, name=None, source_system='b01', seed=42):
    directory = Path(analysis_dir).resolve()
    if Path(root).resolve().is_relative_to(directory):
        raise ValueError('作品目录不能位于冻结分析目录内')
    project, result, receipt, manifest, audio = _verified_analysis(directory)
    global_info, rhythm_provenance = _work_rhythm(audio, result['audio']['duration_seconds'])
    systems = {}
    for key, system in result['systems'].items():
        identity = manifest['systems'][key] if key != 'v2' else manifest['legacy']
        systems[key] = dict(values=system['raw_va'],
                            hold_last_to_end=key == 'v2',
                            times_us=np.rint(np.asarray(system['times_seconds'])*1e6).astype(np.int64),
                            step_us=100000 if key == 'v2' else 500000,
                            label=key.upper()+' 原始预测', alignment='native-decoded-sample-zero',
                            model_path=str(WORKSPACE/'models/next_va_v2/v3-preview-v1/manifest.json'),
                            model_sha256=object_digest(identity))
    provenance = dict(kind='frozen-v3-analysis', source_project=str(directory),
                      source_project_sha256=receipt['project_sha256'], source_result_sha256=receipt['result_sha256'],
                      source_receipt_sha256=digest(directory/'analysis/receipt.json'),
                      manifest_sha256=result['manifest_sha256'],
                      frozen_result_file='analysis/source-v3/analysis/result.json',
                      audio_resolution={'original': result['audio']['path'], 'resolved': str(audio), 'sha256': result['audio']['sha256']},
                      work_rhythm=rhythm_provenance,
                      training_performed=False, label_values_loaded=False)
    work = create_project(root, name or result['audio']['name'], audio, None, None,
                          global_info, provenance=provenance, seed=seed,
                          systems=systems, source_system=source_system, publish=False)
    archived = work/'analysis/source-v3'
    archived.mkdir(parents=True)
    # Only frozen inputs/outputs are analysis provenance; UI exports are not.
    hashes = {}
    for relative in ('project.json', 'analysis/result.json', 'analysis/receipt.json'):
        destination = archived/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = digest(directory/relative)
        shutil.copy2(directory/relative, destination)
        if digest(destination) != expected or digest(directory/relative) != expected:
            raise ValueError('导入时冻结分析发生改变')
        hashes[relative] = expected
    if hashes['project.json'] != receipt['project_sha256'] or hashes['analysis/result.json'] != receipt['result_sha256']:
        raise ValueError('导入源与已验证凭证改变')
    if read_json(archived/'analysis/receipt.json') != receipt:
        raise ValueError('导入期间冻结分析凭证发生改变')
    atomic_json(work/'analysis/source-v3-copy.json', {'source': str(directory), 'sha256': hashes})
    return _publish(work)


def import_v2(source_dir, root=DEFAULT_PROJECTS, name=None):
    source = Path(source_dir).resolve()
    if Path(root).resolve().is_relative_to(source):
        raise ValueError('新作品目录不能位于 V2 源项目内部')
    source_project_hash = digest(source/'project.json')
    p = read_json(source/'project.json')
    if p.get('schema_version') != 1:
        raise ValueError('仅接受已完成的 V2 schema 1 项目')
    if not p.get('current_revision'):
        raise ValueError('旧项目分析未完成')
    source_provenance = p.get('provenance', {})
    if not isinstance(source_provenance, dict):
        raise ValueError('V2 来源元数据格式错误')
    is_synthetic = (source_provenance.get('kind') in ('synthetic', 'demo', 'test')
                    or source_provenance.get('is_synthetic') is True)
    audio = _contained_file(source, p['audio_file'])
    if digest(audio) != p['audio_sha256']:
        raise ValueError('V2 音频校验失败')
    raw = load_raw(source)
    revisions = [load_revision(source, path.parent.name) for path in sorted((source/'revisions').glob('r*/revision.json'))]
    if p['current_revision'] not in {r['id'] for r in revisions}:
        raise ValueError('旧项目当前修订不存在')
    systems = {'v2': dict(values=np.asarray(raw), times_us=raw.times_us, mask=raw.mask, step_us=p['step_us'],
                         hold_last_to_end=True,
                         model_path=p.get('model_path', ''), model_sha256=p.get('model_sha256'),
                         label='V2 历史预测', alignment=p.get('alignment', 'window-last-frame-v2'))}
    work = create_project(root, name or p['name'], audio, None, None,
                          dict(duration=p['duration'], tempo=p.get('tempo', 0), beat_times=p.get('beat_times', [])),
                          seed=p.get('seed', 42), systems=systems, source_system='v2', publish=False,
                          provenance=dict(kind='v2-project-import', source_project=str(source),
                                          source_provenance=source_provenance, is_synthetic=is_synthetic,
                                          legacy_archive='archive/v2-original', training_performed=False))
    hashes = _snapshot_files(source, work/'archive/v2-original')
    if (hashes.get('project.json') != source_project_hash
            or hashes.get('analysis/predictions_raw.npz') != p['raw_sha256']
            or hashes.get(p['audio_file']) != p['audio_sha256']):
        raise ValueError('V2 项目在导入期间改变，拒绝发布不一致的副本')
    mapping = {}
    # Preserve exact old parent relationships and all supported operations.
    from .timeline import validate_edits
    from .smoothing import validate_smoothing, check_times, emotion_values
    from tools.train_next_v2.v3_workbench.motion_schema import validate_config
    wp = _metadata(work)
    for original in revisions:
        allowed = {'schema_version', 'id', 'parent', 'created_at', 'analysis_sha256', 'edits', 'smoothing', 'motion', 'sha256'}
        if set(original)-allowed:
            raise ValueError('旧修订含未支持字段：'+','.join(sorted(set(original)-allowed)))
        for edit in original['edits']:
            allowed_edit = {'id', 'layer', 'field', 'operation', 'value', 'time_us', 'start_us', 'end_us',
                            'transition_in_us', 'transition_out_us', 'enabled', 'interpolation', 'note'}
            if set(edit)-allowed_edit:
                raise ValueError('旧编辑含未支持字段：'+','.join(sorted(set(edit)-allowed_edit)))
        archived_revision = read_json(work/'archive/v2-original/revisions'/original['id']/'revision.json')
        if archived_revision != original:
            raise ValueError('旧修订在导入时发生改变')
        edits = validate_edits(original['edits'], wp['duration_us'])
        smoothing = validate_smoothing(original.get('smoothing'))
        motion = validate_config(original.get('motion'), wp['duration_us'])
        ts = check_times(raw, wp['duration_us'], edits, p['step_us'], smoothing)
        emotion_values(raw, ts, edits, p['step_us'], smoothing)
        new_id = f'r{int(original["id"][1:])+1:06d}'
        parent = original.get('parent')
        if parent is not None and parent not in mapping:
            raise ValueError('旧修订父版本缺失或顺序非法')
        row = dict(schema_version=2, id=new_id, parent=mapping.get(parent, 'r000000'),
                   created_at=original['created_at'], analysis_sha256=wp['systems']['v2']['raw_sha256'],
                   source_system='v2', model_sha256=wp['systems']['v2']['model_sha256'],
                   edits=edits, smoothing=smoothing, motion=motion,
                   imported_from={'revision': original['id'], 'sha256': original['sha256']})
        row['parent_sha256'] = load_revision(work, row['parent'])['sha256']
        row['sha256'] = object_digest(row)
        atomic_json(work/'revisions'/new_id/'revision.json', row)
        mapping[original['id']] = new_id
    wp['current_revision'] = mapping[p['current_revision']]
    wp['provenance']['migration_report'] = 'migration.json'
    atomic_json(work/'project.json', wp)
    atomic_json(work/'migration.json', dict(source=str(source), archive_sha256=hashes, revision_mapping=mapping,
                source_current_revision=p['current_revision'], current_revision=wp['current_revision'],
                preserved='Original five dimensions, plans, previews and renders remain byte-identical in archive',
                warnings=['Historical rendered videos remain in the archive and are not relabelled as new V3 output']))
    return _publish(work)


def analyze_existing(audio_path, root=DEFAULT_PROJECTS, name=None, device='cpu', include_v2=False,
                     progress=None, cancel=None, seed=42):
    from tools.train_next_v2.v3_inference.pipeline import analyze
    audio = workspace_path(audio_path)
    expected = digest(audio)
    for metadata in sorted((DATA_ROOT/'projects').glob('*/project.json')):
        p = read_json(metadata)
        if p.get('audio', {}).get('sha256') == expected and (metadata.parent/'analysis/receipt.json').is_file():
            result = read_json(metadata.parent/'analysis/result.json')
            if include_v2 and 'v2' not in result.get('systems', {}):
                continue
            if cancel and cancel():
                raise InterruptedError('用户取消分析')
            # Invalid matching receipts must be surfaced, not silently replaced.
            return import_analysis(metadata.parent, root, name, seed=seed)
    directory = DATA_ROOT/'projects'/('p-workbench-'+uuid.uuid4().hex[:12])
    def notify(record):
        if progress:
            ratio = {'verify_manifest': .01, 'native_decode': .05, 'features': .15,
                     'inference': .6, 'save_raw_result': .9, 'completed': 1.}.get(record['stage'], .1)
            if getattr(progress, 'accepts_details', False):
                progress(record['stage'], ratio, record)
            else:
                progress(record['stage'], ratio)
    analyze(audio, directory, device=device, include_v2=include_v2, progress=notify, cancelled=cancel)
    return import_analysis(directory, root, name, seed=seed)
