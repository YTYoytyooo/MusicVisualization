"""Stage-specific, content-based provenance. Legacy caches remain untouched."""
import ast
import os
from pathlib import Path
from .store import ROOT, digest, object_digest


def code_parts(filename, names):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8-sig'))
    nodes = [ast.dump(node, include_attributes=False) for node in tree.body
             if getattr(node, 'name', None) in names]
    if len(nodes) != len(names):
        raise ValueError('阶段代码指纹缺少定义: ' + filename)
    return object_digest(nodes)


def code_constants(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8-sig'))
    return object_digest([ast.dump(node,include_attributes=False) for node in tree.body
        if isinstance(node,(ast.Assign,ast.AnnAssign)) and any(
            isinstance(target,ast.Name) and target.id.isupper()
            for target in (node.targets if isinstance(node,ast.Assign) else [node.target]))])


def stage_fingerprint(stage):
    from importlib.metadata import version, PackageNotFoundError
    libraries = {'features': ('numpy','librosa','transformers','torch','soxr'),
                 'prediction': ('numpy','torch'), 'planning': ('numpy',),
                 'render': ('numpy','opencv-python')}
    packages = {}
    for name in libraries[stage]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    if stage == 'features':
        code = {'extraction': code_parts('emotion_model.py', ['_load_clap', 'extract_clap_embeddings']),
                'audio': code_parts('feature_extraction.py', ['load_audio'])}
    elif stage == 'prediction':
        code = {'model': code_parts('emotion_model.py', ['EmotionModel', 'EmotionInterface']),
                'constants':code_constants('emotion_model.py')}
    elif stage == 'planning':
        code = {f: digest(ROOT / f) for f in ('mcts.py','studio/timeline.py','studio/smoothing.py')}
        code['planner'] = code_parts('studio/pipeline.py', ['plan'])
    else:
        code = {f: digest(ROOT / f) for f in ('renderer.py','prediction_overlay.py','studio/timeline.py','studio/smoothing.py','legacy_pipeline.py',
                                             'motion_schema.py','motion_renderer.py','studio/motion.py','studio/motion_features.py')}
        code['render'] = code_parts('studio/pipeline.py', ['render','visual_at','validate_render_options'])
    return object_digest({'schema': 1, 'stage': stage, 'code': code, 'packages': packages})


def clap_snapshot():
    """Resolve the exact existing local model; no implicit network/download."""
    hub = Path(os.environ.get('HF_HUB_CACHE', str(Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface'))) / 'hub')))
    roots = [hub, ROOT / '.runtime/huggingface/hub']
    required = ('config.json','model.safetensors','preprocessor_config.json','tokenizer_config.json','tokenizer.json')
    for root in roots:
        repo = root / 'models--laion--clap-htsat-fused'
        ref = repo / 'refs/main'
        if not ref.is_file():
            continue
        revision = ref.read_text().strip()
        if not revision or Path(revision).name != revision:
            continue
        folder = repo / 'snapshots' / revision
        if all((folder/name).is_file() for name in required):
            hashes = {p.name: digest(p) for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.txt','.safetensors')}
            return {'source': str(folder.resolve()), 'revision': revision, 'sha256': object_digest(hashes), 'files': hashes}
    raise ValueError('未找到完整本地CLAP模型。请先准备模型缓存；工作台不会自动下载或训练。')


def feature_key(audio_hash, snapshot, fingerprint=None):
    return object_digest({'audio': audio_hash, 'stage': fingerprint or stage_fingerprint('features'),
                          'clap': snapshot['sha256'], 'sr':22050,'clap_sr':48000,'window':2.,'hop':.1})


def prediction_key(feature, model_hash, fingerprint=None):
    return object_digest({'feature': feature, 'model': model_hash,
                          'stage': fingerprint or stage_fingerprint('prediction'), 'alignment':'window-last-frame-v2'})
