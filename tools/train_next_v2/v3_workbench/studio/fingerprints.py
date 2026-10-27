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
    libraries = {'planning': ('numpy',),
                 'render': ('numpy','opencv-python')}
    if stage not in libraries:
        raise ValueError('工作台仅规划和渲染；预测凭证由冻结推理桥接提供')
    packages = {}
    for name in libraries[stage]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    if stage == 'planning':
        code = {f: digest(ROOT / f) for f in ('mcts.py','studio/timeline.py','studio/smoothing.py')}
        code['planner'] = code_parts('studio/pipeline.py', ['plan', 'planning_key', '_require_observations'])
    else:
        code = {f: digest(ROOT / f) for f in ('renderer.py','prediction_overlay.py','studio/timeline.py','studio/smoothing.py','legacy_pipeline.py',
                                             'motion_schema.py','motion_renderer.py','studio/motion.py','studio/motion_features.py',
                                             'studio/checkpoints.py','feature_extraction.py')}
        code['render'] = code_parts('studio/pipeline.py', ['render','visual_at','validate_render_options'])
    return object_digest({'schema': 1, 'stage': stage, 'code': code, 'packages': packages})
