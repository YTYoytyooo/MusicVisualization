"""Single Studio 2.0 entry point. Heavy imports are lazy; never trains implicitly."""
import argparse
import json
from pathlib import Path
import sys


def main():
    from studio.store import DEFAULT_PROJECTS, read_json, save_revision
    parser = argparse.ArgumentParser(description='MusicVisualization Studio 2.0 — 编辑、修订、重新生成')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='检查运行环境，不修改任何文件')
    web = sub.add_parser('serve', help='启动本地时间线编辑器')
    web.add_argument('--projects', type=Path, default=DEFAULT_PROJECTS)
    web.add_argument('--port', type=int, default=8765)
    analysis = sub.add_parser('analyze', help='分析音乐并保存原始数值，不自动训练')
    analysis.add_argument('audio', nargs='+')
    analysis.add_argument('--model', required=True)
    analysis.add_argument('--projects', type=Path, default=DEFAULT_PROJECTS)
    analysis.add_argument('--seed', type=int, default=42)
    edit = sub.add_parser('edit', help='从 JSON 编辑操作创建新修订')
    edit.add_argument('project', type=Path)
    edit.add_argument('--base', required=True)
    edit.add_argument('--edits', type=Path, required=True)
    edit.add_argument('--motion', type=Path, help='可选运动配置 JSON；省略时继承基准修订')
    render = sub.add_parser('render', help='从保存的分析和修订生成视频，不运行 CLAP')
    render.add_argument('project', type=Path)
    render.add_argument('--revision')
    render.add_argument('--mode', choices=['analysis', 'presentation'], default='analysis')
    render.add_argument('--start', type=float, default=0)
    render.add_argument('--end', type=float)
    render.add_argument('--fps', type=int, default=30)
    render.add_argument('--width', type=int, default=1280)
    render.add_argument('--height', type=int, default=720)
    demo = sub.add_parser('demo', help='创建明确标注的合成演示项目，不下载模型')
    demo.add_argument('--projects', type=Path, default=DEFAULT_PROJECTS)
    demo.add_argument('--duration', type=float, default=8.)
    worker = sub.add_parser('_worker', help=argparse.SUPPRESS)
    worker.add_argument('--projects', type=Path, required=True)
    worker.add_argument('--job', required=True)
    args = parser.parse_args()
    def progress(stage, ratio):
        print(f'{stage}: {ratio:.0%}', flush=True)
    try:
        if args.command == 'doctor':
            from studio.pipeline import doctor
            result = doctor()
        elif args.command == 'serve':
            from studio.server import serve
            serve(args.projects, args.port)
            return 0
        elif args.command == 'analyze':
            from studio.pipeline import analyze
            result = [str(analyze(audio, args.model, args.projects, seed=args.seed, progress=progress)) for audio in args.audio]
        elif args.command == 'edit':
            from studio.pipeline import prepare_visual_plan, validate_preview
            from studio.store import load_project, load_revision
            edits = read_json(args.edits)
            if load_project(args.project)['current_revision'] != args.base:
                raise ValueError('项目版本已变化，请重新加载后再保存')
            previous = load_revision(args.project, args.base)
            smoothing = previous.get('smoothing')
            motion = read_json(args.motion) if args.motion else previous.get('motion')
            if any(e.get('layer') == 'visual' and e.get('enabled', True) for e in edits):
                prepare_visual_plan(args.project, args.base, edits, progress=progress, smoothing=smoothing)
            validate_preview(args.project, edits, smoothing=smoothing, motion=motion)
            result = save_revision(args.project, args.base, edits, smoothing=smoothing, motion=motion)
        elif args.command == 'render':
            from studio.pipeline import render
            options = {k: getattr(args, k) for k in ('mode', 'start', 'end', 'fps', 'width', 'height')}
            result = render(args.project, args.revision, options, progress=progress)
        elif args.command == 'demo':
            from studio.demo import create_demo
            result = str(create_demo(args.projects, args.duration))
        else:
            from studio.jobs import run_worker
            return run_worker(args.projects, args.job)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
