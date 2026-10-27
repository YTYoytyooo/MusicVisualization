"""Independent V3 editing workbench. No command trains or downloads a model."""
import argparse
import json
from pathlib import Path
import sys


def main():
    from .studio.paths import DEFAULT_PROJECTS
    parser = argparse.ArgumentParser(description='MusicVisualization V3 工作台')
    commands = parser.add_subparsers(dest='command', required=True)
    web = commands.add_parser('serve')
    web.add_argument('--projects', type=Path, default=DEFAULT_PROJECTS)
    web.add_argument('--port', type=int, default=8768)
    commands.add_parser('doctor')
    status = commands.add_parser('status')
    status.add_argument('--projects', type=Path, default=DEFAULT_PROJECTS)
    for name in ('import-analysis', 'import-v2'):
        imp = commands.add_parser(name)
        imp.add_argument('source', type=Path)
        imp.add_argument('--projects', type=Path, default=DEFAULT_PROJECTS)
        imp.add_argument('--name')
        if name == 'import-analysis':
            imp.add_argument('--source-system', default='b01')
    worker = commands.add_parser('_worker', help=argparse.SUPPRESS)
    worker.add_argument('--projects', type=Path, required=True)
    worker.add_argument('--job', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'serve':
            from .studio.server import serve
            serve(args.projects, args.port)
            return 0
        if args.command == '_worker':
            from .studio.jobs import run_worker
            return run_worker(args.projects, args.job)
        if args.command == 'doctor':
            from .studio.pipeline import doctor
            value = doctor()
        elif args.command == 'status':
            from .studio.store import read_json
            from .studio.jobs import recover_interrupted, process_identity
            value = []
            for file in (args.projects / '.jobs').glob('j-*.json'):
                job = recover_interrupted(read_json(file), file)
                value.append({'id': job['id'], 'kind': job['kind'], 'status': job['status'],
                              'stage': job.get('stage'), 'progress': job.get('progress'),
                              'process_alive': bool(job.get('process_identity') and
                                  process_identity(job.get('pid', 0)) == job['process_identity']),
                              'log': job.get('launch', {}).get('log')})
        else:
            from .studio.bridge import import_analysis, import_v2
            func = import_analysis if args.command == 'import-analysis' else import_v2
            opts = {'source_system': args.source_system} if args.command == 'import-analysis' else {}
            value = str(func(args.source, root=args.projects, name=args.name, **opts))
        print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
