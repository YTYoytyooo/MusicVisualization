"""CLI registration/preparation and single-song inference entry points."""
import argparse
import json
from . import analyze, build_inventory, prepare_manifest, verify_result


def main():
    parser = argparse.ArgumentParser(description='V3 fixed-weight research preview; no training')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    sub.add_parser('inventory')
    run = sub.add_parser('analyze')
    run.add_argument('--audio', required=True)
    run.add_argument('--project', required=True)
    run.add_argument('--device', choices=['cpu', 'cuda', 'cuda:0'], default='cuda')
    run.add_argument('--include-v2', action='store_true')
    check = sub.add_parser('verify')
    check.add_argument('--project', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare_manifest())
    elif args.command == 'inventory':
        result = build_inventory()
        print(json.dumps(dict(songs=len(result['songs']), library='data/v3.0/library.json')))
    elif args.command == 'verify':
        result = verify_result(args.project)
        print(json.dumps(dict(status='verified', project_id=result['project_id'])))
    else:
        path = analyze(args.audio, args.project, args.device, include_v2=args.include_v2,
                       progress=lambda state: print(json.dumps(state, ensure_ascii=False), flush=True))
        print(path)


if __name__ == '__main__':
    main()
