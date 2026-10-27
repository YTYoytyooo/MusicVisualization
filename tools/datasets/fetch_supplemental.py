"""Download the official PMEmo/MERP releases with a local provenance ledger."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2] / 'data/datasets'
AGENT = 'MusicVisualization-dataset-preparation/1.0'


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': AGENT, **(headers or {})}), timeout=60)


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def download(root, item):
    path = (root / item['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Download path escapes dataset root')
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = path.with_name(path.name + '.download.json')
    if path.exists():
        if item.get('size') is not None and path.stat().st_size != item['size']:
            raise ValueError(f'Existing file has wrong size: {path}')
        sha = digest(path)
        expected = item.get('sha256')
        if not expected and ledger.is_file():
            previous = json.loads(ledger.read_text())
            if previous['url'] != item['url']:
                raise ValueError('Existing file came from a different release')
            expected = previous['sha256']
        if not expected or sha != expected:
            raise ValueError(f'Existing file cannot be verified: {path}')
        if item.get('md5') and digest(path, 'md5') != item['md5']:
            raise ValueError(f'MD5 mismatch: {path}')
        return dict(path=item['path'], bytes=path.stat().st_size, sha256=sha, cached=True)
    partial = path.with_name(path.name + '.part')
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with request(item['url'], headers) as response:
                if 'text/html' in response.headers.get('Content-Type', ''):
                    raise ValueError('Server returned an HTML page instead of the dataset')
                append = offset > 0 and response.status == 206
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Server resumed at an unexpected offset')
                with partial.open('ab' if append else 'wb') as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        stream.write(block)
            if item.get('size') is not None and partial.stat().st_size != item['size']:
                raise ValueError(f'Incomplete download: {item["path"]}')
            sha = digest(partial)
            if item.get('sha256') and sha != item['sha256']:
                raise ValueError(f'SHA256 mismatch: {item["path"]}')
            if item.get('md5') and digest(partial, 'md5') != item['md5']:
                raise ValueError(f'MD5 mismatch: {item["path"]}')
            partial.replace(path)
            receipt = dict(url=item['url'], bytes=path.stat().st_size, sha256=sha,
                           publisher_hash_verified=bool(item.get('sha256') or item.get('md5')))
            write_json(ledger, receipt)
            print(json.dumps(dict(downloaded=item['path'], bytes=receipt['bytes'])), flush=True)
            return dict(path=item['path'], **receipt)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def main(source, metadata_only=False):
    root = ROOT / source / 'raw'
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'download-manifest.json'
    if manifest.exists():
        plan = json.loads(manifest.read_text())
    elif source == 'merp':
        api = 'https://huggingface.co/api/datasets/amaai-lab/MERP'
        with request(api) as r:
            revision = json.load(r)['sha']
        with request(f'{api}/tree/{revision}?recursive=true&expand=false') as r:
            listing = json.load(r)
        files = [dict(path=x['path'], size=x['size'], sha256=x.get('lfs', {}).get('oid'),
                      url=f'https://huggingface.co/datasets/amaai-lab/MERP/resolve/{revision}/{x["path"]}')
                 for x in listing if x['type'] == 'file' and x['path'] != '.gitattributes']
        plan = dict(dataset=source, source=api, revision=revision, files=files)
    elif source == 'pmemo':
        repo = 'https://api.github.com/repos/HuiZhangDB/PMEmo/commits/master'
        with request(repo) as r:
            revision = json.load(r)['sha']
        files = [dict(path='PMEmo2019.zip', size=679668214,
                      url='https://drive.usercontent.google.com/download?id=1UzC3NCDj30j9Ba7i5lkMzWO5gFqSr0OJ&export=download&confirm=t')]
        files.extend(dict(path=name, url=f'https://raw.githubusercontent.com/HuiZhangDB/PMEmo/{revision}/{name}')
                     for name in ('README.md', 'LICENSE'))
        plan = dict(dataset=source, source='https://github.com/HuiZhangDB/PMEmo', revision=revision,
                    archive_source='https://drive.google.com/drive/folders/1qDk6hZDGVlVXgckjLq9LvXLZ9EgK9gw0', files=files)
    else:
        api = 'https://api.figshare.com/v2/articles/25827034'
        with request(api) as r:
            record = json.load(r)
        write_json(root / 'figshare-record.json', record)
        names = {'MusicRawData.zip', 'README.txt', 'Label_pseudonymous.json', 'label_process.py'}
        files = [dict(path=x['name'], url=x['download_url'], size=x['size'], md5=x['computed_md5'])
                 for x in record['files'] if x['name'] in names or x['name'].endswith('.csv')]
        plan = dict(dataset=source, source=api, revision=record['version'], files=files)
    write_json(manifest, plan)
    selected = [f for f in plan['files'] if not metadata_only or not f['path'].endswith(('.zip', '.wav'))]
    write_json(root / 'download-status.json', dict(status='downloading', files=len(selected), metadata_only=metadata_only))
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda item: download(root, item), selected))
        write_json(root / 'download-status.json', dict(status='completed', metadata_only=metadata_only, files=results))
        print(json.dumps(dict(status='completed', dataset=source, files=len(results))), flush=True)
    except Exception as exc:
        write_json(root / 'download-status.json', dict(status='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=('pmemo', 'merp', 'memo2496'))
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    main(args.source, args.metadata_only)
