"""Create a project-only venv around an existing Isaac Lab Python installation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream/Isaaclab_Parkour'
COMMIT = 'd6766d8883cffe427f2d4c087717d4eca1f23d8c'


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def download(url, path, sha256):
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.download')
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
            while data := response.read(1024 * 1024):
                stream.write(data)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != sha256:
            raise ValueError(f'Download hash mismatch: {path}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isaac-python', type=Path, required=True, help='Existing Isaac Lab venv Python, not the system Python.')
    parser.add_argument('--venv', type=Path, default=ROOT / '.venv')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--download-go2', action='store_true')
    parser.add_argument('--allow-runtime-mismatch', action='store_true')
    args = parser.parse_args()
    base = args.isaac_python.expanduser().absolute()
    probe = '''import json,sys,sysconfig,importlib.metadata as m
names=['isaacsim','isaaclab','torch','torchvision']
print(json.dumps({'python':list(sys.version_info[:3]),'site_packages':sysconfig.get_path('purelib'),'packages':{n:m.version(n) for n in names}}))'''
    runtime = json.loads(subprocess.check_output([str(base), '-c', probe], text=True))
    print(json.dumps(runtime, indent=2), flush=True)
    packages = runtime['packages']
    mismatch = (runtime['python'][:2] != [3, 11] or not packages['isaacsim'].startswith('5.1.') or
                not packages['torch'].startswith('2.7.') or packages['isaaclab'] != '0.54.2')
    if mismatch and not args.allow_runtime_mismatch:
        raise SystemExit('Validated runtime: Python 3.11 / Sim 5.1 / Lab v2.3.2 (package 0.54.2) / Torch 2.7. '
                         'Use an isolated matching runtime; --allow-runtime-mismatch explicitly permits unvalidated combinations.')
    if args.check_only:
        return
    if UPSTREAM.exists():
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=UPSTREAM, text=True).strip()
        if actual != COMMIT:
            raise SystemExit('Existing upstream has another revision; preserving it. Use a fresh clone of RBQ_WMP.')
    else:
        UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
        run(['git', 'clone', '--filter=blob:none', '--no-checkout', 'https://github.com/CAI23sbP/Isaaclab_Parkour.git', UPSTREAM])
        run(['git', 'checkout', '--detach', COMMIT], cwd=UPSTREAM)
    patch = ROOT / 'patches/current_upstream.patch'
    applied = subprocess.run(['git', 'apply', '--reverse', '--check', str(patch)], cwd=UPSTREAM, capture_output=True).returncode == 0
    if not applied:
        clean = subprocess.run(['git', 'diff', '--quiet'], cwd=UPSTREAM).returncode == 0
        if not clean:
            raise SystemExit('Existing upstream modifications differ from this patch; preserving them.')
        run(['git', 'apply', '--check', patch], cwd=UPSTREAM)
        run(['git', 'apply', patch], cwd=UPSTREAM)
    for entry in json.loads((ROOT / 'assets/rbq10/manifest.json').read_text())['files']:
        download(entry['source'], ROOT / 'assets/rbq10' / entry['path'], entry['sha256'])
    if args.download_go2:
        for entry in json.loads((ROOT / 'checkpoints/manifest.json').read_text()):
            download(entry['source_url'], ROOT / 'checkpoints' / entry['file'], entry['sha256'])
    venv = args.venv.expanduser().absolute()
    if venv.exists():
        raise SystemExit(f'Preserving existing venv {venv}; choose a new --venv path or reuse the already configured environment.')
    run([base, '-m', 'venv', venv])
    python = venv / 'bin/python'
    site_packages = Path(subprocess.check_output([str(python), '-c', 'import sysconfig;print(sysconfig.get_path("purelib"))'], text=True).strip())
    (site_packages / 'isaac_runtime.pth').write_text(f'import site; site.addsitedir({runtime["site_packages"]!r})\n')
    run([python, '-m', 'pip', 'install', '-r', ROOT / 'requirements.txt'])
    run([python, '-m', 'pip', 'install', '--no-deps', '-e', UPSTREAM, '-e', UPSTREAM / 'parkour_tasks'])
    print(f'Project venv ready: {venv}. Existing Isaac runtime packages were not modified.', flush=True)
    if venv != ROOT / '.venv':
        print(f'Use RBQ10_PYTHON={python} with run.sh for this nondefault venv.')


if __name__ == '__main__':
    main()
