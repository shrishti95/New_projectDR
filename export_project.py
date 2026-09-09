"""Create a portable project ZIP while keeping the live experiment running."""
from datetime import datetime, timezone
from pathlib import Path
import json
import zipfile

ROOT = Path(__file__).resolve().parent
EXCLUDED = {'.venv', '.cache', 'downloads', '__pycache__', '.ipynb_checkpoints'}


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = ROOT / 'downloads' / f'RobustDRNet-project-{stamp}.zip'
    target.parent.mkdir(exist_ok=True)
    files = []
    skipped = []
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED for part in relative.parts) or not path.is_file():
            continue
        if path.name.endswith('.lock'):
            continue
        if path.name == 'best.pt' and not (path.parent / 'completion_report.json').exists():
            skipped.append(str(relative))
            continue
        files.append(path)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            archive.write(path, 'project/' + str(path.relative_to(ROOT)))
        manifest = {
            'created_utc': stamp,
            'included': 'Paper, source code, splits, logs and results available during export; verified completed-fold checkpoints.',
            'excluded_directories': sorted(EXCLUDED),
            'excluded_checkpoints': skipped,
            'dataset': '/workspace/aptos2019-blindness-detection.zip is external and is not included.',
            'training': 'Continues on the server. Active logs may contain partial progress; this download does not update automatically.',
            'environment': 'Recreate dependencies locally. Server uses Python 3.12, torch 2.9.0+cu128, torchvision 0.24.0+cu128, numpy 2.2.6, albumentations 2.0.8, opencv-python-headless 4.12.0.88.',
            'portability': 'Scripts currently contain /workspace paths; adjust CLI arguments/defaults to local paths before training.'
        }
        archive.writestr('project/DOWNLOAD_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f'ZIP integrity check failed: {bad}')
        print(json.dumps({'archive': str(target), 'bytes': target.stat().st_size,
                          'files': len(archive.namelist()), 'integrity': 'passed'}), flush=True)


if __name__ == '__main__':
    main()
