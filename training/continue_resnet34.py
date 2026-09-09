"""Continue the existing fold 0, then run folds 1–4 without changing training."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path('/workspace/project')
RUNS = ROOT / 'training/runs/resnet34'
KEYS = ('validation_loss', 'accuracy', 'macro_f1', 'macro_auc',
        'cohens_kappa_unweighted', 'mcc')
PAPER = {'accuracy': 0.790, 'macro_auc': 0.908,
         'cohens_kappa_unweighted': 0.678, 'mcc': 0.679}


def read_json(path):
    return json.loads(path.read_text())


def history(fold):
    path = RUNS / f'fold_{fold}/history.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def report(fold, reference):
    import torch
    folder = RUNS / f'fold_{fold}'
    result = read_json(folder / 'summary.json')
    metadata = read_json(folder / 'metadata.json')
    assert metadata['config'] == reference['config'], 'Training config changed'
    for key in ('pretrained', 'amp_bfloat16', 'max_train_batches', 'max_val_batches'):
        assert metadata[key] == reference[key], f'{key} changed'
    rows = history(fold)
    assert [row['epoch'] for row in rows] == list(range(1, len(rows) + 1))
    best = min(rows, key=lambda row: row['validation_loss'])
    assert result['best_epoch'] == best['epoch']
    assert len(rows) <= 30
    assert len(rows) == 30 or len(rows) - best['epoch'] == 5
    checkpoint_path = folder / 'best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == best['epoch']
    assert checkpoint['validation_loss'] == best['validation_loss']
    for key in KEYS:
        expected = best[key]
        assert abs(result[key] - expected) < 1e-6, (key, result[key], expected)
    result.update(fold=fold, stopping_epoch=rows[-1]['epoch'],
                  checkpoint_path=str(checkpoint_path))
    (folder / 'completion_report.json').write_text(json.dumps(result, indent=2))
    print('FOLD_COMPLETE ' + json.dumps(result), flush=True)
    return result


def aggregate(reports):
    import numpy as np
    seen = set()
    for fold in range(5):
        with np.load(RUNS / f'fold_{fold}/validation_predictions.npz') as data:
            ids = set(data['image_id'].tolist())
            assert not (seen & ids), 'Repeated held-out IDs'
            assert len(ids) == len(data['target'])
            assert data['probability'].shape == (len(ids), 5)
            assert np.isfinite(data['probability']).all()
            seen.update(ids)
    assert len(seen) == 3662
    metrics = {}
    for key in KEYS:
        values = [r[key] for r in reports]
        metrics[key] = {'mean': statistics.mean(values),
                        'sample_std': statistics.stdev(values)}
        if key in PAPER:
            metrics[key].update(paper=PAPER[key],
                                difference=statistics.mean(values) - PAPER[key])
    result = {'folds': reports, 'metrics': metrics, 'std_ddof': 1,
              'unique_held_out_images': len(seen),
              'paper_source': 'classification.pdf, page 8, Table 2',
              'limitations': ['Paper does not report ResNet-34 macro F1 or validation loss in Table 2.',
                             'Paper does not identify kappa weighting; this run uses unweighted kappa.',
                             'Existing image-level split and reproduction assumptions are preserved.']}
    (RUNS / 'five_fold_report.json').write_text(json.dumps(result, indent=2))
    lines = ['# ResNet-34 five-fold benchmark', '',
             'Mean ± sample standard deviation (ddof=1) across five best checkpoints.', '',
             '| Metric | Mean ± std | Paper Table 2 | Difference |',
             '|---|---:|---:|---:|']
    for key, value in metrics.items():
        paper = f'{value["paper"]:.4f}' if 'paper' in value else 'Not reported'
        diff = f'{value["difference"]:+.4f}' if 'difference' in value else '—'
        lines.append(f'| {key} | {value["mean"]:.6f} ± {value["sample_std"]:.6f} | {paper} | {diff} |')
    lines += ['', '| Fold | Best epoch | Stopping epoch | Checkpoint |', '|---|---:|---:|---|']
    for r in reports:
        lines.append(f'| {r["fold"]} | {r["best_epoch"]} | {r["stopping_epoch"]} | {r["checkpoint_path"]} |')
    lines += ['', *result['limitations']]
    (RUNS / 'five_fold_report.md').write_text('\n'.join(lines) + '\n')
    print('ALL_FOLDS_COMPLETE ' + json.dumps(metrics), flush=True)


def main():
    import fcntl
    lock = (RUNS / 'continuation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference = read_json(RUNS / 'fold_0/metadata.json')
    assert reference['config']['epochs'] == 30
    assert reference['config']['early_stopping_patience'] == 5
    assert reference['pretrained'] and not reference['amp_bfloat16']
    sources = list((ROOT / 'preprocessing').glob('*.py')) + [ROOT / 'training' / name for name in
               ('config.py', 'train_benchmark.py', 'models.py', 'losses.py', 'metrics.py',
                'splits/aptos_five_folds.csv')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (RUNS / 'continuation_source_hashes.json').write_text(json.dumps(hashes, indent=2))
    original_pid = 580384
    last_epoch = -1
    while not (RUNS / 'fold_0/summary.json').exists():
        process = Path(f'/proc/{original_pid}/cmdline')
        assert process.exists() and b'training.train_benchmark' in process.read_bytes(), 'Fold 0 ended without summary; will not restart'
        rows = history(0)
        if rows and rows[-1]['epoch'] != last_epoch:
            print('FOLD_0_PROGRESS ' + json.dumps(rows[-1]), flush=True)
            last_epoch = rows[-1]['epoch']
        time.sleep(10)
    reports = [report(0, reference)]
    for fold in range(1, 5):
        for path, digest in hashes.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, f'Source changed: {path}'
        folder = RUNS / f'fold_{fold}'
        if (folder / 'summary.json').exists():
            reports.append(report(fold, reference))
            continue
        assert not folder.exists(), f'Will not overwrite or restart existing fold {fold}'
        env = os.environ.copy()
        env.update(TORCH_HOME=str(ROOT / '.cache/torch'), PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1')
        command = [str(ROOT / '.venv/bin/python'), '-m', 'training.train_benchmark',
                   '--model', 'resnet34', '--fold', str(fold)]
        print(f'STARTING_FOLD {fold} fresh ImageNet initialization, unchanged configuration', flush=True)
        with (RUNS / f'fold_{fold}_console.log').open('x') as console:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                console.write(line)
                console.flush()
                print(f'fold_{fold}: {line}', end='', flush=True)
            assert process.wait() == 0, f'Fold {fold} failed; see console log'
        reports.append(report(fold, reference))
    aggregate(reports)


if __name__ == '__main__':
    main()
