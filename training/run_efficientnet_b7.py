"""Queue the EfficientNet-B7 benchmark after DenseNet-121 completes."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from training import continue_resnet34 as audit
from training.config import TrainingConfig

ROOT = Path('/workspace/project')
RUNS = ROOT / 'training/runs/efficientnet_b7'
PAPER = {'accuracy': 0.792, 'macro_auc': 0.898,
         'cohens_kappa_unweighted': 0.829, 'mcc': 0.672}


def aggregate(reports):
    import numpy as np
    seen = set()
    for fold in range(5):
        with np.load(RUNS / f'fold_{fold}/validation_predictions.npz') as data:
            ids = set(data['image_id'].tolist())
            assert not seen.intersection(ids)
            assert len(ids) == len(data['target'])
            assert data['probability'].shape == (len(ids), 5)
            assert np.isfinite(data['probability']).all()
            seen.update(ids)
    assert len(seen) == 3662
    metrics = {}
    for key in audit.KEYS:
        values = [report[key] for report in reports]
        metrics[key] = {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)}
        if key in PAPER:
            metrics[key].update(paper=PAPER[key], difference=statistics.mean(values) - PAPER[key])
    limitations = [
        'Paper Table 2 does not report EfficientNet-B7 macro F1 or validation loss.',
        'Paper kappa 0.829 exceeds accuracy 0.792 and cannot be ordinary unweighted kappa on the same predictions. Weighting or reporting needs clarification.',
        'Unweighted kappa, the existing image-level splits and documented reproduction assumptions are preserved.'
    ]
    result = {'model': 'efficientnet_b7', 'folds': reports, 'metrics': metrics,
              'std_ddof': 1, 'unique_held_out_images': len(seen),
              'paper_source': 'classification.pdf, page 8, Table 2', 'limitations': limitations}
    (RUNS / 'five_fold_report.json').write_text(json.dumps(result, indent=2))
    lines = ['# EfficientNet-B7 five-fold benchmark', '',
             'Mean ± sample standard deviation (ddof=1) across five best checkpoints.', '',
             '| Metric | Mean ± std | Paper |', '|---|---:|---:|']
    for key, value in metrics.items():
        paper = str(value.get('paper', 'Not reported'))
        lines.append(f'| {key} | {value["mean"]:.6f} ± {value["sample_std"]:.6f} | {paper} |')
    lines += ['', *limitations]
    (RUNS / 'five_fold_report.md').write_text('\n'.join(lines) + '\n')
    print('ALL_FOLDS_COMPLETE ' + json.dumps(metrics), flush=True)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    lock = (RUNS / 'continuation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference = audit.read_json(ROOT / 'training/runs/resnet34/fold_0/metadata.json')
    assert TrainingConfig().__dict__ == reference['config']
    hashes = audit.read_json(ROOT / 'training/runs/resnet34/continuation_source_hashes.json')
    previous = ROOT / 'training/runs/densenet121/five_fold_report.json'
    process_path = Path('/proc/700968/cmdline')
    (RUNS / 'queue_status.json').write_text(json.dumps({'status': 'waiting', 'dependency': 'densenet121'}))
    print('WAITING for DenseNet-121 five-fold completion', flush=True)
    while not previous.exists():
        assert process_path.exists() and b'training.run_densenet121' in process_path.read_bytes(), 'DenseNet runner ended without report; refusing to advance'
        time.sleep(10)
    previous_report = audit.read_json(previous)
    assert len(previous_report['folds']) == 5 and previous_report['unique_held_out_images'] == 3662
    audit.RUNS = RUNS
    reports = []
    for fold in range(5):
        for path, digest in hashes.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, f'Source changed: {path}'
        folder = RUNS / f'fold_{fold}'
        if (folder / 'summary.json').exists():
            reports.append(audit.report(fold, reference))
            continue
        assert not folder.exists(), f'Refusing to overwrite or restart fold {fold}'
        env = os.environ.copy()
        env.update(TORCH_HOME=str(ROOT / '.cache/torch'), PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1')
        command = [str(ROOT / '.venv/bin/python'), '-m', 'training.train_benchmark',
                   '--model', 'efficientnet_b7', '--fold', str(fold)]
        print(f'STARTING_FOLD {fold}', flush=True)
        with (RUNS / f'fold_{fold}_console.log').open('x') as console:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT)
            (RUNS / 'queue_status.json').write_text(json.dumps({'status': 'training', 'fold': fold, 'pid': process.pid}))
            if process.wait() != 0:
                (RUNS / 'queue_status.json').write_text(json.dumps({'status': 'failed', 'fold': fold}))
                raise RuntimeError(f'Fold {fold} failed; see console log. No automatic hyperparameter changes.')
        reports.append(audit.report(fold, reference))
    aggregate(reports)
    (RUNS / 'queue_status.json').write_text(json.dumps({'status': 'complete'}))


if __name__ == '__main__':
    main()
