"""Run the five ResNet-50 folds with the established benchmark settings."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

from training import continue_resnet34 as audit
from training.config import TrainingConfig

ROOT = Path('/workspace/project')
RUNS = ROOT / 'training/runs/resnet50'
PAPER = {'accuracy': 0.718, 'macro_auc': 0.843,
         'cohens_kappa_unweighted': 0.794, 'mcc': 0.556}


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
    for key in audit.KEYS:
        values = [report[key] for report in reports]
        metrics[key] = {'mean': statistics.mean(values),
                        'sample_std': statistics.stdev(values)}
        if key in PAPER:
            metrics[key].update(paper=PAPER[key],
                                difference=statistics.mean(values) - PAPER[key])
    limitations = [
        'Paper Table 2 does not report ResNet-50 macro F1 or validation loss.',
        'The reported paper kappa of 0.794 exceeds its accuracy of 0.718; this cannot be ordinary unweighted kappa on the same predictions. Weighting or a reporting error requires author clarification.',
        'This benchmark uses unweighted kappa and preserves the established image-level splits and documented reproduction assumptions.'
    ]
    result = {'model': 'resnet50', 'folds': reports, 'metrics': metrics,
              'std_ddof': 1, 'unique_held_out_images': len(seen),
              'paper_source': 'classification.pdf, page 8, Table 2',
              'limitations': limitations}
    (RUNS / 'five_fold_report.json').write_text(json.dumps(result, indent=2))
    lines = ['# ResNet-50 five-fold benchmark', '',
             'Mean ± sample standard deviation (ddof=1) across five best checkpoints.', '',
             '| Metric | Mean ± std | Paper Table 2 | Difference |',
             '|---|---:|---:|---:|']
    for key, value in metrics.items():
        paper = f'{value["paper"]:.4f}' if 'paper' in value else 'Not reported'
        difference = f'{value["difference"]:+.4f}' if 'difference' in value else '—'
        lines.append(f'| {key} | {value["mean"]:.6f} ± {value["sample_std"]:.6f} | {paper} | {difference} |')
    lines += ['', '| Fold | Best epoch | Stopping epoch | Checkpoint |', '|---|---:|---:|---|']
    for report in reports:
        lines.append(f'| {report["fold"]} | {report["best_epoch"]} | {report["stopping_epoch"]} | {report["checkpoint_path"]} |')
    lines += ['', *limitations]
    (RUNS / 'five_fold_report.md').write_text('\n'.join(lines) + '\n')
    print('ALL_FOLDS_COMPLETE ' + json.dumps(metrics), flush=True)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    lock = (RUNS / 'continuation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference = audit.read_json(ROOT / 'training/runs/resnet34/fold_0/metadata.json')
    assert TrainingConfig().__dict__ == reference['config'], 'Training defaults changed'
    hashes = audit.read_json(ROOT / 'training/runs/resnet34/continuation_source_hashes.json')
    audit.RUNS = RUNS
    reports = []
    for fold in range(5):
        for path, digest in hashes.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, f'Source changed: {path}'
        folder = RUNS / f'fold_{fold}'
        if (folder / 'summary.json').exists():
            reports.append(audit.report(fold, reference))
            continue
        assert not folder.exists(), f'Will not overwrite or restart existing fold {fold}'
        env = os.environ.copy()
        env.update(TORCH_HOME=str(ROOT / '.cache/torch'), PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1')
        command = [str(ROOT / '.venv/bin/python'), '-m', 'training.train_benchmark',
                   '--model', 'resnet50', '--fold', str(fold)]
        print(f'STARTING_FOLD {fold} fresh ImageNet initialization, unchanged configuration', flush=True)
        with (RUNS / f'fold_{fold}_console.log').open('x') as console:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=console,
                                       stderr=subprocess.STDOUT, text=True)
            (RUNS / 'active_process.json').write_text(json.dumps({'fold': fold, 'pid': process.pid}))
            if process.wait() != 0:
                raise RuntimeError(f'Fold {fold} failed; see console log')
        reports.append(audit.report(fold, reference))
    aggregate(reports)


if __name__ == '__main__':
    main()
