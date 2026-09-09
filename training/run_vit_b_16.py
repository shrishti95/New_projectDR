"""Run the ViT-B/16 benchmark with the unchanged five-fold protocol."""
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
RUNS = ROOT / 'training/runs/vit_b_16'
PAPER = {'accuracy': 0.782, 'macro_auc': 0.917,
         'cohens_kappa_unweighted': 0.666, 'mcc': 0.667}


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
    result = {'model': 'vit_b_16', 'folds': reports, 'metrics': metrics,
              'std_ddof': 1, 'unique_held_out_images': len(seen),
              'paper_source': 'classification.pdf, page 8, Table 2',
              'limitations': ['Existing image-level splits and reproduction assumptions preserved.',
                              'Kappa is unweighted; paper weighting is unspecified.',
                              'Paper Table 2 gives ViT AUC 0.917; Table 3 gives 0.920.']}
    (RUNS / 'five_fold_report.json').write_text(json.dumps(result, indent=2))
    lines = ['# ViT-B/16 five-fold benchmark', '',
             '| Metric | Mean ± sample std | Paper Table 2 |', '|---|---:|---:|']
    for key, value in metrics.items():
        lines.append(f'| {key} | {value["mean"]:.6f} ± {value["sample_std"]:.6f} | {value.get("paper", "Not reported")} |')
    lines += ['', *result['limitations']]
    (RUNS / 'five_fold_report.md').write_text('\n'.join(lines) + '\n')


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    lock = (RUNS / 'continuation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference = audit.read_json(ROOT / 'training/runs/resnet34/fold_0/metadata.json')
    assert TrainingConfig().__dict__ == reference['config']
    hashes = audit.read_json(ROOT / 'training/runs/resnet34/continuation_source_hashes.json')
    previous = audit.read_json(ROOT / 'training/runs/convnext_tiny/five_fold_report.json')
    assert len(previous['folds']) == 5 and previous['unique_held_out_images'] == 3662
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
                   '--model', 'vit_b_16', '--fold', str(fold)]
        with (RUNS / f'fold_{fold}_console.log').open('x') as console:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT)
            (RUNS / 'queue_status.json').write_text(json.dumps({'status': 'training', 'fold': fold, 'pid': process.pid}))
            print(f'STARTING_FOLD {fold}: fresh ImageNet initialization, unchanged settings', flush=True)
            if process.wait() != 0:
                (RUNS / 'queue_status.json').write_text(json.dumps({'status': 'failed', 'fold': fold}))
                raise RuntimeError(f'Fold {fold} failed; see console log')
        reports.append(audit.report(fold, reference))
    aggregate(reports)
    (RUNS / 'queue_status.json').write_text(json.dumps({'status': 'complete'}))
    print('ALL_FOLDS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
