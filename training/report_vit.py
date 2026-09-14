"""Verify and report all five completed ViT-B/16 folds without changing them."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from training.metrics import classification_metrics

ROOT = Path(__file__).resolve().parents[1]
PAPER = {'accuracy': 0.782, 'macro_auc': 0.917,
         'cohens_kappa_unweighted': 0.666, 'mcc': 0.667}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT/'training/runs/vit_b_16')
    parser.add_argument('--output', type=Path, default=ROOT/'training/stage3_final')
    args = parser.parse_args()
    with (ROOT/'training/splits/aptos_five_folds.csv').open() as handle:
        manifest = list(csv.DictReader(handle))
    reports, ids_seen, all_targets, all_probabilities = [], set(), [], []
    reference = None
    for fold in range(5):
        folder = args.runs/f'fold_{fold}'
        if fold == 4 and not (folder/'summary.json').exists():
            folder = ROOT/'training/stage3_final/runs/vit_b_16/fold_4'
        summary = json.loads((folder/'summary.json').read_text())
        metadata = json.loads((folder/'metadata.json').read_text())
        if reference is None:
            reference = metadata['config']
        if metadata['config'] != reference or metadata['model'] != 'vit_b_16' or metadata['fold'] != fold:
            raise ValueError(f'Inconsistent configuration: {folder}')
        history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
        best = min(history, key=lambda r:r['validation_loss'])
        if [r['epoch'] for r in history] != list(range(1,len(history)+1)):
            raise ValueError(f'Invalid history: {folder}')
        if len(history)!=reference['epochs'] and len(history)-best['epoch']!=reference['early_stopping_patience']:
            raise ValueError(f'Incomplete training: {folder}')
        checkpoint = torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        if checkpoint['epoch']!=summary['best_epoch'] or checkpoint['epoch']!=best['epoch']:
            raise ValueError(f'Checkpoint epoch mismatch: {folder}')
        if checkpoint['metadata']!=metadata or abs(checkpoint['validation_loss']-summary['validation_loss'])>1e-6:
            raise ValueError(f'Checkpoint metadata/loss mismatch: {folder}')
        del checkpoint
        with np.load(folder/'validation_predictions.npz',allow_pickle=False) as data:
            ids=data['image_id'].tolist(); targets=data['target']; probs=data['probability']
            expected={r['image_id']:int(r['diagnosis']) for r in manifest if int(r['fold'])==fold}
            if len(set(ids))!=len(ids) or set(ids)&ids_seen or dict(zip(ids,targets.tolist()))!=expected:
                raise ValueError(f'Prediction ID/label mismatch: {folder}')
            if probs.shape!=(len(ids),5) or not np.isfinite(probs).all() or np.any(probs<0) or not np.allclose(probs.sum(1),1,atol=1e-5):
                raise ValueError(f'Invalid probabilities: {folder}')
            measured=classification_metrics(targets,probs)
            for key,value in measured.items():
                if abs(value-summary[key])>1e-6 or abs(value-best[key])>1e-6:
                    raise ValueError(f'Metric mismatch: {key}, {folder}')
            ids_seen.update(ids); all_targets.append(targets); all_probabilities.append(probs)
        reports.append(dict(fold=fold,best_epoch=best['epoch'],stopping_epoch=len(history),
                            validation_samples=len(ids),validation_loss=summary['validation_loss'],**measured))
    if len(ids_seen)!=3662:
        raise ValueError('Expected 3662 unique held-out images')
    keys=['validation_loss',*PAPER,'macro_f1']
    metrics={k:dict(mean=float(np.mean([r[k] for r in reports])),
                    sample_std=float(np.std([r[k] for r in reports],ddof=1))) for k in keys}
    result=dict(model='vit_b_16',folds=reports,metrics=metrics,config=reference,
                unique_held_out_images=len(ids_seen),paper_table_2=PAPER,
                pooled_oof_metrics=classification_metrics(np.concatenate(all_targets),np.concatenate(all_probabilities)),
                limitations=['Image-level splits; authors split assignments unavailable.',
                             'Unweighted kappa; paper weighting unspecified.',
                             'Reported validation metrics use validation-selected checkpoints.',
                             'Final fold uses NumPy worker transport; learning settings unchanged.',
                             'ViT benchmark is not the full RobustDRNet ensemble.'])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'vit_five_fold_report.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# ViT-B/16 completed five-fold benchmark','',
           'All 3662 held-out IDs, labels, checkpoints and metrics verified.','',
           '| Metric | Mean ± sample std | Paper Table 2 |','|---|---:|---:|']
    for k,v in metrics.items():
        lines.append(f'| {k} | {v["mean"]:.6f} ± {v["sample_std"]:.6f} | {PAPER.get(k,"Not reported")} |')
    lines+=['','| Fold | Best epoch | Stopping epoch | Accuracy |','|---|---:|---:|---:|']
    lines += [f'| {r["fold"]} | {r["best_epoch"]} | {r["stopping_epoch"]} | {r["accuracy"]:.6f} |' for r in reports]
    lines+=['',*result['limitations']]
    (args.output/'vit_five_fold_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
