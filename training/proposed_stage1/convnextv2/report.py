"""Primary validation-loss results and explicitly separate supplementary analysis."""
import json
from pathlib import Path
import numpy as np
from .config import HERE,ROOT
from .experiment import write_json,records_for_fold
from .metrics import compute_metrics

KEYS=('accuracy','macro_f1','macro_auc','cohens_kappa_unweighted','qwk','mcc')
PAPER={'accuracy':0.821,'macro_auc':0.927,'cohens_kappa_unweighted':0.727,'mcc':0.728}


def aggregate(folds):
    return {key:{'mean':float(np.mean([f[key] for f in folds])),
                 'sample_std':float(np.std([f[key] for f in folds],ddof=1)) if len(folds)>1 else None}
            for key in KEYS}


def verify_export(folder,fold,split):
    records=records_for_fold(fold)[0 if split=='train' else 1]
    with np.load(folder/f'{split}_features.npz',allow_pickle=False) as d:
        ids=d['sample_id'].tolist()
        assert len(ids)==len(set(ids))==len(records)
        assert dict(zip(ids,d['label'].tolist()))=={r.image_id:r.diagnosis for r in records}
        assert np.all(d['fold']==fold)
        assert d['feature'].shape==(len(records),1024) and np.isfinite(d['feature']).all()
        assert np.array_equal(d['predicted_class'],d['probability'].argmax(1))
        return compute_metrics(d['label'],d['probability'])


def write_report():
    folds=[]
    for fold in range(5):
        folder=HERE/'runs'/f'fold_{fold}'
        if not (folder/'summary.json').exists(): continue
        summary=json.loads((folder/'summary.json').read_text())
        verify_export(folder,fold,'train'); measured=verify_export(folder,fold,'validation')
        for key in KEYS:
            if abs(summary['metrics'][key]-measured[key])>1e-6: raise ValueError(f'Report/export mismatch: fold {fold}, {key}')
        folds.append(summary)
    tiny=[]
    for fold in range(5):
        with np.load(ROOT/'training/runs/convnext_tiny'/f'fold_{fold}'/'validation_predictions.npz',allow_pickle=False) as d:
            _,records=records_for_fold(fold)
            assert dict(zip(d['image_id'].tolist(),d['target'].tolist()))=={r.image_id:r.diagnosis for r in records}
            tiny.append(compute_metrics(d['target'],d['probability']))
    result={'model':'convnextv2_base.fcmae_ft_in1k','completed_folds':len(folds),'primary_selection':'validation_loss',
            'primary':aggregate([f['metrics'] for f in folds]) if len(folds)==5 else None,
            'folds':folds,'reproduced_convnext_tiny':aggregate(tiny),'paper_convnext_tiny':PAPER,
            'std_ddof':1,'notes':['Comparisons include differences in architecture, fine-tuning and augmentation; not a controlled architecture-only ablation.',
                                'Paper Table 2 does not report macro F1 or QWK; paper kappa weighting is unspecified.',
                                'Validation-selected checkpoints; no untouched external-test estimate.',
                                'Train exports are in-sample features, not OOF stacking features.']}
    write_json(HERE/'convnextv2_five_fold_report.json',result)
    lines=['# ConvNeXt V2 Base — APTOS 2019 five-fold report','',f'Status: {len(folds)}/5 folds completed.',
           'Primary checkpoint: lowest validation loss. Macro-F1 and QWK checkpoints are supplementary only.','']
    source=json.loads((HERE/'pretrained_source.json').read_text())
    smoke=json.loads((HERE/'smoke_result.json').read_text())
    counts=smoke['parameter_counts']
    lines += [f'Pretrained checkpoint: [{source["repo_id"]}]({source["url"]})',
              f'Revision: `{source["revision"]}`; SHA-256: `{source["sha256"]}`.',
              f'Feature dimension: 1024; total parameters: {counts["total"]:,}; '
              f'head-stage trainable: {counts["head"]:,}; partial-stage trainable: {counts["partial"]:,}.', '']
    if len(folds)==5:
        lines+=['| Metric | ConvNeXt V2 Base mean ± sample std | Reproduced ConvNeXt-Tiny | Paper ConvNeXt-Tiny |',
                '|---|---:|---:|---:|']
        for k in KEYS:
            v=result['primary'][k]; t=result['reproduced_convnext_tiny'][k]
            lines.append(f'| {k} | {v["mean"]:.6f} ± {v["sample_std"]:.6f} | {t["mean"]:.6f} ± {t["sample_std"]:.6f} | {PAPER.get(k,"Not reported")} |')
    else: lines+=['Five-fold means will be calculated only after all five folds finish.']
    for fold in folds:
        m=fold['metrics']; n=fold['fold']
        lines+=['',f'## Fold {n}',f'Best epoch: {fold["best_epoch"]}; stopping epoch: {fold["stopping_epoch"]}; '
                f'final stage: {fold["final_training_stage"]}; training seconds: {fold["training_seconds"]:.1f}.','',
                '| Metric | Official checkpoint |','|---|---:|']
        lines += [f'| {k} | {m[k]:.6f} |' for k in [*KEYS,'ece','validation_loss']]
        lines+=['','| Grade | Precision | Recall | F1 | Support |','|---|---:|---:|---:|---:|']
        lines += [f'| {c["grade"]} | {c["precision"]:.6f} | {c["recall"]:.6f} | {c["f1"]:.6f} | {c["support"]} |' for c in m['per_class']]
        lines+=['','Confusion matrix (rows=true grade; columns=predicted grade):','',
                '| True / predicted | 0 | 1 | 2 | 3 | 4 |','|---|---:|---:|---:|---:|---:|']
        lines += [f'| {i} | '+' | '.join(map(str,row))+' |' for i,row in enumerate(m['confusion_matrix'])]
        lines+=['',f'Grade 1 recall: {m["grade_1_recall"]:.6f}; Grade 3 recall: {m["grade_3_recall"]:.6f}. '
                f'Grade 2→3: {m["grade_2_as_3"]}; Grade 3→2: {m["grade_3_as_2"]}.','',
                'Supplementary selection (not substituted for primary results):','',
                '| Selection | Epoch | Accuracy | Macro F1 | QWK |','|---|---:|---:|---:|---:|']
        for name,s in fold['supplementary'].items():
            lines.append(f'| {name} | {s["epoch"]} | {s["metrics"]["accuracy"]:.6f} | {s["metrics"]["macro_f1"]:.6f} | {s["metrics"]["qwk"]:.6f} |')
    lines+=['',*result['notes'],'','Exact checkpoint provenance: `pretrained_source.json`. Fixed experiment: `protocol.json`.']
    (HERE/'convnextv2_five_fold_report.md').write_text('\n'.join(lines)+'\n')
    if len(folds)==5: print('MODEL_TRAINING_COMPLETE '+json.dumps(result['primary']),flush=True)
    return result


if __name__=='__main__': write_report()
