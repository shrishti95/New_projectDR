"""Training-fold-only smoke test; no validation-driven hyperparameter selection."""
from dataclasses import replace,asdict
import gc
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from preprocessing import AptosZipDataset
from training.train_portable import make_loader
from training.train_benchmark import seed_everything,inverse_frequency_weights
from training.losses import WeightedFocalCrossEntropy
from .config import Config,HERE,ROOT,preprocessing
from .experiment import (build_model,set_stage,training_mode,autocast,records_for_fold,
                         write_json,save_checkpoint,load_checkpoint,digest)
from .metrics import compute_metrics


def check_metrics():
    y=np.repeat(np.arange(5),2); pred=np.asarray([0,1,1,1,2,3,3,2,4,4])
    prob=np.full((10,5),0.025); prob[np.arange(10),pred]=0.9
    got=compute_metrics(y,prob)
    cm=np.zeros((5,5),dtype=int); np.add.at(cm,(y,pred),1)
    weights=(np.arange(5)[:,None]-np.arange(5)[None,:])**2/16
    expected=np.outer(cm.sum(1),cm.sum(0))/cm.sum()
    qwk=1-(weights*cm).sum()/(weights*expected).sum()
    f1=(2*np.diag(cm)/(cm.sum(0)+cm.sum(1))).mean()
    assert abs(got['accuracy']-0.7)<1e-12
    assert abs(got['macro_f1']-f1)<1e-12
    assert abs(got['qwk']-qwk)<1e-12
    assert got['confusion_matrix']==cm.tolist()
    perfect=compute_metrics(y,np.eye(5)[y])
    assert all(abs(perfect[k]-1)<1e-12 for k in ['accuracy','macro_f1','macro_auc','cohens_kappa_unweighted','qwk','mcc'])
    assert perfect['ece']==0
    assert abs(got['ece']-0.2)<1e-12
    return 'passed (manual confusion/F1/QWK/ECE and perfect predictions)'


def run(batch_size):
    config=replace(Config(),batch_size=batch_size)
    seed_everything(config.seed); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 not supported')
    records,_=records_for_fold(0)
    # Fixed, training-only examples: no validation labels or images participate.
    chosen=[]
    for grade in range(5): chosen.extend([r for r in records if r.diagnosis==grade][:6])
    chosen.extend([r for r in records if r not in chosen][:2])
    ds=AptosZipDataset(ROOT/'aptos2019-blindness-detection.zip',chosen,training=False,
                       config=preprocessing(),warn_about_assumptions=False)
    loader=make_loader(ds,batch_size=batch_size,shuffle=False,workers=0,seed=config.seed)
    batches=list(loader); ds.close()
    model=build_model().cuda()
    criterion=WeightedFocalCrossEntropy(inverse_frequency_weights(records,5).cuda(),
                                        gamma=config.gamma,label_smoothing=config.smoothing)
    optimizer,scheduler=set_stage(model,'head',config)
    counts={'total':sum(p.numel() for p in model.parameters()),'head':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    stem_before=model.stem[0].weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    def fixed_loss():
        model.eval()
        with torch.no_grad():
            losses=[]
            for b in batches:
                with autocast(): logits=model(b['image'].cuda())
                losses.append(float(criterion(logits.float(),b['label'].cuda()))*len(b['label']))
        return sum(losses)/32
    def step(stage):
        training_mode(model,stage); optimizer.zero_grad(set_to_none=True)
        for b in batches:
            with autocast(): logits=model(b['image'].cuda())
            assert logits.shape==(len(b['label']),5)
            loss=criterion(logits.float(),b['label'].cuda())
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite BF16 loss')
            (loss*len(b['label'])/32).backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g)>0 for g in grads)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
        optimizer.step()
    initial=fixed_loss()
    for _ in range(10): step('head')
    after_head=fixed_loss()
    assert after_head<initial,(initial,after_head)
    assert torch.equal(stem_before,model.stem[0].weight)
    optimizer,scheduler=set_stage(model,'partial',config)
    counts['partial']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    before_partial=fixed_loss()
    for _ in range(3): step('partial')
    after_partial=fixed_loss()
    assert after_partial<before_partial,(before_partial,after_partial)
    assert torch.equal(stem_before,model.stem[0].weight)
    # Verify actual penultimate path and classifier give the same logits.
    model.eval(); b=batches[0]['image'].cuda()
    with torch.no_grad(),autocast():
        reference=model(b).float().cpu()
        features=model.forward_head(model.forward_features(b),pre_logits=True)
        torch.testing.assert_close(model.get_classifier()(features).float().cpu(),reference)
    assert features.shape==(len(b),1024)
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        ckpt=Path(temp)/'restore.pt'
        save_checkpoint(ckpt,model,optimizer,scheduler,3,'partial',{'validation_loss':after_partial},{'smoke':True})
        with torch.no_grad(): model.get_classifier().weight.add_(0.5)
        load_checkpoint(ckpt,model)
        with torch.no_grad(),autocast(): restored=model(b).float().cpu()
        torch.testing.assert_close(restored,reference,rtol=0,atol=0)
    return {'passed':True,'training_fold':0,'training_only_samples':32,'config':asdict(config),
            'parameter_counts':counts,'feature_dimension':1024,'initial_loss':initial,
            'after_head_loss':after_head,'before_partial_loss':before_partial,
            'after_partial_loss':after_partial,'bf16_stable':True,'checkpoint_restore_exact':True,
            'frozen_stem_unchanged':True,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
            'metrics_check':check_metrics(),'source_hashes':source_hashes()}


def source_hashes():
    paths=list(HERE.glob('*.py'))+list((ROOT/'preprocessing').glob('*.py'))
    paths += [ROOT/'training/train_portable.py',ROOT/'training/losses.py',ROOT/'training/train_benchmark.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


if __name__=='__main__':
    started=time.monotonic(); failures=[]
    for batch in [32,16,8]:
        try:
            result=run(batch); break
        except torch.cuda.OutOfMemoryError:
            failures.append({'batch_size':batch,'failure':'CUDA OOM only; no metric-based adjustment'})
            gc.collect(); torch.cuda.empty_cache()
    else: raise RuntimeError('Smoke test failed: no usable micro-batch size')
    result.update(elapsed_seconds=time.monotonic()-started,memory_fallbacks=failures)
    write_json(HERE/'smoke_result.json',result)
    print(json.dumps(result,indent=2),flush=True)
