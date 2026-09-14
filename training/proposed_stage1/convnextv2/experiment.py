from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import timm
from safetensors.torch import load_file

from preprocessing import AptosZipDataset
from training.train_benchmark import (seed_everything, load_fold_records,
                                      inverse_frequency_weights)
from training.train_portable import make_loader
from training.losses import WeightedFocalCrossEntropy
from .config import Config, ROOT, HERE, MODEL, MANIFEST, MANIFEST_SHA256, preprocessing, protocol
from .metrics import compute_metrics


def write_json(path, data):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n'); tmp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def records_for_fold(fold):
    if digest(MANIFEST)!=MANIFEST_SHA256:
        raise ValueError('Original manifest was modified; refusing to train')
    train,val=load_fold_records(MANIFEST,fold)
    train_ids={r.image_id for r in train}; val_ids={r.image_id for r in val}
    if train_ids & val_ids or len(train_ids)+len(val_ids)!=3662:
        raise ValueError('Invalid or overlapping fold IDs')
    return train,val


def build_model():
    source=json.loads((HERE/'pretrained_source.json').read_text())
    if digest(source['path'])!=source['sha256']:
        raise ValueError('Pretrained checkpoint hash mismatch')
    model=timm.create_model(MODEL,pretrained=False,num_classes=1000)
    model.load_state_dict(load_file(source['path']),strict=True)
    model.reset_classifier(5)
    return model


def set_stage(model, stage, config):
    model.requires_grad_(False)
    model.get_classifier().requires_grad_(True)
    if stage=='partial':
        for module in [*model.stages[2:],model.norm_pre,model.head.norm]:
            module.requires_grad_(True)
    model.eval()
    model.get_classifier().train()
    if stage=='partial':
        for module in [*model.stages[2:],model.norm_pre,model.head.norm]:
            module.train()
    head=list(model.get_classifier().parameters()); head_ids={id(p) for p in head}
    groups=[{'params':head,'lr':config.head_warmup_lr if stage=='head' else config.head_lr,'name':'classifier'}]
    if stage=='partial':
        groups.insert(0,{'params':[p for p in model.parameters() if p.requires_grad and id(p) not in head_ids],
                         'lr':config.backbone_lr,'name':'backbone_final_two_stages'})
    optimizer=torch.optim.AdamW(groups,weight_decay=config.weight_decay)
    scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                  T_max=config.max_epochs-config.head_epochs,eta_min=config.cosine_min_lr)
               if stage=='partial' else None)
    return optimizer,scheduler


def training_mode(model,stage):
    model.eval()
    model.get_classifier().train()
    if stage=='partial':
        for module in [*model.stages[2:],model.norm_pre,model.head.norm]: module.train()


def autocast():
    return torch.autocast('cuda',dtype=torch.bfloat16)


def train_epoch(model,loader,criterion,optimizer,stage,config):
    training_mode(model,stage)
    optimizer.zero_grad(set_to_none=True)
    total,count,window=0.0,0,0
    for i,batch in enumerate(loader):
        images=batch['image'].cuda(non_blocking=True); labels=batch['label'].cuda(non_blocking=True)
        with autocast(): logits=model(images)
        loss=criterion(logits.float(),labels)
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
        n=len(labels); (loss*(n/config.effective_batch_size)).backward()
        total+=float(loss.detach())*n; count+=n; window+=n
        if window>=config.effective_batch_size or i+1==len(loader):
            # Preserve the mean gradient for a final incomplete accumulation window.
            for p in model.parameters():
                if p.grad is not None: p.grad.mul_(config.effective_batch_size/window)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                         config.gradient_clip,error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); window=0
    return total/count


@torch.inference_mode()
def evaluate(model,loader,criterion,collect_features=False):
    model.eval(); total,count=0.0,0; ids=[]; targets=[]; probabilities=[]; features=[]
    for batch in loader:
        images=batch['image'].cuda(non_blocking=True); labels=batch['label'].cuda(non_blocking=True)
        with autocast():
            if collect_features:
                embedding=model.forward_head(model.forward_features(images),pre_logits=True)
                logits=model.get_classifier()(embedding)
            else: logits=model(images)
        loss=criterion(logits.float(),labels)
        probability=torch.softmax(logits.float(),dim=1)
        if not torch.isfinite(loss) or not torch.isfinite(probability).all():
            raise FloatingPointError('Non-finite evaluation output')
        total+=float(loss)*len(labels); count+=len(labels)
        ids.extend(batch['image_id']); targets.append(labels.cpu().numpy()); probabilities.append(probability.cpu().numpy())
        if collect_features:
            if embedding.shape!=(len(labels),1024) or not torch.isfinite(embedding).all():
                raise ValueError('Invalid penultimate features')
            features.append(embedding.float().cpu().numpy())
    target=np.concatenate(targets); probability=np.concatenate(probabilities)
    result={'validation_loss':total/count,**compute_metrics(target,probability)}
    arrays={'sample_id':np.asarray(ids),'label':target,'probability':probability,
            'predicted_class':probability.argmax(1)}
    if collect_features: arrays['feature']=np.concatenate(features)
    return result,arrays


def save_checkpoint(path,model,optimizer,scheduler,epoch,stage,metrics,metadata):
    tmp=path.with_suffix('.tmp')
    torch.save({'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                'scheduler_state':scheduler.state_dict() if scheduler else None,
                'epoch':epoch,'stage':stage,'metrics':metrics,'metadata':metadata},tmp)
    tmp.replace(path)


def load_checkpoint(path,model):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model_state'],strict=True)
    return {k:checkpoint[k] for k in ['epoch','stage','metrics','metadata']}


def verify_arrays(arrays,records,fold):
    expected={r.image_id:int(r.diagnosis) for r in records}
    ids=arrays['sample_id'].tolist()
    if len(ids)!=len(set(ids)) or dict(zip(ids,arrays['label'].tolist()))!=expected:
        raise ValueError('Export IDs/labels do not exactly match original manifest')
    if arrays['feature'].shape!=(len(records),1024) or not np.isfinite(arrays['feature']).all():
        raise ValueError('Invalid feature array')
    if not np.array_equal(arrays['predicted_class'],arrays['probability'].argmax(1)):
        raise ValueError('Predicted class/probability mismatch')
    arrays['fold']=np.full(len(ids),fold,dtype=np.int64)


def run_fold(fold,zip_path,output,config=Config()):
    if output.exists(): raise FileExistsError(f'Refusing to overwrite {output}')
    train_records,val_records=records_for_fold(fold)
    seed_everything(config.seed)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 GPU support is required')
    model=build_model().cuda()
    weights=inverse_frequency_weights(train_records,5).cuda()
    criterion=WeightedFocalCrossEntropy(weights,gamma=config.gamma,label_smoothing=config.smoothing)
    train_ds=AptosZipDataset(zip_path,train_records,training=True,config=preprocessing(),warn_about_assumptions=False)
    val_ds=AptosZipDataset(zip_path,val_records,training=False,config=preprocessing(),warn_about_assumptions=False)
    train_loader=make_loader(train_ds,batch_size=config.batch_size,shuffle=True,workers=config.workers,seed=config.seed+fold)
    val_loader=make_loader(val_ds,batch_size=config.batch_size,shuffle=False,workers=config.workers,seed=config.seed+1000+fold)
    source=json.loads((HERE/'pretrained_source.json').read_text())
    optimizer,scheduler=set_stage(model,'head',config)
    head_count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata={'fold':fold,'protocol':protocol(),'pretrained':source,'feature_dimension':model.num_features,
              'total_parameters':sum(p.numel() for p in model.parameters()),'head_trainable_parameters':head_count,
              'training_samples':len(train_records),'validation_samples':len(val_records),
              'training_class_counts':np.bincount([r.diagnosis for r in train_records],minlength=5).tolist(),
              'class_weights':weights.cpu().tolist(),'gpu':torch.cuda.get_device_name(0),
              'torch':str(torch.__version__),'timm':timm.__version__,'config':asdict(config),
              'manifest_sha256':digest(MANIFEST)}
    output.mkdir(parents=True,exist_ok=False); write_json(output/'metadata.json',metadata)
    best={'validation_loss':float('inf'),'macro_f1':-float('inf'),'qwk':-float('inf')}
    names={'validation_loss':'best_loss.pt','macro_f1':'best_macro_f1.pt','qwk':'best_qwk.pt'}
    bad=0; history=[]; started=time.monotonic()
    for epoch in range(1,config.max_epochs+1):
        stage='head' if epoch<=config.head_epochs else 'partial'
        if epoch==config.head_epochs+1:
            optimizer,scheduler=set_stage(model,'partial',config); bad=0
            metadata['partial_trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
            write_json(output/'metadata.json',metadata)
        tick=time.monotonic(); torch.cuda.reset_peak_memory_stats()
        lrs={g['name']:g['lr'] for g in optimizer.param_groups}
        training_loss=train_epoch(model,train_loader,criterion,optimizer,stage,config)
        result,_=evaluate(model,val_loader,criterion)
        if scheduler: scheduler.step()
        row={'epoch':epoch,'stage':stage,'train_loss':training_loss,**result,'learning_rates':lrs,
             'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
             'micro_batch_size':config.batch_size,'effective_batch_size':config.effective_batch_size,
             'accumulation_steps':config.effective_batch_size//config.batch_size,
             'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
             'elapsed_seconds':time.monotonic()-tick}
        history.append(row)
        with (output/'history.jsonl').open('a') as handle: handle.write(json.dumps(row,allow_nan=False)+'\n')
        print('EPOCH_COMPLETE '+json.dumps({'fold':fold,**row},allow_nan=False),flush=True)
        improved_loss=result['validation_loss']<best['validation_loss']
        for key,filename in names.items():
            value=result[key]
            if value is None: raise ValueError(f'Undefined checkpoint selection metric {key}')
            improved=value<best[key] if key=='validation_loss' else value>best[key]
            if improved:
                best[key]=value; save_checkpoint(output/filename,model,optimizer,scheduler,epoch,stage,result,metadata)
        bad=0 if improved_loss else bad+1
        if stage=='partial' and bad>=config.patience: break
    training_seconds=time.monotonic()-started
    del train_loader,val_loader
    train_ds.close(); val_ds.close()
    # Release optimizer states before deterministic export.
    del optimizer,scheduler
    official=load_checkpoint(output/'best_loss.pt',model)
    exports={}; primary=None
    for split,records in [('train',train_records),('validation',val_records)]:
        ds=AptosZipDataset(zip_path,records,training=False,config=preprocessing(),warn_about_assumptions=False)
        loader=make_loader(ds,batch_size=config.batch_size,shuffle=False,workers=config.workers,seed=config.seed+1000+fold)
        measured,arrays=evaluate(model,loader,criterion,collect_features=True)
        verify_arrays(arrays,records,fold)
        np.savez_compressed(output/f'{split}_features.npz',**arrays)
        exports[split]={'samples':len(records),'feature_dimension':1024,'file':f'{split}_features.npz'}
        if split=='validation':
            primary=measured
            for key in ['validation_loss','accuracy','macro_f1','macro_auc','qwk']:
                if abs(measured[key]-official['metrics'][key])>1e-5: raise ValueError(f'Restored official metric differs: {key}')
        del loader; ds.close()
    supplements={}
    for key in ['macro_f1','qwk']:
        ckpt=load_checkpoint(output/names[key],model)
        ds=AptosZipDataset(zip_path,val_records,training=False,config=preprocessing(),warn_about_assumptions=False)
        loader=make_loader(ds,batch_size=config.batch_size,shuffle=False,workers=config.workers,seed=config.seed+1000+fold)
        measured,_=evaluate(model,loader,criterion)
        supplements[key]={'epoch':ckpt['epoch'],'metrics':measured,'supplementary_only':True}
        del loader; ds.close()
    phases={stage:{'epochs':[r['epoch'] for r in history if r['stage']==stage],
                   'last_epoch_metrics':next(r for r in reversed(history) if r['stage']==stage)}
            for stage in {r['stage'] for r in history}}
    summary={'fold':fold,'official_selection':'validation_loss','best_epoch':official['epoch'],
             'best_stage':official['stage'],'stopping_epoch':history[-1]['epoch'],
             'final_training_stage':history[-1]['stage'],'training_seconds':training_seconds,
             'total_fold_seconds':time.monotonic()-started,'metrics':primary,
             'supplementary':supplements,'phase_observations':phases,'exports':exports,
             'max_gpu_allocated_bytes':max(r['gpu_peak_allocated_bytes'] for r in history)}
    write_json(output/'summary.json',summary)
    print('FOLD_COMPLETE '+json.dumps(summary,allow_nan=False),flush=True)
    return summary
