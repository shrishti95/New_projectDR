"""Swin V2 adapter for the validated ConvNeXt V2 experiment runner."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import timm
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file
from timm.models.swin_transformer_v2 import checkpoint_filter_fn
from training.proposed_stage1.convnextv2 import config as base_config
from training.proposed_stage1.convnextv2 import experiment as base

HERE = Path(__file__).resolve().parent
MODEL = 'swinv2_base_window8_256.ms_in1k'
CONFIG = dataclasses.replace(base_config.Config(), input_size=256)

def preprocessing():
    return dataclasses.replace(base_config.preprocessing(), image_size=(256,256))

def protocol():
    p = base_config.protocol()
    p.update(model=MODEL, training=dataclasses.asdict(CONFIG),
             preprocessing=dataclasses.asdict(preprocessing()),
             stage2='epochs 4-30: final two Swin stages, final normalization, classifier',
             resolution_note='Native 256 input differs from previous 224 runs; comparisons include this difference.')
    return p

def prepare():
    path = HERE/'pretrained_source.json'
    if path.exists():
        return
    repo = 'timm/'+MODEL
    revision = HfApi().model_info(repo).sha
    checkpoint = hf_hub_download(repo, 'model.safetensors', revision=revision)
    base.write_json(path, {'repo':repo, 'revision':revision, 'path':checkpoint,
                         'sha256':base.digest(checkpoint), 'pretraining':'Microsoft ImageNet-1k',
                         'input_resolution':256, 'feature_dimension':1024})
    base.write_json(HERE/'protocol.json', protocol())

def build_model():
    source=json.loads((HERE/'pretrained_source.json').read_text())
    if base.digest(source['path']) != source['sha256']:
        raise RuntimeError('Pretrained checkpoint hash mismatch')
    m=timm.create_model(MODEL, pretrained=False)
    m.load_state_dict(checkpoint_filter_fn(load_file(source['path']),m), strict=True)
    m.reset_classifier(5)
    return m

def training_mode(m,stage):
    m.eval()
    m.get_classifier().train()
    if stage=='partial':
        for part in [*m.layers[2:],m.norm]: part.train()

def set_stage(m,stage,c):
    m.requires_grad_(False)
    m.get_classifier().requires_grad_(True)
    if stage=='partial':
        for part in [*m.layers[2:],m.norm]: part.requires_grad_(True)
    training_mode(m,stage)
    head=list(m.get_classifier().parameters()); ids={id(p) for p in head}
    groups=[{'params':head,'lr':c.head_warmup_lr if stage=='head' else c.head_lr,'name':'classifier'}]
    if stage=='partial':
        groups.insert(0,{'params':[p for p in m.parameters() if p.requires_grad and id(p) not in ids],
                         'lr':c.backbone_lr,'name':'backbone_final_two_stages'})
    opt=torch.optim.AdamW(groups,weight_decay=c.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=c.max_epochs-c.head_epochs) if stage=='partial' else None
    return opt,scheduler

# Load a private instance: overrides never mutate the completed ConvNeXt module.
spec=importlib.util.spec_from_file_location('training.proposed_stage1.convnextv2._swin_adapter',base.__file__)
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.HERE=HERE
runner.MODEL=MODEL
runner.build_model=build_model
runner.preprocessing=preprocessing
runner.protocol=protocol
runner.set_stage=set_stage
runner.training_mode=training_mode

def smoke(zip_path):
    base.seed_everything(42)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 unavailable')
    tr,va=base.records_for_fold(0)
    subset=[next(r for r in tr if r.diagnosis==g) for g in range(5)]
    subset += [r for r in tr if r not in subset][:27]
    ds=base.AptosZipDataset(zip_path,subset,training=False,config=preprocessing(),warn_about_assumptions=False)
    loader=base.make_loader(ds,batch_size=32,shuffle=False,workers=0,seed=42)
    batch=next(iter(loader)); x=batch['image'].cuda(); y=batch['label'].cuda()
    m=build_model().cuda()
    criterion=base.WeightedFocalCrossEntropy(base.inverse_frequency_weights(tr,5).cuda(),gamma=2,label_smoothing=.1)
    counts={}; losses=[]
    for stage in ['head','partial']:
        opt,_=set_stage(m,stage,CONFIG)
        counts[stage]=sum(p.numel() for p in m.parameters() if p.requires_grad)
        steps=8 if stage=='head' else 1
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            with base.autocast(): z=m(x)
            loss=criterion(z.float(),y)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite smoke loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            opt.step()
            if stage=='head': losses.append(loss.item())
    if losses[-1]>=losses[0]: raise RuntimeError('Smoke head loss did not decrease')
    metrics,arr=runner.evaluate(m,loader,criterion,collect_features=True)
    runner.verify_arrays(arr,subset,0)
    perfect=base.compute_metrics(np.arange(5),np.eye(5))
    for k in ['accuracy','macro_f1','macro_auc','qwk','mcc']:
        assert abs(perfect[k]-1)<1e-8,(k,perfect[k])
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        path=Path(tmp)/'restore.pt'
        torch.save(m.state_dict(),path)
        with torch.no_grad(): m.get_classifier().weight.add_(1)
        m.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'),strict=True)
        _,restored=runner.evaluate(m,loader,criterion,collect_features=True)
        np.testing.assert_allclose(arr['probability'],restored['probability'],atol=1e-6)
    result={'passed':True,'batch_size':32,'bf16':True,'total_parameters':sum(p.numel() for p in m.parameters()),
            'trainable_parameters':counts,'head':str(m.get_classifier()),'feature_shape':list(arr['feature'].shape),
            'head_loss_sequence':losses,'checkpoint_restore':True,'sample_alignment':True,'metric_test':True,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'timm':timm.__version__}
    base.write_json(HERE/'smoke_result.json',result)
    print('SMOKE_PASSED '+json.dumps(result),flush=True)
    ds.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','train'],required=True)
    p.add_argument('--fold',type=int,default=0);a=p.parse_args()
    prepare()
    zip_path=base_config.ROOT/'aptos2019-blindness-detection.zip'
    if a.mode=='smoke': smoke(zip_path)
    else:
        if not json.loads((HERE/'smoke_result.json').read_text())['passed']: raise RuntimeError('Smoke required')
        runner.run_fold(a.fold,zip_path,HERE/'runs'/f'fold_{a.fold}',CONFIG)

if __name__=='__main__': main()
