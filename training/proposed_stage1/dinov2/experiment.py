from __future__ import annotations
import csv,hashlib,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch, timm
from preprocessing import AptosZipDataset
from training.train_benchmark import seed_everything,inverse_frequency_weights,load_fold_records
from training.train_portable import make_loader
from training.losses import WeightedFocalCrossEntropy
from .config import *
from .metrics import compute_metrics

def digest(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def model(): return timm.create_model(MODEL,pretrained=True,num_classes=5,img_size=224)
def records(fold):
 if digest(MANIFEST)!=MANIFEST_SHA256: raise RuntimeError('manifest changed')
 return load_fold_records(MANIFEST,fold)
def set_stage(m,stage,c):
 m.requires_grad_(False); m.head.requires_grad_(True)
 if stage=='partial':
  for b in m.blocks[-c.final_blocks:]: b.requires_grad_(True)
  m.norm.requires_grad_(True)
 head_params=list(m.head.parameters()); head_ids={id(p) for p in head_params}
 groups=[{'params':head_params,'lr':c.head_warmup_lr if stage=='head' else c.head_lr,'name':'head'}]
 if stage=='partial': groups.insert(0,{'params':[p for p in m.parameters() if p.requires_grad and id(p) not in head_ids],'lr':c.backbone_lr,'name':'backbone_final_blocks'})
 return torch.optim.AdamW(groups,weight_decay=c.weight_decay)
def train_epoch(m,loader,loss,opt,c):
 m.train();m.blocks[:-(c.final_blocks if any(p.requires_grad for b in m.blocks for p in b.parameters()) else 0)].eval();opt.zero_grad(set_to_none=True);total=n=0
 for batch in loader:
  x=batch['image'].cuda();y=batch['label'].cuda()
  with torch.autocast('cuda',dtype=torch.bfloat16): z=m(x)
  l=loss(z.float(),y)
  if not torch.isfinite(l): raise FloatingPointError('nonfinite loss')
  l.backward();torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],c.gradient_clip,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
  total+=float(l)*len(y);n+=len(y)
 return total/n
@torch.inference_mode()
def evaluate(m,loader,loss,features=False):
 m.eval(); ally=[];allp=[];ids=[];fs=[];total=n=0
 for b in loader:
  x=b['image'].cuda();y=b['label'].cuda()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   f=m.forward_features(x); f=m.forward_head(f,pre_logits=True); z=m.head(f)
  l=loss(z.float(),y);p=torch.softmax(z.float(),1)
  if not torch.isfinite(l) or not torch.isfinite(p).all():raise FloatingPointError('nonfinite eval')
  total+=float(l)*len(y);n+=len(y);ally.append(y.cpu().numpy());allp.append(p.cpu().numpy());ids+=b['image_id'];fs.append(f.float().cpu().numpy())
 y=np.concatenate(ally);p=np.concatenate(allp);return {'validation_loss':total/n,**compute_metrics(y,p)},{'sample_id':np.asarray(ids),'label':y,'probability':p,'predicted_class':p.argmax(1),'feature':np.concatenate(fs)}
def run_fold(fold,zip_path,out,c=Config()):
 if out.exists():raise FileExistsError(out)
 tr,va=records(fold);seed_everything(c.seed);torch.use_deterministic_algorithms(True)
 if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unavailable')
 m=model().cuda();loss=WeightedFocalCrossEntropy(inverse_frequency_weights(tr,5).cuda(),gamma=2,label_smoothing=.1)
 pc=preprocessing();td=AptosZipDataset(zip_path,tr,training=True,config=pc,warn_about_assumptions=False);vd=AptosZipDataset(zip_path,va,training=False,config=pc,warn_about_assumptions=False)
 tl=make_loader(td,batch_size=32,shuffle=True,workers=c.workers,seed=42+fold);vl=make_loader(vd,batch_size=32,shuffle=False,workers=c.workers,seed=1042+fold)
 out.mkdir(parents=True);write(out/'metadata.json',{'fold':fold,'model':MODEL,'feature_dimension':m.num_features,'total_parameters':sum(p.numel() for p in m.parameters()),'train_samples':len(tr),'validation_samples':len(va),'trainable_head':sum(p.numel() for p in m.parameters() if p.requires_grad),'config':asdict(c),'manifest_sha256':digest(MANIFEST),'gpu':torch.cuda.get_device_name(0)})
 best=float('inf');bad=0;history=[];opt=None;sched=None
 for ep in range(1,31):
  stage='head' if ep<=3 else 'partial'
  if ep==1 or ep==4: opt=set_stage(m,stage,c);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=27,eta_min=0) if stage=='partial' else None
  trloss=train_epoch(m,tl,loss,opt,c);res,_=evaluate(m,vl,loss)
  if sched:sched.step()
  row={'epoch':ep,'stage':stage,'train_loss':trloss,**res,'learning_rates':{g['name']:g['lr'] for g in opt.param_groups},'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad)};history.append(row)
  with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  print('DINO_EPOCH '+json.dumps({'fold':fold,**row},allow_nan=False),flush=True)
  if res['validation_loss']<best:
   best=res['validation_loss'];bad=0;torch.save({'model_state':m.state_dict(),'epoch':ep,'metrics':res,'config':asdict(c)},out/'best_loss.pt')
  else:bad+=1
  if stage=='partial' and bad>=5:break
 ck=torch.load(out/'best_loss.pt',map_location='cpu',weights_only=True);m.load_state_dict(ck['model_state']);final,arr=evaluate(m,vl,loss,True)
 np.savez_compressed(out/'validation_features.npz',fold=np.full(len(va),fold),**arr)
 train_eval,train_arr=evaluate(m,make_loader(AptosZipDataset(zip_path,tr,training=False,config=pc,warn_about_assumptions=False),batch_size=32,shuffle=False,workers=c.workers,seed=1042+fold),loss,True)
 np.savez_compressed(out/'train_features.npz',fold=np.full(len(tr),fold),**train_arr)
 write(out/'summary.json',{'fold':fold,'best_epoch':ck['epoch'],'stopping_epoch':history[-1]['epoch'],'metrics':final,'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'exports':['validation_features.npz','train_features.npz']})
 print('DINO_FOLD_COMPLETE '+json.dumps({'fold':fold,**final},allow_nan=False),flush=True)
