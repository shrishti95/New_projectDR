from __future__ import annotations
import hashlib,json,time,sys,os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
from preprocessing import AptosZipDataset
from training.train_benchmark import seed_everything,inverse_frequency_weights,load_fold_records
from training.train_portable import make_loader
from training.losses import WeightedFocalCrossEntropy
from .config import *
from .metrics import compute_metrics
from .retfound_model import RETFound_mae

def digest(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def build_model():
 if not CHECKPOINT.exists(): raise FileNotFoundError(CHECKPOINT)
 m=RETFound_mae(num_classes=5,global_pool=True,img_size=224)
 ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False); state=ck['model'] if isinstance(ck,dict) and 'model' in ck else ck
 # Encoder-only load: remove MAE decoder/mask and incompatible prediction head.
 state={k:v for k,v in state.items() if not k.startswith(('decoder_','decoder','mask_token')) and k not in ('head.weight','head.bias','fc_norm.weight','fc_norm.bias')}
 msg=m.load_state_dict(state,strict=False)
 allowed={'head.weight','head.bias','fc_norm.weight','fc_norm.bias'}
 if set(msg.missing_keys)-allowed: raise RuntimeError(f'unexpected missing RETFound keys: {msg.missing_keys}')
 torch.nn.init.trunc_normal_(m.head.weight,std=2e-5); torch.nn.init.zeros_(m.head.bias)
 return m

def records(fold):
 if digest(MANIFEST)!=MANIFEST_SHA256: raise RuntimeError('manifest changed')
 return load_fold_records(MANIFEST,fold)
def set_stage(m,stage,c):
 m.requires_grad_(False); m.head.requires_grad_(True)
 if stage=='partial':
  for b in m.blocks[-c.final_blocks:]: b.requires_grad_(True)
  if hasattr(m,'fc_norm'): m.fc_norm.requires_grad_(True)
  else: m.norm.requires_grad_(True)
 groups=[{'params':list(m.head.parameters()),'lr':c.head_warmup_lr if stage=='head' else c.head_lr,'name':'classifier'}]
 if stage=='partial':
  ids={id(p) for p in m.head.parameters()}; groups.insert(0,{'params':[p for p in m.parameters() if p.requires_grad and id(p) not in ids],'lr':c.backbone_lr,'name':'retfound_final_blocks'})
 return torch.optim.AdamW(groups,weight_decay=c.weight_decay)
def features_logits(m,x):
 f=m.forward_features(x)
 if f.ndim==3: f=f[:,0]
 return f,m.head(f)
def train_epoch(m,loader,loss,opt,c):
 m.train();
 if any(not p.requires_grad for b in m.blocks for p in b.parameters()):
  for b in m.blocks:
   if not any(p.requires_grad for p in b.parameters()): b.eval()
 total=n=0; opt.zero_grad(set_to_none=True)
 for b in loader:
  x=b['image'].cuda(non_blocking=True); y=b['label'].cuda(non_blocking=True)
  with torch.autocast('cuda',dtype=torch.bfloat16): _,z=features_logits(m,x)
  l=loss(z.float(),y)
  if not torch.isfinite(l): raise FloatingPointError('nonfinite training loss')
  l.backward(); torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],c.gradient_clip,error_if_nonfinite=True); opt.step(); opt.zero_grad(set_to_none=True)
  total+=float(l)*len(y); n+=len(y)
 return total/n
@torch.inference_mode()
def evaluate(m,loader,loss):
 m.eval(); ys=[]; ps=[]; ids=[]; fs=[]; total=n=0
 for b in loader:
  x=b['image'].cuda(non_blocking=True); y=b['label'].cuda(non_blocking=True)
  with torch.autocast('cuda',dtype=torch.bfloat16): f,z=features_logits(m,x)
  l=loss(z.float(),y); p=torch.softmax(z.float(),1)
  if not torch.isfinite(l) or not torch.isfinite(p).all(): raise FloatingPointError('nonfinite eval')
  total+=float(l)*len(y); n+=len(y); ys.append(y.cpu().numpy()); ps.append(p.cpu().numpy()); ids+=b['image_id']; fs.append(f.float().cpu().numpy())
 y=np.concatenate(ys); p=np.concatenate(ps)
 return {'validation_loss':total/n,**compute_metrics(y,p)},{'sample_id':np.asarray(ids),'label':y,'probability':p,'predicted_class':p.argmax(1),'feature':np.concatenate(fs)}
def run_fold(fold,zip_path,out,c=Config()):
 if out.exists() and (out/'summary.json').exists(): raise FileExistsError(out)
 tr,va=records(fold); seed_everything(c.seed); torch.use_deterministic_algorithms(True)
 if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 unavailable')
 m=build_model().cuda(); weights=inverse_frequency_weights(tr,5).cuda(); loss=WeightedFocalCrossEntropy(weights,gamma=c.gamma,label_smoothing=c.smoothing)
 pc=preprocessing(); td=AptosZipDataset(zip_path,tr,training=True,config=pc,warn_about_assumptions=False); vd=AptosZipDataset(zip_path,va,training=False,config=pc,warn_about_assumptions=False)
 tl=make_loader(td,batch_size=c.batch_size,shuffle=True,workers=c.workers,seed=42+fold); vl=make_loader(vd,batch_size=c.batch_size,shuffle=False,workers=c.workers,seed=1042+fold)
 out.mkdir(parents=True,exist_ok=True); write(out/'metadata.json',{'fold':fold,'architecture':'official RETFound MAE ViT-Large/16','checkpoint':str(CHECKPOINT),'feature_dimension':1024,'total_parameters':sum(p.numel() for p in m.parameters()),'train_samples':len(tr),'validation_samples':len(va),'class_weights':weights.cpu().tolist(),'manifest_sha256':digest(MANIFEST),'gpu':torch.cuda.get_device_name(0),'config':asdict(c)})
 best=float('inf'); bad=0; history=[]; opt=None; sched=None
 for ep in range(1,c.max_epochs+1):
  stage='head' if ep<=c.head_epochs else 'partial'
  if ep==1 or ep==c.head_epochs+1:
   opt=set_stage(m,stage,c); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=c.max_epochs-c.head_epochs,eta_min=0.) if stage=='partial' else None
  t=time.time(); trloss=train_epoch(m,tl,loss,opt,c); res,_=evaluate(m,vl,loss)
  if sched: sched.step()
  row={'epoch':ep,'stage':stage,'train_loss':trloss,'epoch_seconds':time.time()-t,**res,'learning_rates':{g['name']:g['lr'] for g in opt.param_groups},'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad)}; history.append(row)
  with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  print('RETFOUND_EPOCH '+json.dumps({'fold':fold,**row},allow_nan=False),flush=True)
  if res['validation_loss']<best: best=res['validation_loss']; bad=0; torch.save({'model_state':m.state_dict(),'epoch':ep,'metrics':res,'config':asdict(c)},out/'best_loss.pt')
  else: bad+=1
  if stage=='partial' and bad>=c.patience: break
 ck=torch.load(out/'best_loss.pt',map_location='cpu',weights_only=True); m.load_state_dict(ck['model_state']); final,arr=evaluate(m,vl,loss)
 np.savez_compressed(out/'validation_features.npz',fold=np.full(len(va),fold),**arr)
 trl=make_loader(AptosZipDataset(zip_path,tr,training=False,config=pc,warn_about_assumptions=False),batch_size=c.batch_size,shuffle=False,workers=c.workers,seed=1042+fold); _,ta=evaluate(m,trl,loss); np.savez_compressed(out/'train_features.npz',fold=np.full(len(tr),fold),**ta)
 write(out/'summary.json',{'fold':fold,'best_epoch':ck['epoch'],'stopping_epoch':history[-1]['epoch'],'metrics':final,'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'feature_dimension':1024,'exports':['validation_features.npz','train_features.npz']})
 print('RETFOUND_FOLD_COMPLETE '+json.dumps({'fold':fold,**final},allow_nan=False),flush=True)
