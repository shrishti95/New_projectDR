"""Create exact model-state recovery copies without changing original checkpoints."""
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

for name in ['retfound','swinv2']:
    base=ROOT/'proposed_stage1'/name
    manifest=[]
    for fold in range(5):
        src=base/'runs'/f'fold_{fold}'/'best_loss.pt'
        dst=base/'recovery'/f'fold_{fold}_best_weights.pt'
        dst.parent.mkdir(exist_ok=True)
        ck=torch.load(src,map_location='cpu',weights_only=True)
        compact={k:v for k,v in ck.items() if k not in ['optimizer_state','scheduler_state']}
        if not dst.exists(): torch.save(compact,dst)
        restored=torch.load(dst,map_location='cpu',weights_only=True)
        assert restored['model_state'].keys()==ck['model_state'].keys()
        assert all(torch.equal(v,restored['model_state'][k]) for k,v in ck['model_state'].items())
        manifest.append({'file':str(dst.relative_to(ROOT.parent)),'sha256':sha(dst),'bytes':dst.stat().st_size,'fold':fold,'epoch':ck['epoch'],'selection':'minimum validation loss','tensor_exact':True})
        print(name,fold,dst.stat().st_size,flush=True)
        del ck,compact,restored
    (base/'recovery/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
