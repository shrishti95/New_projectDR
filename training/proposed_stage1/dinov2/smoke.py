import json,tempfile
from dataclasses import asdict
from pathlib import Path
import torch
from .config import ROOT,HERE,Config
from .experiment import model,set_stage
from training.losses import WeightedFocalCrossEntropy
from .metrics import compute_metrics

def main():
 if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unavailable')
 m=model().cuda(); m.eval(); x=torch.randn(2,3,224,224,device='cuda');y=torch.tensor([0,1],device='cuda')
 with torch.autocast('cuda',dtype=torch.bfloat16):z=m(x)
 assert z.shape==(2,5);loss=WeightedFocalCrossEntropy(torch.ones(5,device='cuda'))(z.float(),y);loss.backward()
 opt=set_stage(m,'head',Config());opt.step();
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):f=m.forward_head(m.forward_features(x),pre_logits=True)
 assert f.shape==(2,768) and torch.isfinite(f).all()
 p=torch.softmax(z.float(),1).detach().cpu().numpy();assert compute_metrics(y.cpu().numpy(),p)['qwk'] is not None
 out={'passed':True,'variant':'vit_base_patch14_dinov2.lvd142m','feature_dimension':768,'total_parameters':sum(p.numel() for p in m.parameters()),'head_trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'input_resolution':224,'bf16_stable':True,'feature_shape':list(f.shape),'metrics':'validated'}
 (HERE/'smoke_result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
