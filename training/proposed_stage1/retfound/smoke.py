import sys,tempfile,torch
from pathlib import Path
from .experiment import build_model,features_logits

def main():
 m=build_model().cuda(); m.eval(); x=torch.randn(2,3,224,224,device='cuda')
 with torch.autocast('cuda',dtype=torch.bfloat16): f,z=features_logits(m,x)
 loss=z.float().sum(); loss.backward(); assert f.shape==(2,1024) and z.shape==(2,5)
 p=Path(tempfile.mktemp(suffix='.pt')); torch.save({'model_state':m.state_dict()},p); ck=torch.load(p,map_location='cpu',weights_only=True); m.load_state_dict(ck['model_state']); p.unlink(); print({'forward':True,'backward':True,'bf16':True,'feature_shape':list(f.shape),'logit_shape':list(z.shape),'checkpoint_restore':True,'total_parameters':sum(p.numel() for p in m.parameters())},flush=True)
if __name__=='__main__': main()
