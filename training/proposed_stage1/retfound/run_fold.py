import argparse
from pathlib import Path
from .experiment import run_fold
from .config import HERE,ROOT
p=argparse.ArgumentParser(); p.add_argument('--fold',type=int,required=True); p.add_argument('--zip',default=str(ROOT/'aptos2019-blindness-detection.zip')); p.add_argument('--out',default=None); a=p.parse_args(); run_fold(a.fold,a.zip,Path(a.out) if a.out else HERE/'runs'/f'fold_{a.fold}')
