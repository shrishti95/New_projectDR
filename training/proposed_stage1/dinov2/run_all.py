"""Sequentially run DINOv2 folds 1-4 after the existing fold 0."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
from .config import ROOT,HERE

def main():
    out=HERE/'runs'; logs=HERE/'logs'; logs.mkdir(exist_ok=True)
    status=HERE/'status.json'
    zip_path=ROOT/'aptos2019-blindness-detection.zip'
    for fold in range(5):
        folder=out/f'fold_{fold}'
        summary=folder/'summary.json'
        if summary.exists():
            print(f'FOLD_{fold}_ALREADY_COMPLETE',flush=True);continue
        if fold==0:
            while not summary.exists():
                if not Path('/proc/57975').exists() and not (folder/'best_loss.pt').exists():
                    raise RuntimeError('Existing fold 0 stopped without summary')
                time.sleep(10)
            continue
        if folder.exists(): raise RuntimeError(f'Incomplete fold directory: {folder}')
        log=logs/f'fold_{fold}.log'
        env=os.environ.copy();env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        code=("from training.proposed_stage1.dinov2.experiment import run_fold; "
              "from training.proposed_stage1.dinov2.config import ROOT; "
              f"run_fold({fold},ROOT/'aptos2019-blindness-detection.zip',ROOT/'training/proposed_stage1/dinov2/runs/fold_{fold}')")
        with log.open('x') as handle:
            p=subprocess.Popen([sys.executable,'-u','-c',code],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            status.write_text(json.dumps({'stage':'training','fold':fold,'pid':os.getpid(),'child_pid':p.pid,'log':str(log)},indent=2)+'\n')
            if p.wait()!=0: raise RuntimeError(f'fold {fold} failed; inspect {log}')
    status.write_text(json.dumps({'stage':'complete','folds':5,'report':'dinov2_five_fold_report.md'},indent=2)+'\n')
    print('DINOV2_ALL_FOLDS_COMPLETE',flush=True)
if __name__=='__main__': main()
