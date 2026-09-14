"""Smoke-gated, sequential five-fold experiment with persistent progress logs."""
from dataclasses import replace,asdict
from datetime import datetime,timezone
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from .config import ROOT,HERE,Config,protocol
from .experiment import digest,write_json,run_fold
from .report import write_report
from .smoke import source_hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold',type=int,choices=range(5))
    parser.add_argument('--zip',type=Path,default=ROOT/'aptos2019-blindness-detection.zip')
    args=parser.parse_args()
    smoke=json.loads((HERE/'smoke_result.json').read_text())
    if not smoke['passed']: raise RuntimeError('Smoke test has not passed')
    for relative,expected in smoke['source_hashes'].items():
        if digest(ROOT/relative)!=expected: raise RuntimeError(f'Source changed after smoke test: {relative}')
    config=replace(Config(),batch_size=smoke['config']['batch_size'])
    if config.effective_batch_size%config.batch_size: raise ValueError('Invalid accumulation factor')
    if args.fold is not None:
        run_fold(args.fold,args.zip,HERE/'runs'/f'fold_{args.fold}',config)
        return
    lock=(HERE/'experiment.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**kw):
        value={'stage':stage,'pid':os.getpid(),'updated_utc':datetime.now(timezone.utc).isoformat(),**kw}
        write_json(HERE/'status.json',value); print(json.dumps(value),flush=True)
    try:
        planned=protocol(); planned['training']=asdict(config)
        planned['pretrained']=json.loads((HERE/'pretrained_source.json').read_text())
        planned['source_hashes']=source_hashes()
        if (HERE/'protocol.json').exists():
            if json.loads((HERE/'protocol.json').read_text())!=planned:
                raise RuntimeError('Predeclared protocol differs; do not overwrite it')
        else: write_json(HERE/'protocol.json',planned)
        (HERE/'runs').mkdir(exist_ok=True); (HERE/'logs').mkdir(exist_ok=True)
        # Verify ZIP labels against the unchanged manifest before any full fold.
        from preprocessing.dataset import load_aptos_records
        from .experiment import records_for_fold
        train,val=records_for_fold(0)
        uploaded=load_aptos_records(args.zip)
        if {r.image_id:r.diagnosis for r in uploaded}!={r.image_id:r.diagnosis for r in train+val}:
            raise ValueError('Uploaded labels do not match original manifest')
        write_report()
        env=os.environ.copy(); env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',
                                         MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        for fold in range(5):
            for relative,expected in planned['source_hashes'].items():
                if digest(ROOT/relative)!=expected: raise RuntimeError(f'Source drift: {relative}')
            folder=HERE/'runs'/f'fold_{fold}'
            if (folder/'summary.json').exists():
                write_report(); continue
            if folder.exists(): raise RuntimeError(f'Incomplete fold requires inspection: {folder}')
            log=HERE/'logs'/f'fold_{fold}.log'
            with log.open('x') as handle:
                child=subprocess.Popen([sys.executable,'-u','-m','training.proposed_stage1.convnextv2.run',
                                        '--fold',str(fold),'--zip',str(args.zip.resolve())],cwd=ROOT,env=env,
                                       stdout=handle,stderr=subprocess.STDOUT)
                status('training',fold=fold,child_pid=child.pid,log=str(log))
                if child.wait()!=0: raise RuntimeError(f'Fold {fold} failed; inspect {log}')
            summary=json.loads((folder/'summary.json').read_text())
            print('FOLD_OFFICIAL_METRICS '+json.dumps({'fold':fold,'best_epoch':summary['best_epoch'],**summary['metrics']}),flush=True)
            write_report()
        status('complete',report=str(HERE/'convnextv2_five_fold_report.md'))
    except Exception as exc:
        status('failed',error=str(exc)); raise


if __name__=='__main__': main()
