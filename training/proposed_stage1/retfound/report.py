import json
from pathlib import Path
import numpy as np
from .config import HERE,protocol
runs=HERE/'runs'; rows=[]
for f in range(5):
 p=runs/f'fold_{f}'/'summary.json'
 if p.exists(): rows.append(json.loads(p.read_text()))
if len(rows)==5:
 keys=['accuracy','macro_f1','macro_auc','cohen_kappa','qwk','mcc','grade1_recall','grade3_recall']
 lines=['# RETFound five-fold report','', 'Official model: RETFound MAE ViT-Large/16, `RETFound_mae_natureCFP.pth` from the official Hugging Face repository.','', '| Metric | Mean ± sample SD |','|---|---:|']
 for k in keys:
  v=np.array([r['metrics'][k] for r in rows],float); lines.append(f'| {k} | {v.mean():.6f} ± {v.std(ddof=1):.6f} |')
 lines += ['', '## Fold results','', '| Fold | Best epoch | Val loss | Accuracy | Macro F1 | QWK | Grade 1 recall | Grade 3 recall |','|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  m=r['metrics']; lines.append(f"| {r['fold']} | {r['best_epoch']} | {m['validation_loss']:.6f} | {m['accuracy']:.6f} | {m['macro_f1']:.6f} | {m['qwk']:.6f} | {m['grade1_recall']:.6f} | {m['grade3_recall']:.6f} |")
 (HERE/'retfound_five_fold_report.md').write_text('\n'.join(lines)+'\n'); (HERE/'retfound_five_fold_report.json').write_text(json.dumps({'protocol':protocol(),'folds':rows},indent=2))
 print('\n'.join(lines))
else: print(f'completed folds: {len(rows)}/5')
