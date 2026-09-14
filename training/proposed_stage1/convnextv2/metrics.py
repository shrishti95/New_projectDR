import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix)


def compute_metrics(target, probability):
    target=np.asarray(target); probability=np.asarray(probability,dtype=np.float64)
    if target.ndim!=1 or probability.shape!=(len(target),5) or len(target)==0:
        raise ValueError('Expected nonempty targets and N x 5 probabilities')
    if not np.isfinite(probability).all() or np.any(probability<0) or not np.allclose(probability.sum(1),1,atol=1e-5):
        raise ValueError('Invalid probability matrix')
    if not np.isin(target,np.arange(5)).all():
        raise ValueError('Invalid target labels')
    pred=probability.argmax(1)
    precision,recall,f1,support=precision_recall_fscore_support(target,pred,labels=range(5),zero_division=0)
    auc=[]
    for c in range(5):
        positive=(target==c)
        auc.append(float(roc_auc_score(positive,probability[:,c])) if 0<positive.sum()<len(target) else None)
    cm=confusion_matrix(target,pred,labels=range(5))
    conf=probability.max(1); correct=pred==target
    bins=np.minimum((conf*15).astype(int),14)
    ece=sum(float((bins==b).mean())*abs(float(correct[bins==b].mean())-float(conf[bins==b].mean()))
            for b in range(15) if (bins==b).any())
    def finite(value):
        return float(value) if np.isfinite(value) else None
    return {'accuracy':float(accuracy_score(target,pred)), 'macro_f1':float(f1.mean()),
            'macro_auc':float(np.mean(auc)) if all(x is not None for x in auc) else None,
            'cohens_kappa_unweighted':finite(cohen_kappa_score(target,pred,labels=range(5))),
            'qwk':finite(cohen_kappa_score(target,pred,labels=range(5),weights='quadratic')),
            'mcc':float(matthews_corrcoef(target,pred)), 'ece':float(ece),
            'per_class':[dict(grade=c,precision=float(precision[c]),recall=float(recall[c]),
                              f1=float(f1[c]),support=int(support[c]),auc=auc[c]) for c in range(5)],
            'confusion_matrix':cm.tolist(), 'grade_1_recall':float(recall[1]),
            'grade_3_recall':float(recall[3]), 'grade_2_as_3':int(cm[2,3]),
            'grade_3_as_2':int(cm[3,2])}
