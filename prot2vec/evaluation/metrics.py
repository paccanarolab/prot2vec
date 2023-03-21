from sklearn.metrics import auc
from sklearn.metrics._ranking import _binary_clf_curve
from rich.progress import track
import numpy as np
import pandas as pd

"""For the following functions, we assume that the gold standard and predictions
have a shape of (proteins, goterms)."""


def get_metrics(y_true, y_pred, information_content):
    rounded_pred = y_pred
    if len(np.unique(y_pred)) > 10000:
        rounded_pred = np.around(y_pred, decimals=4)

    per_gene = get_metrics_per_gene(y_true, rounded_pred, information_content)
    per_term = get_metrics_per_term(y_true, rounded_pred, information_content)
    return pd.concat([per_gene, per_term])


def _get_metrics(y_true, y_pred, information_content):
    N = np.prod(y_true.shape[0])

    qty_pos = np.sum(y_true > 0)
    qty_neg = N - qty_pos


    fp, tp, th = _binary_clf_curve(y_true, y_pred, pos_label=1.0)

    tn = qty_neg - fp
    fn = qty_pos - tp

    if qty_pos * qty_neg == 0 or th.shape[0] < 2:
        pre = np.array([0, 1])
        rec = np.array([qty_pos / N, qty_pos / N])
        f_max = 0
        tpr = np.array([0, 1])
        fpr = np.array([0, 1])
        auc_roc = 0.5
        auc_pr = qty_pos / N
    else:

        rec = tp / (tp + fn)
        pre = tp / (tp + fp)

        f_max = np.max(2.0 * (np.multiply(pre, rec)) / (pre + rec + np.finfo(np.double).tiny))

        fpr = fp / qty_neg
        tpr = tp / qty_pos

        auc_roc = auc(fpr, tpr)
        auc_pr = auc(rec, pre)

    ru = np.zeros(th.shape[0])
    mi = np.zeros(th.shape[0])
    s = np.zeros(th.shape[0])

    for i, t in enumerate(th):
        false_neg = ((y_pred < t) & (y_true > 0)).astype(float)
        false_pos = ((y_pred >= t) & (y_true < 1)).astype(float)
        ru[i] = np.sum(false_neg * information_content) / N
        mi[i] = np.sum(false_pos * information_content) / N
        s[i] = np.sqrt(ru[i]**2 + mi[i]**2)

    s_min = s.min()

    return {
        'tp': tp, 'fp': fp,
        'tn': tn, 'fn': fn,
        'qty_pos': qty_pos,
        'qty_neg': qty_neg,
        'rec': rec,
        'pre': pre,
        'f_max': f_max,
        'fpr': fpr,
        'tpr': tpr,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'ru': ru,
        'mi': mi,
        's': s,
        's_min': s_min
    }


def get_metrics_per_gene(y_true, y_pred, information_content):
    metrics = []
    for row in track(range(y_true.shape[0]), description="Calculating per gene metrics..."):
        true = y_true[row, :].flatten()
        pred = y_pred[row, :].flatten()
        metrics.append(_get_metrics(true, pred, information_content))
        metrics[-1]['metric_type'] = 'per_gene'
    return pd.DataFrame(metrics)


def get_metrics_per_term (y_true, y_pred, information_content):
    metrics = []
    for col in track(range(y_true.shape[1]), description="Calculating per term metrics..."):
        true = y_true[:, col].flatten()
        pred = y_pred[:, col].flatten()
        metrics.append(_get_metrics(true, pred, information_content[col]))
        metrics[-1]['metric_type'] = 'per_term'
    return pd.DataFrame(metrics)

if __name__ == '__main__':
    import os
    data_dir = "/Users/torresmateo/OneDrive - FGV/prot2vec"
    if os.name == 'nt':
        data_dir = "D:/OneDrive - FGV/prot2vec"
    with open(os.path.join(data_dir, 'metric_test.npy'), 'rb') as f:
        y_proba = np.load(f)
        y_test = np.load(f)
        ic_sorted = np.load(f)
    df = get_metrics(y_test, y_proba, ic_sorted)
    print('done')