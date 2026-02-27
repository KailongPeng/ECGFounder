"""
SelfMIS Evaluation Script

Evaluates a fine-tuned SelfMIS (or baseline) model on the PTB-XL test set
(fold 10) for 9-class MI detection. Reports per-MI-type AUROC with 95%
bootstrap confidence intervals.

Also provides compare_ablations() to generate a side-by-side comparison table
across ablation variants.

Usage:
    # Evaluate one model
    python selfmis_eval.py \
        --model_pth ./checkpoint/selfmis_finetune/full/best_model.pth \
        --save_dir  ./res/selfmis/full

    # Compare all ablations
    python selfmis_eval.py --compare
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

from net1d import Net1D
from selfmis_dataset import MI_CODES, PTBXLMIDataset
from selfmis_finetune import build_finetune_model

# ============================================================
# Constants
# ============================================================

PTB_ROOT = '/home/p00929643/ECG/data/physionet.org/files/ptb-xl/1.0.1/'
PTB_CSV  = os.path.join(PTB_ROOT, 'ptbxl_database.csv')

# Directories for each ablation variant (used by compare_ablations)
DEFAULT_RESULT_DIRS = {
    'full':             './res/selfmis/full',
    'no_s_pretrained':  './res/selfmis/no_s_pretrained',
    'no_m_pretrained':  './res/selfmis/no_m_pretrained',
    'alignment_disabled': './res/selfmis/alignment_disabled',
}


# ============================================================
# Metric helpers
# ============================================================

def compute_metrics(gt: np.ndarray, pred: np.ndarray,
                    threshold: float) -> dict:
    """Per-class metrics at a fixed threshold."""
    pred_bin = (pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(gt, pred_bin, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1   = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0.0
    if len(np.unique(gt)) < 2:
        auroc = float('nan')
        auprc = float('nan')
    else:
        auroc = roc_auc_score(gt, pred)
        auprc = average_precision_score(gt, pred)
    return dict(sens=sens, spec=spec, ppv=ppv, npv=npv,
                f1=f1, auroc=auroc, auprc=auprc)


def find_optimal_threshold(gt: np.ndarray, pred: np.ndarray,
                            n_thresholds: int = 100) -> float:
    """Youden-J optimal threshold (maximises sens+spec-1)."""
    if len(np.unique(gt)) < 2:
        return 0.5
    thresholds = np.linspace(0, 1, n_thresholds)
    best_t, best_j = 0.5, -1.0
    for t in thresholds:
        pred_bin = (pred >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(gt, pred_bin, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, t
    return best_t


def bootstrap_metric(gt: np.ndarray, pred: np.ndarray,
                     threshold: float, key: str,
                     n_bootstrap: int = 1000,
                     seed: int = 42) -> tuple:
    """Returns (lower_95ci, upper_95ci) for a given metric key."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(gt), len(gt))
        m = compute_metrics(gt[idx], pred[idx], threshold)
        v = m[key]
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < 2:
        return (float('nan'), float('nan'))
    return (float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


# ============================================================
# Main evaluation function
# ============================================================

def evaluate_selfmis(
    model_pth: str,
    ptbxl_root: str = PTB_ROOT,
    ptbxl_csv: str = PTB_CSV,
    save_dir: str = './res/selfmis/full',
    n_classes: int = 9,
    batch_size: int = 256,
    num_workers: int = 4,
    n_bootstrap: int = 1000,
    pretrained_format: str = 'selfmis',
    threshold: float = 0.0,
) -> dict:
    """
    Evaluate a fine-tuned model on PTB-XL test set (fold 10).

    Saves:
        save_dir/all_gt.csv
        save_dir/all_pred.csv
        save_dir/res.csv        (per-MI-type metrics + 95% CI)
        save_dir/summary.json   (mean metrics)

    Returns: summary dict (mean AUROC, etc.)
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Dataset ----
    test_ds = PTBXLMIDataset(ptbxl_root, ptbxl_csv,
                             folds=[10], threshold=threshold)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    print(f'Test set: {len(test_ds)} recordings')

    # ---- Model ----
    model = build_finetune_model(device, model_pth, n_classes=n_classes,
                                 linear_prob=False,
                                 pretrained_format=pretrained_format)
    # Load the full fine-tuned state_dict (including dense head)
    ckpt = torch.load(model_pth, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'Loaded: {model_pth}')

    # ---- Inference ----
    all_gt, all_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='Inference'):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            prob   = torch.sigmoid(logits).cpu().numpy()
            all_pred.append(prob)
            all_gt.append(y.numpy())

    all_gt   = np.concatenate(all_gt)    # (N, 9)
    all_pred = np.concatenate(all_pred)  # (N, 9)

    pd.DataFrame(all_gt,   columns=MI_CODES).to_csv(
        os.path.join(save_dir, 'all_gt.csv'),   index=False, float_format='%.5f')
    pd.DataFrame(all_pred, columns=MI_CODES).to_csv(
        os.path.join(save_dir, 'all_pred.csv'), index=False, float_format='%.5f')

    # ---- Per-class evaluation ----
    rows = []
    aurocs = []
    for i, code in enumerate(MI_CODES):
        gt_i   = all_gt[:, i]
        pred_i = all_pred[:, i]
        thresh = find_optimal_threshold(gt_i, pred_i)
        m      = compute_metrics(gt_i, pred_i, thresh)

        row = {'MI_code': code, 'threshold': round(thresh, 4)}
        for key in ('auroc', 'auprc', 'sens', 'spec', 'ppv', 'npv', 'f1'):
            row[key] = round(m[key], 4) if not np.isnan(m[key]) else float('nan')
            if n_bootstrap > 0 and key in ('auroc', 'sens', 'spec', 'f1'):
                lo, hi = bootstrap_metric(gt_i, pred_i, thresh, key,
                                          n_bootstrap=n_bootstrap)
                row[f'{key}_ci_lo'] = round(lo, 4)
                row[f'{key}_ci_hi'] = round(hi, 4)
        rows.append(row)
        if not np.isnan(m['auroc']):
            aurocs.append(m['auroc'])

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(save_dir, 'res.csv'), index=False)

    # ---- Summary ----
    mean_auroc = float(np.mean(aurocs)) if aurocs else float('nan')
    summary = {
        'mean_auroc': round(mean_auroc, 4),
        'n_valid_classes': len(aurocs),
        'per_class': {row['MI_code']: row['auroc'] for row in rows},
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- Print ----
    print(f'\n{"MI type":<12} {"AUROC":>7} {"Sens":>7} {"Spec":>7} {"F1":>7}')
    print('-' * 48)
    for row in rows:
        auc  = f'{row["auroc"]:.4f}' if not np.isnan(row['auroc']) else '   NaN'
        sens = f'{row["sens"]:.4f}'
        spec = f'{row["spec"]:.4f}'
        f1   = f'{row["f1"]:.4f}'
        print(f'{row["MI_code"]:<12} {auc:>7} {sens:>7} {spec:>7} {f1:>7}')
    print('-' * 48)
    print(f'{"Mean AUROC":<12} {mean_auroc:>7.4f}  (over {len(aurocs)} valid classes)')
    print(f'\nResults saved to {save_dir}/')

    return summary


# ============================================================
# Ablation comparison table
# ============================================================

def compare_ablations(
    result_dirs: dict = None,
    output_path: str = './res/selfmis/comparison_table.csv',
) -> pd.DataFrame:
    """
    Load summary.json from each ablation directory and build a comparison table.

    result_dirs: dict mapping ablation name → result directory
    Columns: MI_code | full | no_s_pretrained | no_m_pretrained | alignment_disabled
    """
    if result_dirs is None:
        result_dirs = DEFAULT_RESULT_DIRS

    # Load per-class AUROC for each ablation
    data = {'MI_code': MI_CODES + ['Mean']}
    for name, rdir in result_dirs.items():
        summary_path = os.path.join(rdir, 'summary.json')
        if not os.path.isfile(summary_path):
            print(f'  [skip] {name}: {summary_path} not found')
            data[name] = [float('nan')] * (len(MI_CODES) + 1)
            continue
        with open(summary_path) as f:
            s = json.load(f)
        per_class = s.get('per_class', {})
        aurocs = [per_class.get(code, float('nan')) for code in MI_CODES]
        aurocs.append(s.get('mean_auroc', float('nan')))
        data[name] = aurocs

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.4f')

    print('\n=== Ablation Comparison (AUROC) ===')
    print(df.to_string(index=False))
    print(f'\nSaved to {output_path}')
    return df


# ============================================================
# CLI entry point
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='SelfMIS Evaluation')
    p.add_argument('--model_pth',  default=None,
                   help='Path to fine-tuned model checkpoint')
    p.add_argument('--ptbxl_root', default=PTB_ROOT)
    p.add_argument('--ptbxl_csv',  default=PTB_CSV)
    p.add_argument('--save_dir',   default='./res/selfmis/full')
    p.add_argument('--n_classes',  type=int, default=9)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers',type=int, default=4)
    p.add_argument('--n_bootstrap',type=int, default=1000,
                   help='Bootstrap resamples for CI (0 to skip)')
    p.add_argument('--pretrained_format', default='selfmis',
                   choices=['selfmis', 'ecgfounder'])
    p.add_argument('--threshold',  type=float, default=0.0)
    p.add_argument('--compare',    action='store_true',
                   help='Run compare_ablations() instead of single model eval')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.compare:
        compare_ablations()
    else:
        assert args.model_pth, '--model_pth is required for single model evaluation'
        evaluate_selfmis(
            model_pth=args.model_pth,
            ptbxl_root=args.ptbxl_root,
            ptbxl_csv=args.ptbxl_csv,
            save_dir=args.save_dir,
            n_classes=args.n_classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_bootstrap=args.n_bootstrap,
            pretrained_format=args.pretrained_format,
            threshold=args.threshold,
        )
