"""
Zero-shot 9-MI Evaluation using ECGFounder 1-lead checkpoint.

Loads the 1-lead ECGFounder (150-class), runs inference on PTB-XL fold 10,
extracts the 9 MI-relevant columns from the 150-class sigmoid output,
and computes per-class and mean AUROC.

Mapping from SCP MI codes to ECGFounder 150-class indices
(verified against tasks.txt and ptbxl_label.csv empirical co-occurrence):

    ALMI  → idx 54  ANTEROLATERAL INFARCT
    AMI   → idx 17  ANTERIOR INFARCT
    ASMI  → idx 40  ANTEROSEPTAL INFARCT
    ILMI  → idx 24  LATERAL INFARCT
    IMI   → idx 61  INFERIOR INFARCT
    IPLMI → idx 88  INFERIOR-POSTERIOR INFARCT
    IPMI  → idx 88  INFERIOR-POSTERIOR INFARCT
    LMI   → idx 24  LATERAL INFARCT
    PMI   → idx 132 POSTERIOR INFARCT

Usage:
    python eval_zeroshot_mi9.py
    python eval_zeroshot_mi9.py --ckpt ./checkpoint/1_lead_ECGFounder.pth
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from net1d import Net1D
from selfmis_dataset import MI_CODES, PTBXLMIDataset, preprocess_ecg, read_wfdb

# ============================================================
# MI SCP code → ECGFounder 150-class index mapping
# ============================================================
MI_TO_ECGF_IDX = {
    'ALMI':  54,   # ANTEROLATERAL INFARCT
    'AMI':   17,   # ANTERIOR INFARCT
    'ASMI':  40,   # ANTEROSEPTAL INFARCT
    'ILMI':  24,   # LATERAL INFARCT
    'IMI':   61,   # INFERIOR INFARCT
    'IPLMI': 88,   # INFERIOR-POSTERIOR INFARCT
    'IPMI':  88,   # INFERIOR-POSTERIOR INFARCT
    'LMI':   24,   # LATERAL INFARCT
    'PMI':  132,   # POSTERIOR INFARCT
}

# Ordered extraction indices (matching MI_CODES order)
EXTRACT_INDICES = [MI_TO_ECGF_IDX[code] for code in MI_CODES]

# ============================================================
# Constants
# ============================================================
PTB_ROOT = '/home/p00929643/ECG/data/physionet.org/files/ptb-xl/1.0.1/'
PTB_CSV  = os.path.join(PTB_ROOT, 'ptbxl_database.csv')
CKPT_1LEAD = './checkpoint/1_lead_ECGFounder.pth'


def load_model(ckpt_path: str, device: torch.device) -> Net1D:
    """Load 1-lead ECGFounder with 150-class output (no modification)."""
    model = Net1D(
        in_channels=1,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=150,
        return_features=False,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['state_dict']
    model.load_state_dict(sd, strict=True)
    model = model.to(device)
    model.eval()
    return model


def evaluate(ckpt_path: str = CKPT_1LEAD,
             ptbxl_root: str = PTB_ROOT,
             ptbxl_csv: str = PTB_CSV,
             test_folds: list = None,
             batch_size: int = 64,
             num_workers: int = 4,
             sampling_rate: int = 500):
    """
    Zero-shot evaluation: extract MI-relevant columns from 150-class output.
    """
    if test_folds is None:
        test_folds = [10]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Test folds: {test_folds}')
    print(f'Sampling rate: {sampling_rate}Hz')

    # Load model
    model = load_model(ckpt_path, device)

    # Load test dataset (single-lead)
    test_ds = PTBXLMIDataset(
        ptbxl_root, ptbxl_csv,
        folds=test_folds,
        threshold=0.0,
        lead='single',
        sampling_rate=sampling_rate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f'Test samples: {len(test_ds)}')

    # Positive counts
    labels_all = test_ds.get_label_array()
    pos_counts = labels_all.sum(axis=0).astype(int)
    print(f'MI positive counts: '
          + ', '.join(f'{MI_CODES[i]}={pos_counts[i]}' for i in range(len(MI_CODES))))

    # Inference — get 150-class sigmoid outputs
    all_gt = []
    all_pred_150 = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='Inference'):
            x = x.to(device, non_blocking=True)
            logits_150 = model(x)                          # (B, 150)
            prob_150 = torch.sigmoid(logits_150).cpu().numpy()
            all_pred_150.append(prob_150)
            all_gt.append(y.numpy())

    all_gt = np.concatenate(all_gt)           # (N, 9)
    all_pred_150 = np.concatenate(all_pred_150)  # (N, 150)

    # Extract the 9 MI-relevant columns
    all_pred_mi9 = all_pred_150[:, EXTRACT_INDICES]  # (N, 9)

    # Compute per-class AUROC
    print(f"\n{'MI code':<10} {'ECGFounder idx':>14} {'AUROC':>8} {'N_pos':>6}")
    print('-' * 44)
    aurocs = []
    for i, code in enumerate(MI_CODES):
        gt_i = all_gt[:, i]
        pred_i = all_pred_mi9[:, i]
        n_pos = int(gt_i.sum())
        if len(np.unique(gt_i)) < 2:
            auc = float('nan')
            print(f'{code:<10} {EXTRACT_INDICES[i]:>14d} {"N/A":>8} {n_pos:>6}')
        else:
            auc = roc_auc_score(gt_i, pred_i)
            aurocs.append(auc)
            print(f'{code:<10} {EXTRACT_INDICES[i]:>14d} {auc:>8.4f} {n_pos:>6}')

    mean_auroc = float(np.mean(aurocs)) if aurocs else float('nan')
    print('-' * 44)
    print(f'{"Mean":<10} {"":>14} {mean_auroc:>8.4f}')
    print(f'\nClasses with valid AUROC: {len(aurocs)}/{len(MI_CODES)}')

    return mean_auroc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Zero-shot MI9 evaluation')
    p.add_argument('--ckpt', default=CKPT_1LEAD)
    p.add_argument('--ptbxl_root', default=PTB_ROOT)
    p.add_argument('--ptbxl_csv', default=PTB_CSV)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--sampling_rate', type=int, default=500, choices=[100, 500])
    args = p.parse_args()

    evaluate(
        ckpt_path=args.ckpt,
        ptbxl_root=args.ptbxl_root,
        ptbxl_csv=args.ptbxl_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampling_rate=args.sampling_rate,
    )
