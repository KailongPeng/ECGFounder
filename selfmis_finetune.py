"""
SelfMIS Fine-tuning Script

Fine-tunes the SelfMIS-pretrained single-lead encoder (fs) on PTB-XL
for 9-class myocardial infarction detection (Lead I only).

Usage:
    # After pre-training (SelfMIS)
    python selfmis_finetune.py \
        --fs_pth ./checkpoint/selfmis_pretrain/full/selfmis_pretrained_fs.pth \
        --ablation full

    # Baseline: ECGFounder 1-lead without SelfMIS pre-training
    python selfmis_finetune.py \
        --fs_pth ./checkpoint/1_lead_ECGFounder.pth \
        --ablation alignment_disabled \
        --pretrained_format ecgfounder
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from net1d import Net1D
from selfmis_dataset import MI_CODES, PTBXLMIDataset

# ============================================================
# Constants
# ============================================================

PTB_ROOT = '/home/p00929643/ECG/data/physionet.org/files/ptb-xl/1.0.1/'
PTB_CSV  = os.path.join(PTB_ROOT, 'ptbxl_database.csv')


# ============================================================
# Model builder
# ============================================================

def build_finetune_model(device: torch.device,
                         fs_pth: str,
                         n_classes: int = 9,
                         linear_prob: bool = True,
                         pretrained_format: str = 'selfmis') -> Net1D:
    """
    Load fs backbone and prepare for fine-tuning.

    Args:
        fs_pth:            Path to checkpoint file.
        n_classes:         Output classes (9 for MI detection).
        linear_prob:       If True, freeze all layers except the dense head.
        pretrained_format: 'selfmis'    → ckpt['state_dict'] is the complete fs Net1D
                           'ecgfounder' → original 150-class ECGFounder checkpoint

    Returns:
        Net1D model with dense layer replaced by Linear(1024, n_classes).
    """
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
        n_classes=n_classes,
        return_features=False,
    )

    ckpt = torch.load(fs_pth, map_location=device, weights_only=False)
    sd = ckpt['state_dict']

    if pretrained_format == 'ecgfounder':
        # Strip the 150-class dense layer; load backbone only
        sd = {k: v for k, v in sd.items() if not k.startswith('dense.')}
        model.load_state_dict(sd, strict=False)
    else:
        # SelfMIS checkpoint: state_dict is already the full fs (may have dense.*)
        # We strip the old dense layer and let the new one initialise randomly
        sd = {k: v for k, v in sd.items() if not k.startswith('dense.')}
        model.load_state_dict(sd, strict=False)

    # Replace dense head for the downstream task
    model.dense = nn.Linear(1024, n_classes)

    if linear_prob:
        for name, param in model.named_parameters():
            if 'dense' not in name:
                param.requires_grad = False

    return model.to(device)


# ============================================================
# Class-imbalance helper
# ============================================================

def compute_pos_weight(labels: np.ndarray,
                       clip_min: float = 1.0,
                       clip_max: float = 100.0) -> torch.Tensor:
    """
    Compute per-class positive weight for BCEWithLogitsLoss.
    pos_weight[i] = neg_count[i] / pos_count[i], clipped to [clip_min, clip_max].
    labels: (N, C) binary float array
    """
    pos = labels.sum(axis=0) + 1e-6
    neg = (1.0 - labels).sum(axis=0) + 1e-6
    pw = np.clip(neg / pos, clip_min, clip_max)
    return torch.FloatTensor(pw)


# ============================================================
# Evaluation helper (AUROC per class)
# ============================================================

def _compute_auroc(all_gt: np.ndarray, all_pred: np.ndarray,
                   mi_codes: list) -> dict:
    from sklearn.metrics import roc_auc_score
    results = {}
    aurocs = []
    for i, code in enumerate(mi_codes):
        gt_i   = all_gt[:, i]
        pred_i = all_pred[:, i]
        if len(np.unique(gt_i)) < 2:
            results[code] = float('nan')
        else:
            auc = roc_auc_score(gt_i, pred_i)
            results[code] = auc
            aurocs.append(auc)
    results['mean'] = float(np.mean(aurocs)) if aurocs else float('nan')
    return results


def _infer(model: Net1D, loader: DataLoader,
           device: torch.device):
    """Run inference and return (all_gt, all_pred) numpy arrays."""
    model.eval()
    all_gt, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()
            all_pred.append(prob)
            all_gt.append(y.numpy())
    return np.concatenate(all_gt), np.concatenate(all_pred)


# ============================================================
# Fine-tuning loop
# ============================================================

def finetune(
    fs_pth: str,
    ptbxl_root: str = PTB_ROOT,
    ptbxl_csv: str = PTB_CSV,
    save_dir: str = './checkpoint/selfmis_finetune',
    ablation: str = 'full',
    pretrained_format: str = 'selfmis',
    n_classes: int = 9,
    linear_prob: bool = True,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
    num_workers: int = 4,
    threshold: float = 0.0,
):
    """
    Fine-tune the single-lead encoder on PTB-XL MI task.

    Train/val/test split follows the official PTB-XL stratified folds:
      train: folds 1-8  (≈17,441 recordings)
      val:   fold 9     (≈ 2,193 recordings, used for early stopping)
      test:  fold 10    (≈ 2,203 recordings, final evaluation)

    Saves:
        save_dir/{ablation}/best_model.pth   (best val mean AUROC)
        save_dir/{ablation}/final_model.pth  (last epoch)
    """
    save_dir = os.path.join(save_dir, ablation)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  Ablation: {ablation}  |  linear_prob: {linear_prob}')

    # ---- Datasets ----
    train_ds = PTBXLMIDataset(ptbxl_root, ptbxl_csv,
                              folds=list(range(1, 9)), threshold=threshold)
    val_ds   = PTBXLMIDataset(ptbxl_root, ptbxl_csv, folds=[9],  threshold=threshold)
    test_ds  = PTBXLMIDataset(ptbxl_root, ptbxl_csv, folds=[10], threshold=threshold)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}')

    # ---- MI class positive counts ----
    train_labels = train_ds.get_label_array()
    pos_counts = train_labels.sum(axis=0).astype(int)
    print(f'Positive counts per MI class: '
          + ', '.join(f'{MI_CODES[i]}={pos_counts[i]}' for i in range(len(MI_CODES))))

    # ---- Model ----
    model = build_finetune_model(device, fs_pth, n_classes=n_classes,
                                 linear_prob=linear_prob,
                                 pretrained_format=pretrained_format)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'Trainable params: {trainable:,} / {total:,}')

    # ---- Loss (with class-imbalance weights) ----
    pos_weight = compute_pos_weight(train_labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ---- Optimizer ----
    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.AdamW(opt_params, lr=lr, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    # ---- Training ----
    best_val_auroc   = -1.0
    patience_counter = 0
    best_path = os.path.join(save_dir, 'best_model.pth')

    for epoch in range(epochs):
        # -- train --
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch:02d} train', leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # -- validate --
        val_gt, val_pred = _infer(model, val_loader, device)
        val_auroc_dict   = _compute_auroc(val_gt, val_pred, MI_CODES)
        val_mean_auroc   = val_auroc_dict['mean']

        scheduler.step()

        print(f'Epoch {epoch:02d}  train_loss={train_loss:.4f}  '
              f'val_mean_AUROC={val_mean_auroc:.4f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}')

        # -- early stopping --
        if val_mean_auroc > best_val_auroc:
            best_val_auroc   = val_mean_auroc
            patience_counter = 0
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'val_auroc': val_mean_auroc,
                        'ablation': ablation},
                       best_path)
            print(f'  -> Best model saved (val_AUROC={val_mean_auroc:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch} (patience={patience})')
                break

    # -- save final model --
    final_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'val_auroc': val_mean_auroc,
                'ablation': ablation},
               final_path)

    # ---- Test evaluation (using best model) ----
    print('\n=== Test evaluation (best model) ===')
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['state_dict'])

    test_gt, test_pred = _infer(model, test_loader, device)
    test_auroc_dict    = _compute_auroc(test_gt, test_pred, MI_CODES)

    print(f"{'MI type':<12} {'AUROC':>7}")
    print('-' * 22)
    for code in MI_CODES:
        auc = test_auroc_dict[code]
        print(f'{code:<12} {auc:>7.4f}' if not np.isnan(auc) else f'{code:<12}     NaN')
    print('-' * 22)
    print(f"{'Mean':<12} {test_auroc_dict['mean']:>7.4f}")

    # Save results
    res_df = pd.DataFrame({
        'MI_code': MI_CODES,
        'AUROC':   [test_auroc_dict[c] for c in MI_CODES],
    })
    res_df.to_csv(os.path.join(save_dir, 'test_results.csv'), index=False)
    print(f'\nResults saved to {save_dir}/test_results.csv')

    return test_auroc_dict


# ============================================================
# CLI entry point
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='SelfMIS Fine-tuning')
    p.add_argument('--fs_pth', required=True,
                   help='Path to pretrained fs checkpoint '
                        '(selfmis_pretrained_fs.pth or 1_lead_ECGFounder.pth)')
    p.add_argument('--ptbxl_root', default=PTB_ROOT)
    p.add_argument('--ptbxl_csv',  default=PTB_CSV)
    p.add_argument('--save_dir',   default='./checkpoint/selfmis_finetune')
    p.add_argument('--ablation',   default='full',
                   choices=['full', 'no_s_pretrained', 'no_m_pretrained',
                            'alignment_disabled'])
    p.add_argument('--pretrained_format', default='selfmis',
                   choices=['selfmis', 'ecgfounder'])
    p.add_argument('--n_classes',    type=int,   default=9)
    p.add_argument('--linear_prob',  action='store_true', default=True)
    p.add_argument('--full_finetune', action='store_true',
                   help='Full fine-tuning (all layers); overrides --linear_prob')
    p.add_argument('--epochs',       type=int,   default=20)
    p.add_argument('--batch_size',   type=int,   default=64)
    p.add_argument('--lr',           type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience',     type=int,   default=5)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--threshold',    type=float, default=0.0)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    linear_prob = not args.full_finetune
    finetune(
        fs_pth=args.fs_pth,
        ptbxl_root=args.ptbxl_root,
        ptbxl_csv=args.ptbxl_csv,
        save_dir=args.save_dir,
        ablation=args.ablation,
        pretrained_format=args.pretrained_format,
        n_classes=args.n_classes,
        linear_prob=linear_prob,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )
