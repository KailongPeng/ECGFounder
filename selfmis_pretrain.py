"""
SelfMIS Pre-training Script

Implements self-alignment learning via SigLIP loss between:
  - fs: single-lead ECG encoder (Lead I, trainable)
  - fm: multi-lead ECG encoder (12-lead, frozen / stop-gradient)

Both encoders use Net1D backbone initialized from ECGFounder checkpoints.

Usage:
    python selfmis_pretrain.py --ablation full --data_source ptbxl \
        --ptbxl_root /home/p00929643/ECG/data/physionet.org/files/ptb-xl/1.0.1/ \
        --ptbxl_csv  /home/p00929643/ECG/data/physionet.org/files/ptb-xl/1.0.1/ptbxl_database.csv

Ablation variants:
    full             fs=1-lead ckpt, fm=12-lead ckpt  (default)
    no_s_pretrained  fs=random init,  fm=12-lead ckpt
    no_m_pretrained  fs=1-lead ckpt,  fm=random init
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from net1d import Net1D
from selfmis_dataset import MIMICECGPretrainDataset, PTBXLPretrainDataset

# ============================================================
# Default checkpoint paths
# ============================================================
CKPT_1LEAD  = './checkpoint/1_lead_ECGFounder.pth'
CKPT_12LEAD = './checkpoint/12_lead_ECGFounder.pth'


# ============================================================
# SigLIP Loss
# ============================================================

class SigLIPLoss(nn.Module):
    """
    Sigmoid loss for self-alignment (adapted from SigLIP, Zhai et al. 2023).

    For a batch B of (single-lead, multi-lead) pairs from the same recording:
      - Positive pairs: (S_i, M_i)  → z_ij = +1
      - Negative pairs: (S_i, M_j), i≠j → z_ij = -1

    L = -1/B * sum_ij log(sigmoid(z_ij * (scale * <S_i, M_j>)))

    Temperature (scale = 1/t) is learnable; initialized to 1/0.07 ≈ 14.3.
    """

    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        # log_scale = log(1/t), so scale = exp(log_scale)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))

    def forward(self, s_emb: torch.Tensor, m_emb: torch.Tensor) -> torch.Tensor:
        """
        s_emb: (B, D) L2-normalized single-lead embeddings
        m_emb: (B, D) L2-normalized multi-lead embeddings (already detached upstream)
        Returns: scalar loss
        """
        B = s_emb.shape[0]
        scale = self.log_scale.exp()
        logits = scale * (s_emb @ m_emb.T)                        # (B, B)
        # +1 on diagonal (positive pairs), -1 elsewhere
        labels = 2.0 * torch.eye(B, device=s_emb.device) - 1.0   # (B, B)
        loss = -torch.mean(torch.log(torch.sigmoid(labels * logits) + 1e-8))
        return loss


# ============================================================
# Encoder loading
# ============================================================

def load_encoder(device: torch.device, pth: str,
                 in_channels: int, pretrained: bool = True) -> Net1D:
    """
    Build a Net1D encoder with return_features=True.

    If pretrained=True, loads ECGFounder backbone weights (dense layer excluded).
    The model is placed on device and returned in eval mode with requires_grad=True
    for fs; caller is responsible for freezing fm.
    """
    model = Net1D(
        in_channels=in_channels,
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
        return_features=True,
    )
    if pretrained:
        ckpt = torch.load(pth, map_location=device, weights_only=False)
        sd = {k: v for k, v in ckpt['state_dict'].items()
              if not k.startswith('dense.')}
        model.load_state_dict(sd, strict=False)
    return model.to(device)


def build_encoders(device: torch.device,
                   ablation: str = 'full',
                   single_pth: str = CKPT_1LEAD,
                   multi_pth: str = CKPT_12LEAD):
    """
    Build (fs, fm) encoder pair according to ablation variant.

    ablation:
        'full'            fs from 1-lead ckpt, fm from 12-lead ckpt
        'no_s_pretrained' fs random init,       fm from 12-lead ckpt
        'no_m_pretrained' fs from 1-lead ckpt,  fm random init
    fm is always frozen (all requires_grad=False).
    """
    assert ablation in ('full', 'no_s_pretrained', 'no_m_pretrained'), \
        f"Unknown ablation: {ablation}"

    fs_pretrained = ablation != 'no_s_pretrained'
    fm_pretrained = ablation != 'no_m_pretrained'

    fs = load_encoder(device, single_pth, in_channels=1,  pretrained=fs_pretrained)
    fm = load_encoder(device, multi_pth,  in_channels=12, pretrained=fm_pretrained)

    # Freeze fm completely
    for param in fm.parameters():
        param.requires_grad = False
    fm.eval()

    return fs, fm


# ============================================================
# SelfMIS dual-encoder model
# ============================================================

class SelfMISModel(nn.Module):
    """
    Wraps (fs, fm) encoders.
    fs: single-lead (updated during pre-training)
    fm: multi-lead  (frozen / stop-gradient)
    """

    def __init__(self, fs: Net1D, fm: Net1D):
        super().__init__()
        self.fs = fs
        self.fm = fm

    def encode_single(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 5000) -> (B, 1024) L2-normalized"""
        _, feat = self.fs(x)
        return F.normalize(feat, dim=-1)

    def encode_multi(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 12, 5000) -> (B, 1024) L2-normalized, no gradient"""
        with torch.no_grad():
            _, feat = self.fm(x)
        return F.normalize(feat.detach(), dim=-1)

    def forward(self, x_single: torch.Tensor, x_multi: torch.Tensor):
        s_emb = self.encode_single(x_single)
        m_emb = self.encode_multi(x_multi)
        return s_emb, m_emb


# ============================================================
# Pre-training loop
# ============================================================

def pretrain(
    data_source: str,
    ptbxl_root: str = None,
    ptbxl_csv: str = None,
    mimic_root: str = None,
    mimic_max_records: int = None,
    single_pth: str = CKPT_1LEAD,
    multi_pth: str = CKPT_12LEAD,
    save_dir: str = './checkpoint/selfmis_pretrain',
    ablation: str = 'full',
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    num_workers: int = 4,
    resume: str = None,
    train_folds: list = None,
    val_folds: list = None,
):
    """
    SelfMIS pre-training.

    Saves checkpoints to:
        save_dir/{ablation}/selfmis_pretrain_epoch{NN}.pth
        save_dir/{ablation}/selfmis_pretrained_fs.pth   (best val loss)
    """
    save_dir = os.path.join(save_dir, ablation)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  Ablation: {ablation}  |  Data: {data_source}')

    # ---- Datasets ----
    if train_folds is None:
        train_folds = list(range(1, 9))    # folds 1-8
    if val_folds is None:
        val_folds = [9]

    if data_source == 'ptbxl':
        assert ptbxl_root and ptbxl_csv, 'ptbxl_root and ptbxl_csv required'
        train_ds = PTBXLPretrainDataset(ptbxl_root, ptbxl_csv, folds=train_folds)
        val_ds   = PTBXLPretrainDataset(ptbxl_root, ptbxl_csv, folds=val_folds)
    elif data_source == 'mimic':
        assert mimic_root, 'mimic_root required'
        # MIMIC does not have official folds; use a random 95/5 split
        full_ds = MIMICECGPretrainDataset(mimic_root, max_records=mimic_max_records)
        n_val = max(1, int(0.05 * len(full_ds)))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42))
    else:
        raise ValueError(f'Unknown data_source: {data_source}')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f'Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples')

    # ---- Model & loss ----
    fs, fm = build_encoders(device, ablation, single_pth, multi_pth)
    model = SelfMISModel(fs, fm).to(device)
    criterion = SigLIPLoss(init_temp=0.07).to(device)

    # Only update fs parameters + learnable temperature
    trainable_params = list(model.fs.parameters()) + [criterion.log_scale]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    scaler = GradScaler()

    start_epoch = 0
    best_val_loss = float('inf')

    # ---- Resume from checkpoint ----
    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.fs.load_state_dict(ckpt['state_dict'])
        criterion.log_scale.data.fill_(ckpt.get('log_scale', math.log(1.0 / 0.07)))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f'Resumed from epoch {ckpt["epoch"]}  val_loss={best_val_loss:.4f}')

    # ---- Training loop ----
    for epoch in range(start_epoch, epochs):
        # -- train --
        model.fs.train()
        train_loss = _run_epoch(model, criterion, train_loader, device,
                                optimizer=optimizer, scaler=scaler,
                                grad_clip=grad_clip, train=True)

        # -- validate --
        model.fs.eval()
        val_loss = _run_epoch(model, criterion, val_loader, device, train=False)

        scheduler.step()

        temp = math.exp(-criterion.log_scale.item())   # current temperature
        print(f'Epoch {epoch:02d}/{epochs}  '
              f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
              f'temp={temp:.4f}  lr={scheduler.get_last_lr()[0]:.2e}')

        # -- save epoch checkpoint --
        ckpt_path = os.path.join(save_dir, f'selfmis_pretrain_epoch{epoch:02d}.pth')
        _save_checkpoint(ckpt_path, model, criterion, optimizer, scheduler,
                         epoch, val_loss, ablation)

        # -- save best model --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(save_dir, 'selfmis_pretrained_fs.pth')
            _save_checkpoint(best_path, model, criterion, optimizer, scheduler,
                             epoch, val_loss, ablation)
            print(f'  -> Best model saved (val_loss={val_loss:.4f})')

    print(f'\nPre-training complete. Best checkpoint: {best_path}')
    return best_path


def _run_epoch(model, criterion, loader, device,
               optimizer=None, scaler=None, grad_clip=1.0, train=True):
    total_loss = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x_single, x_multi in tqdm(loader,
                                       desc='train' if train else 'val',
                                       leave=False):
            x_single = x_single.to(device, non_blocking=True)
            x_multi  = x_multi.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    s_emb, m_emb = model(x_single, x_multi)
                    loss = criterion(s_emb, m_emb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.fs.parameters() if p.requires_grad]
                    + [criterion.log_scale],
                    grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast():
                    s_emb, m_emb = model(x_single, x_multi)
                    loss = criterion(s_emb, m_emb)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def _save_checkpoint(path, model, criterion, optimizer, scheduler,
                     epoch, val_loss, ablation):
    torch.save({
        'epoch':      epoch,
        'state_dict': model.fs.state_dict(),
        'log_scale':  criterion.log_scale.item(),
        'optimizer':  optimizer.state_dict(),
        'scheduler':  scheduler.state_dict(),
        'val_loss':   val_loss,
        'ablation':   ablation,
    }, path)


# ============================================================
# CLI entry point
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='SelfMIS Pre-training')
    p.add_argument('--data_source', default='ptbxl',
                   choices=['ptbxl', 'mimic'])
    p.add_argument('--ptbxl_root',
                   default='/home/p00929643/ECG/data/physionet.org/files/ptb-xl/1.0.1/')
    p.add_argument('--ptbxl_csv',
                   default='/home/p00929643/ECG/data/physionet.org/files/ptb-xl/1.0.1/ptbxl_database.csv')
    p.add_argument('--mimic_root',    default=None)
    p.add_argument('--mimic_max',     type=int, default=None)
    p.add_argument('--single_pth',    default=CKPT_1LEAD)
    p.add_argument('--multi_pth',     default=CKPT_12LEAD)
    p.add_argument('--save_dir',      default='./checkpoint/selfmis_pretrain')
    p.add_argument('--ablation',      default='full',
                   choices=['full', 'no_s_pretrained', 'no_m_pretrained'])
    p.add_argument('--epochs',        type=int,   default=20)
    p.add_argument('--batch_size',    type=int,   default=128)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--weight_decay',  type=float, default=0.1)
    p.add_argument('--grad_clip',     type=float, default=1.0)
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--resume',        default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pretrain(
        data_source=args.data_source,
        ptbxl_root=args.ptbxl_root,
        ptbxl_csv=args.ptbxl_csv,
        mimic_root=args.mimic_root,
        mimic_max_records=args.mimic_max,
        single_pth=args.single_pth,
        multi_pth=args.multi_pth,
        save_dir=args.save_dir,
        ablation=args.ablation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        resume=args.resume,
    )
