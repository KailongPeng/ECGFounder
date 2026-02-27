"""
SelfMIS Dataset Classes

Provides dataset classes for:
  - PTBXLPretrainDataset: PTB-XL pre-training (returns Lead I + 12-lead pair)
  - PTBXLMIDataset:       PTB-XL fine-tuning (9-class MI labels, Lead I only)
  - MIMICECGPretrainDataset: MIMIC-IV-ECG pre-training (interface reserved)

Preprocessing follows the existing eval pipeline in run_ptbxl_eval_1lead.py:
  z-score normalize -> resample to 5000 samples (no bandpass filter).
"""

import ast
import os

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

# ============================================================
# 9 MI SCP codes evaluated in the SelfMIS paper (Table 1)
# ============================================================
MI_CODES = ['ALMI', 'AMI', 'ASMI', 'ILMI', 'IMI', 'IPLMI', 'IPMI', 'LMI', 'PMI']

# Standard 12-lead order used by ECGFounder (HEEDB / PTB-XL)
STANDARD_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# MIMIC-IV-ECG lead order (aVF and aVL are swapped vs standard)
MIMIC_LEADS = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# Reorder indices: for each lead in STANDARD_LEADS, find its index in MIMIC_LEADS
MIMIC_TO_STANDARD = [MIMIC_LEADS.index(lead) for lead in STANDARD_LEADS]


# ============================================================
# Shared preprocessing utilities
# ============================================================

def z_score(signal: np.ndarray) -> np.ndarray:
    """Per-recording z-score normalization. signal: (C, T)"""
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def resample_unequal(ts: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """
    Resample (C, T_in) to (C, fs_out) via linear interpolation.
    Matches existing ECGFounder preprocessing exactly.
    """
    if fs_in == 0 or ts.shape[1] == 0 or fs_in == fs_out:
        return ts
    if 2 * fs_out == fs_in:          # fast path: 1000Hz -> 500 samples
        return ts[:, ::2]
    t = ts.shape[1] / fs_in
    x_old = np.linspace(0, t, num=ts.shape[1], endpoint=True)
    x_new = np.linspace(0, t, num=fs_out, endpoint=True)
    out = np.zeros((ts.shape[0], fs_out), dtype=ts.dtype)
    for i in range(ts.shape[0]):
        out[i] = interp1d(x_old, ts[i], kind='linear')(x_new)
    return out


def preprocess_ecg(data: np.ndarray, fs_in: int = 500, fs_out: int = 5000) -> np.ndarray:
    """
    Standard ECGFounder preprocessing pipeline.
    data: (C, T) numpy array
    Returns: (C, 5000) numpy array
    """
    data = np.nan_to_num(data, nan=0.0)
    data = z_score(data)
    data = resample_unequal(data, fs_in, fs_out)
    return data


def read_wfdb(path: str):
    """Read WFDB record. Returns (data: np.ndarray (C, T), fs: int)."""
    signal, meta = wfdb.rdsamp(path)
    data = np.nan_to_num(signal, nan=0.0).T  # (T, C) -> (C, T)
    return data, meta['fs']


# ============================================================
# PTB-XL Pre-training Dataset
# ============================================================

class PTBXLPretrainDataset(Dataset):
    """
    PTB-XL pre-training dataset for SelfMIS.

    Each __getitem__ reads one 12-lead recording and returns:
      (lead_i, multi_lead): ((1, 5000), (12, 5000)) float32 tensors

    The same recording provides both the single-lead and multi-lead views —
    this is the natural positive pair for SigLIP contrastive pre-training.

    Args:
        ecg_path:    Root directory of PTB-XL (contains records500/).
        ptbxl_csv:   Path to ptbxl_database.csv.
        folds:       List of strat_fold values to include (e.g. [1..8] for train).
                     If None, all folds are used.
    """

    def __init__(self, ecg_path: str, ptbxl_csv: str, folds=None):
        df = pd.read_csv(ptbxl_csv)
        if folds is not None:
            df = df[df['strat_fold'].isin(folds)].reset_index(drop=True)
        self.records = df['filename_hr'].tolist()
        self.ecg_path = ecg_path

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path = os.path.join(self.ecg_path, self.records[idx])
        data, fs = read_wfdb(path)          # (12, T)
        data = preprocess_ecg(data, fs)     # (12, 5000)
        lead_i = data[0:1, :]               # (1, 5000) — Lead I is index 0 in PTB-XL
        return torch.FloatTensor(lead_i), torch.FloatTensor(data)


# ============================================================
# PTB-XL MI Fine-tuning Dataset
# ============================================================

class PTBXLMIDataset(Dataset):
    """
    PTB-XL fine-tuning dataset for 9-class MI detection (Lead I only).

    Labels are built from scp_codes column of ptbxl_database.csv.
    Non-MI recordings are included as all-zero label vectors (negatives).

    Args:
        ecg_path:    Root directory of PTB-XL.
        ptbxl_csv:   Path to ptbxl_database.csv.
        folds:       List of strat_fold values (train=[1..8], val=[9], test=[10]).
        threshold:   Minimum SCP confidence to count as positive (default 0.0).
        lead:        'single' → Lead I only (1, 5000); 'multi' → 12-lead (12, 5000).
    """

    def __init__(self, ecg_path: str, ptbxl_csv: str,
                 folds=None, threshold: float = 0.0, lead: str = 'single'):
        assert lead in ('single', 'multi'), "lead must be 'single' or 'multi'"
        df = pd.read_csv(ptbxl_csv)
        if folds is not None:
            df = df[df['strat_fold'].isin(folds)].reset_index(drop=True)
        self.records = df['filename_hr'].tolist()
        self.labels = self._build_labels(df, threshold)
        self.ecg_path = ecg_path
        self.lead = lead

    def _build_labels(self, df: pd.DataFrame, threshold: float) -> np.ndarray:
        """Returns (N, 9) binary float32 array."""
        labels = np.zeros((len(df), len(MI_CODES)), dtype=np.float32)
        for i, row in df.iterrows():
            try:
                scp = ast.literal_eval(row['scp_codes'])
                for j, code in enumerate(MI_CODES):
                    if code in scp and scp[code] > threshold:
                        labels[i, j] = 1.0
            except Exception:
                pass
        return labels

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path = os.path.join(self.ecg_path, self.records[idx])
        data, fs = read_wfdb(path)          # (12, T)
        data = preprocess_ecg(data, fs)     # (12, 5000)
        if self.lead == 'single':
            data = data[0:1, :]             # (1, 5000)
        label = torch.FloatTensor(self.labels[idx])
        return torch.FloatTensor(data), label

    def get_label_array(self) -> np.ndarray:
        """Returns the full (N, 9) label array, useful for computing pos_weight."""
        return self.labels


# ============================================================
# MIMIC-IV-ECG Pre-training Dataset (interface reserved)
# ============================================================

class MIMICECGPretrainDataset(Dataset):
    """
    MIMIC-IV-ECG pre-training dataset for SelfMIS.

    Directory structure expected:
        data_root/
            record_list.csv        (columns: subject_id, study_id, file_name, ecg_time, path)
            files/
                p1000/
                    p10000032/
                        s40689238/
                            40689238.hea
                            40689238.dat

    Each __getitem__ returns:
      (lead_i, multi_lead): ((1, 5000), (12, 5000)) float32 tensors

    Note: MIMIC stores leads as I, II, III, aVR, aVF, aVL, V1-V6 (aVF/aVL swapped
    vs standard ECGFounder order). We reorder to match ECGFounder convention.

    Args:
        data_root:       Root directory containing record_list.csv and files/.
        max_records:     If set, only use the first N records (useful for testing).
    """

    def __init__(self, data_root: str, max_records: int = None):
        csv_path = os.path.join(data_root, 'record_list.csv')
        df = pd.read_csv(csv_path)
        if max_records is not None:
            df = df.head(max_records)
        self.paths = df['path'].tolist()    # e.g. "files/p1000/p10000032/s40689238/40689238"
        self.data_root = data_root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        full_path = os.path.join(self.data_root, self.paths[idx])
        data, fs = read_wfdb(full_path)             # (12, T), MIMIC lead order
        data = data[MIMIC_TO_STANDARD, :]           # reorder to standard lead order
        data = preprocess_ecg(data, fs)             # (12, 5000)
        lead_i = data[0:1, :]                       # (1, 5000) — Lead I (index 0 after reorder)
        return torch.FloatTensor(lead_i), torch.FloatTensor(data)


# ============================================================
# Quick sanity check
# ============================================================

if __name__ == '__main__':
    PTB_ROOT = '/home/p00929643/ECG/data/physionet.org/files/ptb-xl/1.0.1/'
    PTB_CSV  = os.path.join(PTB_ROOT, 'ptbxl_database.csv')

    print('=== PTBXLPretrainDataset ===')
    ds = PTBXLPretrainDataset(PTB_ROOT, PTB_CSV, folds=[1, 2])
    print(f'  size: {len(ds)}')
    x_s, x_m = ds[0]
    print(f'  lead_i shape:  {x_s.shape}  dtype: {x_s.dtype}')
    print(f'  multi   shape: {x_m.shape}  dtype: {x_m.dtype}')
    assert x_s.shape == (1, 5000), f'expected (1,5000), got {x_s.shape}'
    assert x_m.shape == (12, 5000), f'expected (12,5000), got {x_m.shape}'

    print('\n=== PTBXLMIDataset (train) ===')
    mi = PTBXLMIDataset(PTB_ROOT, PTB_CSV, folds=list(range(1, 9)))
    print(f'  size: {len(mi)}')
    x, y = mi[0]
    print(f'  signal shape: {x.shape}, label shape: {y.shape}')
    labels = mi.get_label_array()
    print(f'  MI class positive counts: {labels.sum(axis=0).astype(int).tolist()}')
    print(f'  MI codes: {MI_CODES}')
    assert x.shape == (1, 5000)
    assert y.shape == (9,)

    print('\nAll assertions passed.')
