"""
PTB-XL 150-class evaluation script with corrected paths.
Based on ptbxl_eval.py with paths adapted for our environment.
"""
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
from shutil import copyfile
import pickle
import time
import wfdb
import ast
from scipy import signal
import json
from net1d import Net1D
from util import eval_with_dynamic_thresh
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score
from scipy.stats import bootstrap
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from scipy.interpolate import interp1d

class PTBXL_Dataset(torch.utils.data.Dataset):
    def __init__(self, ecg_path, csv_path):
        self.data = pd.read_csv(csv_path)
        self.data = self.data.dropna(subset=['filename_hr', 'label'])
        self.fs = 5000
        self.ecg_path = ecg_path

    def z_score_normalization(self, signal):
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    def resample_unequal(self, ts, fs_in, fs_out):
        if fs_in == 0 or len(ts) == 0:
            return ts
        t = ts.shape[1] / fs_in
        fs_in, fs_out = int(fs_in), int(fs_out)
        if fs_out == fs_in:
            return ts
        if 2 * fs_out == fs_in:
            return ts[:, ::2]
        resampled_ts = np.zeros((ts.shape[0], fs_out))
        x_old = np.linspace(0, t, num=ts.shape[1], endpoint=True)
        x_new = np.linspace(0, t, num=int(fs_out), endpoint=True)
        for i in range(ts.shape[0]):
            y_old = ts[i, :]
            f = interp1d(x_old, y_old, kind='linear')
            resampled_ts[i, :] = f(x_new)
        return resampled_ts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        hash_file_name = row['filename_hr']
        label = row['label']
        label = json.loads(label)
        label = torch.tensor(label, dtype=torch.float)

        sample_rate = 500
        data = [wfdb.rdsamp(self.ecg_path + hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = data.squeeze(0)
        data = np.transpose(data, (1, 0))
        data = self.z_score_normalization(data)
        signal = self.resample_unequal(data, sample_rate, self.fs)
        signal = torch.FloatTensor(signal)
        return signal, label


# ==================== Configuration ====================
saved_dir = './res/eval'
csv_filepath = './csv/ptbxl_label.csv'
ecg_filepath = '/home/p00929643/ECG/data/physionet.org/files/ptb-xl/1.0.1/'
checkpoint_path = './checkpoint/12_lead_ECGFounder.pth'

tasks = []
batch_size = 512
with open(os.path.join('./tasks.txt'), 'r') as fin:
    for line in fin:
        tasks.append(line.strip())

print(f"Number of tasks: {len(tasks)}")
print(f"ECG data path: {ecg_filepath}")
print(f"Checkpoint: {checkpoint_path}")

# ==================== Dataset ====================
testset = PTBXL_Dataset(ecg_path=ecg_filepath, csv_path=csv_filepath)
print(f"Test set size: {len(testset)}")
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=min(os.cpu_count(), 8))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ==================== Model ====================
model = Net1D(
    in_channels=12,
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
    n_classes=150)

model.to(device)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
state_dict = checkpoint['state_dict']
log = model.load_state_dict(state_dict, strict=False)
print(f"Model loaded. Missing keys: {log.missing_keys}, Unexpected keys: {log.unexpected_keys}")

for name, param in model.named_parameters():
    param.requires_grad = False

model.to(device)
model.eval()

# ==================== Inference ====================
print("\n=== Running inference ===")
prog_iter_test = tqdm(testloader, desc="Testing", leave=True)
all_gt = []
all_pred_prob = []

with torch.no_grad():
    for batch_idx, batch in enumerate(prog_iter_test):
        input_x, input_y = tuple(t.to(device) for t in batch)
        logits = model(input_x)
        pred = torch.sigmoid(logits)
        all_pred_prob.append(pred.cpu().data.numpy())
        all_gt.append(input_y.cpu().data.numpy())

all_pred_prob = np.concatenate(all_pred_prob)
all_gt = np.concatenate(all_gt)
print(f"Predictions shape: {all_pred_prob.shape}, GT shape: {all_gt.shape}")

# Save raw predictions and ground truth
df_gt = pd.DataFrame(all_gt)
df_gt.to_csv(os.path.join(saved_dir, 'all_gt.csv'), index=False, float_format='%.5f')
df_pred = pd.DataFrame(all_pred_prob)
df_pred.to_csv(os.path.join(saved_dir, 'all_pred.csv'), index=False, float_format='%.5f')

# ==================== Evaluation with dynamic thresholds ====================
print("\n=== Evaluating with dynamic thresholds ===")
res_test, res_test_auroc, res_test_sens, res_test_spec, res_test_f1, optimal_thresholds = eval_with_dynamic_thresh(all_gt, all_pred_prob)

all_thre_df = []
for i, task in enumerate(tasks):
    all_thre_df.append([task, res_test_auroc[i], res_test_sens[i], res_test_spec[i], res_test_f1[i], optimal_thresholds[i]])

columns = ['Field_ID', 'AUROC', 'sensitivity', 'specificity', 'f1', 'optimal_thresholds']
all_thre_df = pd.DataFrame(all_thre_df, columns=columns)
all_thre_df.to_csv(os.path.join(saved_dir, 'res_thre.csv'), index=False, float_format='%.5f')

# ==================== Bootstrap CI ====================
print("\n=== Computing bootstrap confidence intervals ===")
df_gt = pd.read_csv(os.path.join(saved_dir, 'all_gt.csv'))
df_pred = pd.read_csv(os.path.join(saved_dir, 'all_pred.csv'))

all_gt_df = df_gt.T
all_pred_df = df_pred.T

all_thre_df = pd.read_csv(os.path.join(saved_dir, 'res_thre.csv'))
all_thre_df = all_thre_df.iloc[:, -1].T

all_pred_df.index = all_pred_df.index.astype(int)
all_gt_df.index = all_gt_df.index.astype(int)


def calculate_performance_metrics(true, pred, threshold):
    true = np.array(true)
    pred = np.array(pred)
    pred_binary = (pred >= threshold).astype(int)

    true_positive = np.sum((true == 1) & (pred_binary == 1))
    false_positive = np.sum((true == 0) & (pred_binary == 1))
    true_negative = np.sum((true == 0) & (pred_binary == 0))
    false_negative = np.sum((true == 1) & (pred_binary == 0))

    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    ppv = precision
    npv = true_negative / (true_negative + false_negative) if (true_negative + false_negative) > 0 else 0

    if (precision + sensitivity) > 0:
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1_score = 0

    if len(np.unique(true)) == 1:
        auroc = np.nan
        auprc = np.nan
    else:
        auroc = roc_auc_score(true, pred)
        auprc = average_precision_score(true, pred)

    return sensitivity, specificity, precision, f1_score, ppv, npv, auroc, auprc


def bootstrap_ci(metric_func, true, pred, threshold, n_resamples=10):
    true = np.array(true)
    pred = np.array(pred)

    bootstrap_distribution = []
    for _ in range(n_resamples):
        indices = np.random.choice(len(true), len(true), replace=True)
        resampled_true = true[indices]
        resampled_pred = pred[indices]
        metric_value = metric_func(resampled_true, resampled_pred, threshold)
        bootstrap_distribution.append(metric_value)

    lower_bound = np.percentile(bootstrap_distribution, 2.5)
    upper_bound = np.percentile(bootstrap_distribution, 97.5)
    return (round(lower_bound, 3), round(upper_bound, 3))


# Calculate metrics and their 95% CI
results = []
for i, task in enumerate(tqdm(tasks, desc="Computing CIs")):
    true = all_gt_df.loc[i]
    pred = all_pred_df.loc[i]
    threshold = all_thre_df.loc[i]
    sens, spec, prec, f1, ppv, npv, auroc, auprc = calculate_performance_metrics(true, pred, threshold)

    sens_ci = bootstrap_ci(lambda true, pred, threshold: calculate_performance_metrics(true, pred, threshold)[0], true, pred, threshold)
    spec_ci = bootstrap_ci(lambda true, pred, threshold: calculate_performance_metrics(true, pred, threshold)[1], true, pred, threshold)
    f1_ci = bootstrap_ci(lambda true, pred, threshold: calculate_performance_metrics(true, pred, threshold)[3], true, pred, threshold)
    ppv_ci = bootstrap_ci(lambda true, pred, threshold: calculate_performance_metrics(true, pred, threshold)[4], true, pred, threshold)
    npv_ci = bootstrap_ci(lambda true, pred, threshold: calculate_performance_metrics(true, pred, threshold)[5], true, pred, threshold)
    auroc_ci = bootstrap_ci(lambda true, pred, threshold: calculate_performance_metrics(true, pred, threshold)[6], true, pred, threshold)
    auprc_ci = bootstrap_ci(lambda true, pred, threshold: calculate_performance_metrics(true, pred, threshold)[7], true, pred, threshold)

    results.append({
        'Label': task,
        'Sensitivity': round(sens, 3), 'Sensitivity_CI': sens_ci,
        'Specificity': round(spec, 3), 'Specificity_CI': spec_ci,
        'F1': round(f1, 3), 'F1_CI': f1_ci,
        'PPV': round(ppv, 3), 'PPV_CI': ppv_ci,
        'NPV': round(npv, 3), 'NPV_CI': npv_ci,
        'AUROC': round(auroc, 3) if not np.isnan(auroc) else np.nan, 'AUROC_CI': auroc_ci,
        'AUPRC': round(auprc, 3) if not np.isnan(auprc) else np.nan, 'AUPRC_CI': auprc_ci
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(saved_dir, 'res.csv'), index=False)

# ==================== Print summary ====================
print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
print(f"Mean AUROC (dynamic threshold): {res_test:.4f}")
print(f"Mean Sensitivity: {np.mean(res_test_sens):.4f}")
print(f"Mean Specificity: {np.mean(res_test_spec):.4f}")
print(f"Mean F1: {np.mean(res_test_f1):.4f}")
print(f"\nResults saved to: {saved_dir}/")
print(f"  - res.csv (per-task metrics with 95% CI)")
print(f"  - res_thre.csv (metrics with optimal thresholds)")
print(f"  - all_gt.csv, all_pred.csv (raw predictions)")

# Print top-10 tasks by AUROC
print("\n--- Top 10 tasks by AUROC ---")
sorted_df = results_df.dropna(subset=['AUROC']).sort_values('AUROC', ascending=False).head(10)
for _, row in sorted_df.iterrows():
    print(f"  {row['Label']:40s}  AUROC={row['AUROC']:.3f}  F1={row['F1']:.3f}  Sens={row['Sensitivity']:.3f}  Spec={row['Specificity']:.3f}")
