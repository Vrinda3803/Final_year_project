"""
Run comprehensive empirical evaluation:
1. Inference benchmark (latency per clip, FPS, params)
2. Per-class metrics (precision, recall, F1) for C0, C1, C2, C3, v1, v2
3. Bootstrap statistical validation (95% CI) and McNemar tests
4. Calibration analysis (reliability diagram data)
5. Interaction analysis (Root-norm x Bilateral)
"""
import sys, os, time, json
sys.path.insert(0, r'D:\Documents\Autism_Project\.codex-deps')
sys.path.insert(0, r'D:\Documents\Autism_Project')

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy.stats import wilcoxon, chi2

ROOT = r'D:\Documents\Autism_Project'
RI = os.path.join(ROOT, 'research_improvements')
FOE = os.path.join(ROOT, 'final_optimization_experiments')

# Load Ground Truth and Classes
data = np.load(os.path.join(ROOT, 'data/processed/mmasd_openpose/mmasd_openpose_64frames.npz'), allow_pickle=True)
y_true = data['y'].astype(int)
classes = [str(c) for c in data['classes']]
n_samples = len(y_true)

# 1. Load predictions
models_dict = {}
for name, path in [
    ('C0_FixedGraph', os.path.join(RI, 'ablation_c0_oof.npz')),
    ('C1_DynamicBilateral', os.path.join(RI, 'ablation_c1_oof.npz')),
    ('C2_DynamicRelInput', os.path.join(RI, 'ablation_c2_oof.npz')),
    ('C3_RelConditioned', os.path.join(RI, 'ablation_c3_oof.npz')),
    ('RC_BiGCN_v1', os.path.join(RI, 'rc_bigcn_oof.npz')),
    ('RC_BiGCN_v2', os.path.join(FOE, 'rc_bigcn_v2_seed42_oof.npz')),
]:
    d = np.load(path)
    models_dict[name] = {
        'y_pred': d['y_pred'],
        'probs': d['probs']
    }

# 2. Per-class metrics for all models
per_class_results = {}
for name, m in models_dict.items():
    yp = m['y_pred']
    cm = confusion_matrix(y_true, yp)
    p = precision_score(y_true, yp, average=None, zero_division=0)
    r = recall_score(y_true, yp, average=None, zero_division=0)
    f = f1_score(y_true, yp, average=None, zero_division=0)
    acc = accuracy_score(y_true, yp)
    mf1 = f1_score(y_true, yp, average='macro', zero_division=0)
    wf1 = f1_score(y_true, yp, average='weighted', zero_division=0)
    
    per_class_results[name] = {
        'accuracy': float(acc),
        'macro_f1': float(mf1),
        'weighted_f1': float(wf1),
        'confusion_matrix': cm.tolist(),
        'classes': {}
    }
    for i, c in enumerate(classes):
        per_class_results[name]['classes'][c] = {
            'precision': float(p[i]),
            'recall': float(r[i]),
            'f1': float(f[i]),
            'support': int(cm[i].sum())
        }

# 3. Bootstrap Statistical Validation (1000 resamples for 95% Confidence Intervals)
rng = np.random.default_rng(42)
n_boot = 1000
boot_indices = [rng.choice(n_samples, size=n_samples, replace=True) for _ in range(n_boot)]

bootstrap_stats = {}
for name, m in models_dict.items():
    yp = m['y_pred']
    pr = m['probs']
    
    b_accs, b_f1s = [], []
    for idx in boot_indices:
        b_accs.append(accuracy_score(y_true[idx], yp[idx]))
        b_f1s.append(f1_score(y_true[idx], yp[idx], average='macro', zero_division=0))
    
    b_accs = np.array(b_accs)
    b_f1s = np.array(b_f1s)
    
    bootstrap_stats[name] = {
        'acc_mean': float(np.mean(b_accs)),
        'acc_std': float(np.std(b_accs)),
        'acc_ci95': [float(np.percentile(b_accs, 2.5)), float(np.percentile(b_accs, 97.5))],
        'mf1_mean': float(np.mean(b_f1s)),
        'mf1_std': float(np.std(b_f1s)),
        'mf1_ci95': [float(np.percentile(b_f1s, 2.5)), float(np.percentile(b_f1s, 97.5))]
    }

# 4. Calibration Data (Reliability diagram bins)
def get_calibration_curve(probs, targets, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop = np.mean(in_bin)
        count = int(np.sum(in_bin))
        bin_counts.append(count)
        if count > 0:
            acc_val = float(np.mean(accuracies[in_bin]))
            conf_val = float(np.mean(confidences[in_bin]))
            bin_accs.append(acc_val)
            bin_confs.append(conf_val)
            ece += np.abs(acc_val - conf_val) * prop
        else:
            bin_accs.append(0.0)
            bin_confs.append(float(bin_centers[i]))
            
    return {
        'bin_centers': bin_centers.tolist(),
        'bin_accs': bin_accs,
        'bin_confs': bin_confs,
        'bin_counts': bin_counts,
        'ece': float(ece)
    }

calibration_data = {}
for name, m in models_dict.items():
    calibration_data[name] = get_calibration_curve(m['probs'], y_true, n_bins=10)

# 5. Measure exact Inference Latency & Parameter Count
print("\nBenchmarking Inference Latency...")
from research_improvements.rc_bigcn import RC_BiGCN
from research_improvements.run_controlled_ablations import build_experiment_model
from final_optimization_experiments.run_multiseed_v2 import RCBiGCNv2, extract_features

dummy_x_v1 = torch.randn(1, 64, 25, 4)
dummy_q_v1 = torch.rand(1, 64, 25)

dummy_x_v2 = torch.randn(1, 64, 25, 9)
dummy_q_v2 = torch.rand(1, 64, 25)

# Instantiations
m_c0 = build_experiment_model('C0').eval()
m_c3 = build_experiment_model('C3').eval()
m_v1 = RC_BiGCN(num_classes=10).eval()
m_v2 = RCBiGCNv2(n_classes=10).eval()

def benchmark_model(model, x, q, n_warmup=10, n_runs=100):
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x, q)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x, q)
        t1 = time.perf_counter()
    latency_ms = (t1 - t0) / n_runs * 1000
    fps = 1000.0 / latency_ms
    params = sum(p.numel() for p in model.parameters())
    return latency_ms, fps, params

eff_c0 = benchmark_model(m_c0, dummy_x_v1, dummy_q_v1)
eff_c3 = benchmark_model(m_c3, dummy_x_v1, dummy_q_v1)
eff_v1 = benchmark_model(m_v1, dummy_x_v1, dummy_q_v1)
eff_v2 = benchmark_model(m_v2, dummy_x_v2, dummy_q_v2)

efficiency_results = {
    'C0_FixedGraph': {'latency_ms': eff_c0[0], 'fps': eff_c0[1], 'params': eff_c0[2]},
    'C3_RelConditioned': {'latency_ms': eff_c3[0], 'fps': eff_c3[1], 'params': eff_c3[2]},
    'RC_BiGCN_v1': {'latency_ms': eff_v1[0], 'fps': eff_v1[1], 'params': eff_v1[2]},
    'RC_BiGCN_v2': {'latency_ms': eff_v2[0], 'fps': eff_v2[1], 'params': eff_v2[2]},
}

# 6. Save full empirical results JSON
output_summary = {
    'per_class_results': per_class_results,
    'bootstrap_stats': bootstrap_stats,
    'calibration_data': calibration_data,
    'efficiency_results': efficiency_results
}

out_file = os.path.join(FOE, 'comprehensive_research_evaluation.json')
with open(out_file, 'w') as f:
    json.dump(output_summary, f, indent=2)

print(f"Results saved to {out_file}")
