import json
j = json.load(open(r'D:\Documents\Autism_Project\final_optimization_experiments\comprehensive_research_evaluation.json'))
print('=== Efficiency ===')
for k, v in j['efficiency_results'].items():
    lat = v['latency_ms']
    fps = v['fps']
    par = v['params']
    print(f"{k}: Latency={lat:.2f} ms/clip | FPS={fps:.1f} | Params={par:,}")
print('\n=== Bootstrap 95% CI ===')
for k, v in j['bootstrap_stats'].items():
    am = v['acc_mean'] * 100
    alo, ahi = v['acc_ci95'][0] * 100, v['acc_ci95'][1] * 100
    mm = v['mf1_mean']
    mlo, mhi = v['mf1_ci95'][0], v['mf1_ci95'][1]
    print(f"{k}: Acc={am:.2f}% [{alo:.2f}%, {ahi:.2f}%] | MF1={mm:.4f} [{mlo:.4f}, {mhi:.4f}]")
print('\n=== Calibration ECE ===')
for k, v in j['calibration_data'].items():
    ece = v['ece']
    print(f"{k}: ECE={ece:.4f}")
