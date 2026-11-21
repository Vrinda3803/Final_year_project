# HYPERPARAMETER & OPTIMISATION SEARCH REPORT

**Date:** November 18, 2026  
**Search Scope:** RC-BiGCN Optimization on Grouped Validation Sets Only  

---

## 1. Controlled Hyperparameter Tuning Summary

All hyperparameters were tuned strictly using the inner validation splits (18% disjoint recording groups held out from the outer training set) to preserve test fold purity:

| Hyperparameter | Values Investigated | Selected Value | Justification / Impact |
|:---|:---|:---:|:---|
| **Base Channels ($C_0$)** | [16, 32, 64, 128] | **32** | 32 channels yielded 72K parameters, striking the optimal balance between representation capacity and overfitting avoidance on 1,100 training samples. 64 channels showed marginal train gain but worse val generalization. |
| **Learning Rate** | [3e-4, 8e-4, 1e-3, 2e-3] | **1e-3** | 1e-3 with Cosine Annealing demonstrated the fastest convergence to low validation loss without instability. |
| **Scheduler** | [StepLR, CosineAnnealing, ReduceLROnPlateau] | **CosineAnnealingLR** | Smoothly decayed learning rate to $10^{-5}$ across 65 epochs. |
| **Dropout Rate** | [0.1, 0.2, 0.3, 0.4, 0.5] | **0.3** | 0.3 spatial and temporal dropout provided the necessary regularization against camera noise. |
| **Batch Size** | [16, 32, 64] | **32** | Batch size 32 provided gradient stability while fitting easily within standard memory limits. |
| **Weight Decay** | [1e-5, 1e-4, 5e-4, 1e-3] | **1e-4** | Standard AdamW weight decay. |
| **Class Weighting** | [None, Inverse Freq, Effective Number ($\beta=0.99$)] | **Inverse Freq** | Mitigated the severe majority bias of class `dr` (25%), elevating minority class F1 (`mfs` from 0.33 to 0.54). |
| **TCN Kernel Sizes**| [3, [3, 5], [3, 5, 7]] | **[3, 5, 7] with dilations [1, 2]**| Multi-scale receptive fields captured both rapid flapping/clapping (small kernel) and sustained torso rotation/swaying (large dilated kernel). |
