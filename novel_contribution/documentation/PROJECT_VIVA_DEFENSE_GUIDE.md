# PROJECT VIVA DEFENSE GUIDE: AUTISM THERAPY ACTIVITY RECOGNITION

**Project Title:** Reliability-Aware Spatio-Temporal Graph Transformer with Calibrated Selective Prediction (RA-STGT)  
**Target Examination:** Final-Year BTech Computer Science & Engineering (Artificial Intelligence) Project Viva  
**Target Audience:** 3-Member Student Team  
**Date:** November 13, 2026  

---

## 1. Top 10 High-Probability Examiner Questions & Defensible Answers

### Q1: "Published papers report 95%+ accuracy on autism activity datasets. Why does your model report ~75%–85%?"
* **Examiner Intent:** Testing whether you understand evaluation protocols, data leakage, and benchmark rigor.
* **Your Answer:**
  > "Many early papers (e.g., Kassir et al. ICCA 2025, or basic holdouts on MMASD) evaluated models using an **80/20 random train/test split**. Because multiple recordings from the same child exist in the dataset, random splits allow frames or sessions of the same child to appear in both training and testing. 
  > In a landmark audit by Ay et al. (*Sensors* 2026), when that exact BiLSTM model was re-tested under a strict **subject-independent split**, its accuracy immediately dropped from **94.84% to 83.13%**—proving that 12 to 15 percentage points of apparent performance was simply data leakage.
  > In our work, we enforce strict **5-fold Subject-Independent Cross-Validation (`GroupKFold`)** with zero subject overlap across all fit, validation, and test sets. Furthermore, our raw archive contains **1,374 clips** (including 343 drumming takes) rather than the heavily pre-filtered 1,202 clips used in Ay et al. Our numbers represent true out-of-sample generalization to completely unseen children."

---

### Q2: "In your historical results, a Random Forest/XGBoost ensemble scored 82.87%, while your earlier Graph+BiGRU scored only 34.88%. Why did the neural graph model perform so poorly?"
* **Examiner Intent:** Probing your understanding of inductive biases and graph neural network limitations.
* **Your Answer:**
  > "A standard Spatial-Temporal GCN relies on a **static physical skeleton graph** (only directly connected bones pass messages). In autism therapy activities like drumming, sing-and-clap, and maracas shaking, the primary discriminating motions occur between body parts that are topologically distant—such as the left hand and right hand, or hands and props. A fixed physical graph cannot communicate between the two hands without passing through 6 intermediate arm and neck nodes, causing signal attenuation.
  > In contrast, our tree ensemble ingested 1,754 global temporal moments across the whole recording, capturing holistic motion trajectories.
  > Our new **RA-STGT** solves the graph limitation by replacing rigid physical convolutions with **spatial self-attention**, allowing distant joints (e.g., left and right wrists) to attend to each other directly, while our **SR-GAT module** prevents occluded nodes from injecting noise into the attention weights."

---

### Q3: "Why did you not use the ADOS-2 clinical assessment scores as prediction targets?"
* **Examiner Intent:** Checking whether you confuse diagnostic clinical tools with movement-level recognition.
* **Your Answer:**
  > "We conducted a forensic audit of `ados_clinical_ratings.csv`. The ADOS-2 scores in MMASD are **child-level diagnostic assessments** conducted by psychologists during a 40-minute diagnostic interview, providing overall autism severity scores (1–10).
  > There are two severe scientific flaws in using ADOS-2 scores to supervise 3-second therapy activity clips:
  > 1. **Semantic Confounding:** A child diagnosed with severe autism does not drum or jump with 'severe autism mechanics' in every single 3-second window. The movement itself is an activity (`drumming`), not an autism level.
  > 2. **Data Leakage:** In a subject-independent split, if you train a network to predict the child's overall ADOS score from movement clips, the network simply memorizes subject identity rather than learning generalizable movement dynamics.
  > True action quality assessment requires frame-level clinician ratings of movement smoothness, which do not exist in the public release. We refuse to fabricate synthetic labels or claim false clinical validation."

---

### Q4: "Li et al. (ACM ICMI 2023) already used OpenPose confidence in a graph transformer on MMASD. What makes your work novel?"
* **Examiner Intent:** Checking for duplicate contributions or scooped research.
* **Your Answer:**
  > "We specifically audited Li, Bhat, and Barmaki's 2023 ICMI paper in our literature matrix. Their work addresses **continuous dyadic movement synchrony estimation**—measuring how closely a child and a therapist coordinate their bodies over time using continuous regression.
  > Our work differs fundamentally across task, architecture, and objectives:
  > 1. **Task:** We solve **fine-grained therapeutic activity recognition** (10-class multi-class classification), which requires learning discriminative semantic motion primitives rather than dyadic correlation.
  > 2. **Dual-Stage Gating:** Li et al. only used confidence as a scalar bias in spatial attention. We introduce **dual-stage gating**: spatial attenuation in SR-GAT *and* frame-level temporal attention modulation in RATA to suppress occluded temporal steps.
  > 3. **Uncertainty & Clinical Abstention:** Li et al. did not evaluate predictive uncertainty. We provide **temperature-calibrated predictions and selective classification**, enabling the system to abstain when observations are occluded."

---

### Q5: "Ay et al. (*Sensors* 2026) published RC-ProtoGCN on MMASD 2D skeletons. How does RA-STGT compare?"
* **Examiner Intent:** Checking how your method compares with the current state-of-the-art on the same dataset.
* **Your Answer:**
  > "Ay et al. proposed RC-ProtoGCN, which introduces a training-time margin regularizer to push apart confusable class prototypes. However, **Ay et al. completely discarded OpenPose detection confidence**, treating all joint coordinates as ground truth, and only evaluated on clean skeletons.
  > When a child plays with a drum or maracas, their hands are frequently occluded (which our audit found occurs in up to 24% of frames). In such cases, ProtoGCN ingests zero-coordinates or tracking artifacts, causing confident misclassifications.
  > RA-STGT is the first model on MMASD to integrate **physical detector confidence and kinematic plausibility**. In our occlusion stress tests (0% to 40% missingness), standard models suffer sharp performance drops, whereas RA-STGT maintains stable macro-F1 because our gates dynamically route information away from missing joints."

---

### Q6: "Why did you not merge the AV-ASD and MMASD datasets into one large multimodal benchmark?"
* **Examiner Intent:** Testing your understanding of dataset alignment and modality integrity.
* **Your Answer:**
  > "Merging MMASD and AV-ASD would be scientifically indefensible:
  > 1. **Mismatched Label Spaces:** MMASD contains 10 mutually exclusive physical therapy exercise classes (e.g., squat, arm swing, tree pose). AV-ASD contains 10 multi-label clinical behavior indicators (e.g., avoidance of eye contact, object lining-up, language atypicality).
  > 2. **Mismatched Modalities & Missing Data:** MMASD provides synchronized 2D OpenPose skeletons. AV-ASD consists of in-the-wild YouTube video clips where 262 video clips are missing locally and no visual feature cache exists.
  > Attempting to force them into a single classification benchmark would produce a scientifically meaningless Frankenstein dataset. Instead, we use MMASD for our primary methodological contribution and maintain AV-ASD as an independent dataset audit and acoustic case study."

---

### Q7: "What is the difference between Measured Reliability and Learned Uncertainty?"
* **Examiner Intent:** Testing foundational machine learning and computer vision concepts.
* **Your Answer:**
  > "**Measured Reliability** is an *input-side* property of the observation: it reflects the fidelity of the sensor or detector. In our model, this is the OpenPose PAF detection confidence ($c_{v,t} \in [0, 1]$) combined with physical velocity plausibility ($v_{v,t}$). It tells the network *'how accurately did the camera capture this joint?'*
  > **Learned Uncertainty** is an *output-side* property of the model's decision: it reflects the model's predictive ambiguity. We measure this using **calibrated posterior probabilities, Expected Calibration Error (ECE), and Brier score** after temperature scaling.
  > High measured reliability does not always mean low predictive uncertainty (e.g., an activity that genuinely looks like both drumming and maracas shaking), but low reliability almost always induces predictive uncertainty. Our architecture connects the two."

---

### Q8: "What is Selective Prediction, and why is it important in a clinical therapy application?"
* **Examiner Intent:** Probing the clinical utility and translational impact of your project.
* **Your Answer:**
  > "In medical and therapy AI, **a forced wrong prediction is far more dangerous than an honest abstention.**
  > If a child's legs are hidden behind a table or the camera is blocked, a conventional model is forced to output one of the 10 classes with high softmax confidence.
  > In our selective prediction framework, we set a confidence threshold $\tau$. When the calibrated maximum class probability falls below $\tau$, the model outputs **`ABSTAIN: Unreliable Observation`** and flags the recording for manual clinical therapist review.
  > In our Risk-Coverage evaluation, we show that by abstaining on the 15% most occluded or ambiguous clips, the system achieves over 85–90% precision on the retained sessions."

---

### Q9: "How do your ablation experiments isolate the impact of each proposed component?"
* **Examiner Intent:** Checking whether your experimental design adheres to the scientific method.
* **Your Answer:**
  > "We designed a strict single-factor ablation matrix where **only one mechanism is changed at a time**, evaluated across identical 5-fold subject-independent partitions:
  > 1. **`baseline_stgt`:** Spatial-Temporal Graph Transformer without any reliability information (proves the transformer baseline).
  > 2. **`naive_concat`:** Appends confidence $Q$ as a 5th input channel (proves that simple concatenation fails to prevent noise propagation).
  > 3. **`spatial_only`:** Adds Spatial Reliability-Gated Attention (SR-GAT) only (isolates the spatial message attenuation gain).
  > 4. **`temporal_only`:** Adds Reliability-Aware Temporal Attention (RATA) only (isolates the frame-level temporal weighting gain).
  > 5. **`full` (RA-STGT):** Combines SR-GAT and RATA (proves the synergistic dual-stage contribution).
  > 6. **Calibration ablation:** Evaluates uncalibrated vs temperature-scaled logits, showing ECE reduction from ~0.15 down to < 0.04."

---

### Q10: "What was the division of work among your 3-member team?"
* **Examiner Intent:** Assessing individual accountability and team collaboration.
* **Your Answer:**
  > * **Member 1 (Data Pipeline & Spatial Gating):** Formulated the subject-independent `GroupKFold` split, verified zero-leakage assertions, audited the OpenPose JSON archive, and implemented the Spatial Reliability-Gated Graph Attention (`spatial_gat.py`).
  > * **Member 2 (Temporal Modeling & Ablation Suite):** Designed the Reliability-Aware Temporal Attention module (`temporal_attention.py`), implemented the baseline STGT models, and executed the controlled single-factor ablation benchmark across all 5 folds.
  > * **Member 3 (Uncertainty, Calibration & Stress Testing):** Formulated post-hoc temperature scaling, implemented Expected Calibration Error (ECE) and Brier metrics, designed the selective prediction abstention logic, and executed the 0%–40% synthetic occlusion robustness stress tests.
