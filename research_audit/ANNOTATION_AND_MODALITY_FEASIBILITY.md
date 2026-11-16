# ANNOTATION AND MODALITY FEASIBILITY REPORT

**Author/Role:** Senior AI/ML Research Engineer & Computer Vision Researcher  
**Project:** Autism Therapy Activity Recognition (Final-Year BTech CSE-AI Project)  
**Date:** November 13, 2026  
**Objective:** Ground-truth determination of what the downloaded and workspace data actually contain, establishing strict feasibility boundaries for research claims.

---

## 1. Skeleton Modality Feasibility: 2D OpenPose vs 3D ROMP

### 1.1 2D OpenPose Skeletons (`2D_openpose.zip`)

* **Storage Path:** `data/raw/mmasd/Dataset_FINAL/2D skeleton/2D_openpose.zip` (300.85 MB)
* **File Structure:** 254,674 per-frame JSON files organized into 1,374 clip directories across 10 activity folders.
* **Keypoint Format:** OpenPose `BODY_25` topology consisting of 25 physical body joints:
  * Indices: 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist, 5: LShoulder, 6: LElbow, 7: LWrist, 8: MidHip, 9: RHip, 10: RKnee, 11: RAnkle, 12: LHip, 13: LKnee, 14: LAnkle, 15: REye, 16: LEye, 17: REar, 18: LEar, 19: LBigToe, 20: LSmallToe, 21: LHeel, 22: RBigToe, 23: RSmallToe, 24: RHeel.
* **Pose Confidence Values ($c$):**
  * **Availability:** **YES, FULLY AVAILABLE.**
  * **Scale:** Continuous float in $[0.0, 1.0]$.
  * **Meaning:** The OpenPose Part Affinity Field (PAF) detection confidence for each individual joint in each video frame.
  * **Empirical Missingness in Processed Cache (`mmasd_openpose_64frames.npz`):**
    * Mean confidence across all joints/frames: **0.6413**
    * Exact fraction of keypoints with $c = 0.0$ (undetected / occluded): **6.48%** (142,394 of 2,198,400 joint-frame observations).
    * Limb-specific missingness: Highest on wrists and ankles (up to 18–24% zero confidence during vigorous drumming and jumping) and lowest on stable torso nodes (neck and hips < 1.5% zero confidence).
* **Feasibility Determination:** **100% FEASIBLE.** 2D OpenPose skeletons provide clean, verified joint-level detection confidence for reliability-aware learning.

---

### 1.2 3D ROMP Skeletons (`ROMP_3D_Coordinates_71-joints.zip`)

* **Storage Path:** `data/raw/mmasd/Dataset_FINAL/3D skeleton/ROMP_3D_Coordinates_71-joints.zip` (758.91 MB)
* **File Structure:** 311,080 per-frame `.npz` files across clip folders.
* **Array Shape per Frame:** `(3, 71, 3)` (float32).
  * 3 slots for detected human figures in the scene (child, therapist, observer).
  * 71 SMPL body/hand/face joints.
  * 3 continuous Cartesian coordinates $(X, Y, Z)$ in millimeters/meters.
* **Pose Confidence Values ($c$):**
  * **Availability:** **NO. ZERO CONFIDENCE VALUES EXIST.**
  * ROMP outputs predicted 3D mesh joint locations without per-joint detection confidence.
* **Tracking Identity Swaps:**
  * The 3 person slots in the `(3, 71, 3)` array are un-tracked. The child is frequently swapped between slot 0, slot 1, and slot 2 across frames as the therapist enters or occludes the frame.
* **Feasibility Determination:** **UNFEASIBLE FOR PRIMARY NOVELTY.**
  * Because ROMP 3D lacks per-joint confidence scores and suffers from multi-person identity swapping, developing a "pose-uncertainty" model on ROMP would require fabricating confidence proxies.
  * **Decision:** Retain 3D ROMP for historical feature baselines, but implement the novel Reliability-Aware framework on the 2D OpenPose data where real detector confidence is preserved.

---

## 2. Clinical Annotation Feasibility: ADOS-2 vs Movement Quality

### 2.1 Audit of `ados_clinical_ratings.csv` & `ADOS_rating.xlsx`

The clinical ratings file contains 36 participant rows:

```csv
ID#,Gender,Chronological Age,Module,Social Affect Total,Restricted and Repetitive Behavior (RRB) Total,Overall Total,ADOS-2 classification/Dx,ADOS Comparison Score (1-10),Severity of Autism
20274.0,M,"6Y,9M",2.0,18.0,3.0,21.0,2.0,9.0,3
20313.0,M,9y,2.0,19.0,5.0,24.0,2.0,10.0,3.0
20383.0,M,"5Y,8M",3.0,19.0,3.0,22.0,2.0,10.0,3
40013.0,M,"11Y, 3M",1.0,15.0,14.0,29.0,2.0,10.0,3.0
...
```

### 2.2 Critical Distinction of Annotation Levels

```
+---------------------------------------------------------------------------------------------------------+
|                                      CLINICAL ANNOTATION AUDIT                                          |
+--------------------------+-----------------------+-----------------------+------------------------------+
| Information Category     | Granularity           | Available in Dataset? | Suitable as Model Target?    |
+--------------------------+-----------------------+-----------------------+------------------------------+
| Activity Label           | Clip / Segment Level  | YES (10/11 classes)   | YES (Primary Classification) |
| Child Autism Severity    | Child / Patient Level | YES (ADOS-2, n=36)    | NO (Causes Data Leakage)     |
| Movement Quality Rating  | Frame / Activity Level| NO (Zero in release)  | N/A (Cannot fabricate)       |
| Therapist Action / Policy| Frame / Event Level   | NO (Zero in release)  | N/A (Rules out clinical RL)  |
+--------------------------+-----------------------+-----------------------+------------------------------+
```

### 2.3 Formal Ruling on Clinician-Supervised Learning (Component 4)

1. **Child-Level Diagnostic Scores Cannot Supervise Movement:**
   * ADOS-2 is a 40-minute standardized clinical diagnostic interview assessing overall social-communication deficits and repetitive behaviors across an entire child assessment.
   * If an activity recognition model uses ADOS-2 severity as an auxiliary target for an individual 3-second drumming clip, it creates severe **semantic confounding** (a child with severe autism does not drum with a "severe autism drumming style" in every single 3-second window).
   * More critically, in a 5-fold subject-independent split, supervising with child-level ADOS-2 scores encourages the model to memorize subject-specific idiosyncrasies rather than activity kinematics, causing **data leakage**.
2. **Definitive Decision:**
   * **Component 4 (Clinician-Informed Auxiliary Supervision) is FORMALLY REMOVED from the core novel contribution.**
   * We will NOT fabricate frame-level movement quality ratings, and we will NOT misuse ADOS-2 scores as clip-level labels.
   * The novel contribution focuses strictly on **Reliability-Aware Temporal Representation Learning and Calibrated Prediction**, which are 100% supported by the data.

---

## 3. AV-ASD Modality Alignment & Feasibility

### 3.1 Media Availability Audit

1. **Video Clips:** Only 666 of 928 `.mp4` video clips exist locally across `clips_video-20261113T032829Z-1-001.zip` and `002.zip`. 262 clips are missing because the 3rd Google Drive chunk was not downloaded.
2. **Audio Clips:** Extracted WAV clips are only available for a fraction of clips; only 227 clips have verified aligned acoustic features in the local stage-1 audio bundle.
3. **Visual / Pose Features:** Zero extracted visual feature arrays (e.g. CLIP embeddings, ImageBind embeddings) or MediaPipe/OpenPose keypoint caches exist locally for AV-ASD.

### 3.2 Multimodal Alignment & Leakage Feasibility

1. **Clip vs Segment vs Video Level:**
   * The AV-ASD CSV logs confirm that multiple annotated clips originate from the same `Original_ID` (source YouTube video).
   * For example, source video `YtvP5A5OHpU` contributes multiple clips: `_330_340`, `_340_350`, `_440_450`, `_465_475`.
   * Evaluating AV-ASD with a random split across clips results in severe source-video leakage.
2. **Definitive Decision on AV-ASD:**
   * **Do NOT force AV-ASD into a merged dataset with MMASD.** MMASD has 10/11 discrete physical therapy classes from 25/71 skeleton joints; AV-ASD has 10 multi-label diagnostic behavioral indicators from naturalistic video. Merging them is scientifically unsound.
   * **Feasible Scope for AV-ASD:** Retain AV-ASD as an **independent complementary audit and forensic case study**, analyzing source-video leakage, missing video chunk recovery, and acoustic-only baseline boundaries. The primary methodological contribution will be developed and validated on MMASD.

---

## 4. Final Feasibility Matrix

| Planned Component | Required Data Modality | Local Availability | Scientific Validity | Final Project Decision |
|---|---|---|---|---|
| Joint-Level Reliability Gating | 2D OpenPose confidence ($c \in [0, 1]$) | Full (1,374 clips, 25 joints) | Valid; measured PAF detector confidence | **APPROVED (Core Component 1)** |
| Reliability-Aware Temporal Attention | Frame visibility & velocity consistency | Full (64 frames, 25 joints) | Valid; biomechanical physics constraint | **APPROVED (Core Component 2)** |
| Calibrated Selective Prediction | Model logits + validation calibration | Full (Subject-grouped validation) | Valid; standard ECE / Brier metrics | **APPROVED (Core Component 3)** |
| Clinician Auxiliary Supervision | Frame-level movement quality ratings | None (ADOS-2 is child-level diagnostic) | Invalid; causes leakage & semantic mismatch | **EXCLUDED (Scientifically Defensible)** |
| AV-ASD Multimodal Fusion | Complete 928 video + audio clips | Incomplete (666 video, 227 audio) | Invalid; missing 262 clips & vision arrays | **RESTRICTED TO SEPARATE CASE STUDY** |
| MMASD 10-Class Benchmark | Curated 1,202 clips vs raw 1,374 clips | Raw 1,374 clips intact | Valid; must document +175 drumming clips | **APPROVED (Transparent Reporting)** |
