# COMPLEMENTARY RESEARCH CASE STUDY: AV-ASD DATASET AUDIT & BENCHMARK ANALYSIS

**Dataset:** Audio-Visual Autism Spectrum Disorder (AV-ASD)  
**Authors of Benchmark Paper:** Deng et al. (*IEEE Transactions on Multimedia*, 2025; arXiv:2406.02554)  
**Role:** Senior AI/ML Research Engineer & Academic Research Assistant  
**Project:** Final-Year BTech CSE (AI) Research Project  
**Date:** November 13, 2026  

---

## 1. Executive Summary & Purpose

The AV-ASD dataset represents a pioneering effort by Deng et al. to introduce in-the-wild video benchmarks for autism-associated social and motor behavior recognition. 

This document serves as an **independent forensic case study** evaluating the feasibility, reproducibility, and methodological boundaries of the AV-ASD dataset as stored in the local project workspace. It explains:
1. Why AV-ASD is evaluated as a separate complementary study rather than merged with MMASD.
2. The empirical media gap (262 missing video clips in local downloads).
3. The acoustic-only baseline boundaries.
4. The critical issue of source-video leakage in clip-level splits.

---

## 2. Dataset Composition & Modalities

### 2.1 The Published Benchmark (Deng et al. 2025)
* **Scope:** 928 annotated video clips extracted from 569 source YouTube videos.
* **Duration:** Clip lengths range from 1 to 61 seconds (median: ~8 seconds).
* **Target Task:** 10 multi-label behavioral indicators:
  1. Absence or Avoidance of Eye Contact
  2. Aggressive Behavior
  3. Hyper- or Hyporeactivity to Sensory Input
  4. Non-Responsiveness to Verbal Interaction
  5. Non-Typical Language
  6. Object Lining-Up
  7. Self-Hitting or Self-Injurious Behavior
  8. Self-Spinning or Spinning Objects
  9. Upper Limb Stereotypies
  10. Background (Control / Non-atypical)
* **Published State of the Art:**
  * LLaVA-ASD (Fine-tuned Vision + Speech): **59.77% Test Macro-F1**
  * ImageBind + Whisper Temporal Fusion (Audio + Speech): **53.17% Test Macro-F1**
  * ImageBind Video Temporal: 50.87%
  * CLIP + MLP (Vision only): 45.72%

---

## 3. Local Forensic Audit Findings

### 3.1 Media Availability & Missing Video Archive
* The local download directory `D:\Downloads` contains:
  * `clips_video-20261113T032829Z-1-001.zip` (333 `.mp4` video clips, 1.91 GB)
  * `clips_video-20261113T032829Z-1-002.zip` (333 `.mp4` video clips, 1.96 GB)
* **Total Local Video Clips:** **666 clips.**
* **Total Required by `dataset.csv`:** **928 clips.**
* **Deficit:** **262 video clips are missing locally.** The raw video release was chunked into 2GB zip volumes on Google Drive; the 3rd volume containing the remaining 262 clips was not completed.
* Furthermore, `csvs/recovery_attempts.csv` records:
  * `recovered`: 359
  * `download_failed`: 184
  * `bad_timestamp`: 2
  Confirming that 184 source YouTube videos were permanently taken down or privated upstream.

### 3.2 Modality Features & Alignment Audit
* The stage-1 audio bundle (`AV_ASD_audio_stage1_audit_bundle.zip`) provides 5,410 segment acoustic descriptors (32-dim OpenSMILE features).
* When aligned with the official 928 annotated clip boundaries, **only 227 clips possessed valid, continuous acoustic features.**
* **Zero local visual feature arrays exist:** There are no pre-extracted CLIP embeddings, ImageBind visual features, or MediaPipe skeletal keypoint arrays in the local project directory.
* **Historical Correction:** An earlier experimental script (`experiments/avasd_multimodal_gated_fusion.py`) claimed "gated multimodal fusion", but an audit of the source code confirms it simply weighted audio probabilities with class-prior offsets. **No video frames or visual features were ingested.** This has been corrected in the project record.

---

## 4. Evaluation Pitfall: Source-Video & Segment Leakage

Our forensic audit exposed a critical data-leakage hazard in AV-ASD:

1. **Multiple Clips from One Source Video:**
   * In `clip_extraction_log.csv`, many clips share identical `Original_ID` values.
   * For example, source video `_gEps3r7FfQ` contributes `_gEps3r7FfQ_50_53` and `_gEps3r7FfQ_25_28`.
   * If an experiment randomly splits clips into train and test sets (as in standard 80/20 holdouts), the test set contains different segments of the exact same child recorded in the exact same room with the exact same lighting and microphone.
2. **The 506-Segment Domination Problem:**
   * In an earlier local 3-class segment task (`avasd_3class_hyper.py`), a reported accuracy of **76.59%** was achieved.
   * Our audit revealed that a single 29-minute video contributed **506 out of the 961 total segments**, all labeled `Upper Limb Stereotypies`.
   * The high accuracy was an artifact of extreme sample weight from one video, not generalizable behavior detection.
   * When aggregated to the video level (`avasd_video_level_nested.py`), accuracy dropped to **60.92%** (against a majority-class baseline of 52.87%).

---

## 5. Methodological Boundary & Recommendation

```
+---------------------------------------------------------------------------------------------------+
|                                  AV-ASD VS MMASD COMPARATIVE REALITY                              |
+--------------------------+------------------------------------+-----------------------------------+
| Feature / Dimension      | MMASD (Therapy Exercise Skeletons) | AV-ASD (In-the-Wild Video Clips)  |
+--------------------------+------------------------------------+-----------------------------------+
| Nature of Task           | 10 physical exercise classes       | 10 multi-label clinical behaviors |
| Input Modality           | Continuous 2D OpenPose skeletons   | Unaligned raw video & audio       |
| Measured Confidence      | Real OpenPose PAF confidence       | No keypoint confidence available  |
| Local Integrity          | 100% of 1,374 clips intact         | 262 video clips missing (666/928) |
| Usability for RA-STGT    | Fully Supported & Verified         | Unfeasible without full media     |
+--------------------------+------------------------------------+-----------------------------------+
```

### Final Conclusion for BTech Thesis
* **Do NOT force multimodal fusion on an incomplete dataset:** Attempting to build an audio-visual fusion model on AV-ASD with 262 missing video clips and no visual features would require synthetic shortcuts that would not survive academic scrutiny.
* **Rigorous Positioning:** Present AV-ASD as an **independent forensic evaluation and baseline case study** that exposes the challenges of in-the-wild video collection and source-video data leakage.
* **Core Contribution Focus:** Keep the proposed **RA-STGT** framework firmly anchored on MMASD, where raw 2D keypoints, measured detection confidence, and strict subject-independent splits are verified and 100% complete.
