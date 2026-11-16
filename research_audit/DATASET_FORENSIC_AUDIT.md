# DATASET FORENSIC AUDIT: MMASD & AV-ASD

**Author/Role:** Senior AI/ML Research Engineer & Computer Vision Researcher  
**Project:** Autism Therapy Activity Recognition (Final-Year BTech CSE-AI Project)  
**Date of Audit:** November 13, 2026  
**Status:** Audit Completed; Workspace & Raw Archives Preserved Without Modification  

---

## 1. Executive Summary & Purpose

This document provides a comprehensive forensic inventory of the datasets located in the workspace `D:\Documents\Autism_Project` and the primary system download directory `D:\Downloads`. 

In accordance with strict research preservation principles:
1. **Zero Overwrite / Zero Modification:** No existing archive, preprocessed array, experiment output, or source script was modified, renamed, moved, or deleted.
2. **Empirical Verification:** Every file size, item count, shape, and label set reported below was verified through direct inspection of archive headers and array manifests.
3. **Task & Metric Separation:** MMASD (therapy activity recognition from skeletal kinematics) and AV-ASD (multimodal autism behavior detection from video clips) are treated as separate benchmarks with distinct label spaces, sample units, and evaluation protocols.

---

## 2. Archive Inventory & Storage Verification

### 2.1 Primary Download Directory (`D:\Downloads`)

| Archive / File Name | Size (Bytes) | Size (MB / GB) | Last Modified | Contents Summary |
|---|---:|---:|---|---|
| `mmasd-20261111T131057Z-1-001.zip` | 1,744,826,861 | 1.62 GB (1,664 MB) | 2026-11-11 19:44:30 | MMASD raw release containing `Dataset_FINAL` directory with 2D/3D skeleton zip files and ADOS clinical ratings |
| `clips_video-20261113T032829Z-1-001.zip` | 2,048,231,607 | 1.91 GB (1,953 MB) | 2026-11-13 11:21:30 | Part 1 of AV-ASD raw video clips: 333 `.mp4` video recordings |
| `clips_video-20261113T032829Z-1-002.zip` | 2,105,005,273 | 1.96 GB (2,007 MB) | 2026-11-13 11:34:43 | Part 2 of AV-ASD raw video clips: 333 `.mp4` video recordings |
| `csvs-20261112T164029Z-1-001.zip` | 53,788 | 52.5 KB | 2026-11-12 22:10:45 | AV-ASD annotation and partition files (`dataset.csv`, `train.csv`, `val.csv`, `test.csv`, `clip_extraction_log.csv`, `recovery_attempts.csv`) |
| `AV_ASD_audio_stage1_audit_bundle (1).zip` | 3,067,308 | 2.92 MB | 2026-11-11 15:58:24 | AV-ASD audio audit bundle: 5,410 segment acoustic descriptors and alignment logs |
| `AV_ASD_audio_stage1_audit_bundle (2).zip` | 3,067,246 | 2.92 MB | 2026-11-11 16:48:45 | Secondary copy of audio audit bundle |
| `AV_ASD_audio_stage1_audit_bundle.zip` | 97,152 | 94.9 KB | 2026-11-11 15:22:05 | Initial audio audit archive |
| `MMASD_publication_explainability_figures.zip` | 1,992,767 | 1.90 MB | 2026-11-16 23:30:19 | Explainability figure bundle (Grad-CAM joint/temporal attribution maps and SVGs) |
| `MMASD.pdf` | 2,416,256 | 2.30 MB | 2026-10-17 06:41:03 | 11-page project milestone presentation by Team 39 |

### 2.2 Workspace Raw Directory (`D:\Documents\Autism_Project\data\raw`)

| Subdirectory / Path | Size (Bytes) | File / Item Count | Contents Description |
|---|---:|---:|---|
| `mmasd/Dataset_FINAL/2D skeleton/2D_openpose.zip` | 315,459,994 | 254,674 files (256,379 items) | Frame-by-frame OpenPose JSON files across 1,374 activity recording folders |
| `mmasd/Dataset_FINAL/3D skeleton/ROMP_3D_Coordinates_71-joints.zip` | 795,772,821 | 311,080 files (312,773 items) | Frame-by-frame ROMP 3D coordinate `.npz` files (71 joints, 3 tracked individuals) |
| `mmasd/Dataset_FINAL/ados_clinical_ratings.csv` | 2,411 | 42 lines | ADOS-2 diagnostic assessment scores across 36 child records |
| `mmasd/Dataset_FINAL/ADOS_rating.xlsx` | 47,541 | 1 sheet (`ADOS-2`) | Excel workbook mirror of ADOS-2 clinical diagnostic ratings |
| `mmasd/Dataset_FINAL/README.txt` | 108 | 1 line | Points to official GitHub repository (`Li-Jicheng/MMASD-...`) |
| `mmasd/mmasd/Dataset_FINAL-20260813T140204Z-1-002.zip` | 1,744,560,345 | 6 files | Complete upstream zip mirror containing the 2D and 3D sub-archives |
| `AV-ASD/clips_video-20261113T032829Z-1-001.zip` | 2,048,231,607 | 333 files | Exact mirror of Part 1 video clips in workspace raw storage |
| `AV-ASD/clips_video-20261113T032829Z-1-002.zip` | 2,105,005,273 | 333 files | Exact mirror of Part 2 video clips in workspace raw storage |

### 2.3 Workspace Processed Directory (`D:\Documents\Autism_Project\data\processed`)

| Processed File | Size | Data Shapes & Types | Internal Keys & Description |
|---|---:|---|---|
| `mmasd_features/sessions_features.npz` | 12.04 MB | `features`: `(1693,)` (object array)<br>`labels`: `(1693,)` (`int64`)<br>`subjects`: `(1693,)` (`U5`)<br>`sessions`: `(1693,)` (`U19`) | Pre-extracted multi-bank temporal features (1,248 dims) for 1,693 sessions across 11 activity classes and 32 stored subject IDs |
| `mmasd_features/sessions_rich_features.npz` | 3.11 MB | `X`: `(1693, 506)` (`float32`) | Pre-extracted rich domain features (506 dims) for 1,693 sessions |
| `mmasd_openpose/mmasd_openpose_64frames.npz` | 36.56 MB | `X`: `(1374, 64, 25, 4)` (`float32`)<br>`Q`: `(1374, 64, 25)` (`float32`)<br>`y`: `(1374,)` (`int64`)<br>`groups`: `(1374,)` (`U5`)<br>`clips`: `(1374,)` (`U74`)<br>`classes`: `(10,)` (`U3`) | Uniformly resampled (64 frames) 2D OpenPose keypoints ($x, y, \Delta x, \Delta y$) and per-joint detection confidence ($Q \in [0, 1]$) for 1,374 clips across 10 classes and 32 stored IDs |

---

## 3. MMASD Dataset Forensic Analysis

### 3.1 Dataset Overview & Provenance

* **Dataset Name:** Multimodal Autism Spectrum Disorder (MMASD) Dataset
* **Official Release Repository:** `https://github.com/Li-Jicheng/MMASD-A-Multimodal-Dataset-for-Autism-Intervention-Analysis`
* **Original Publication:** Li et al., "MMASD: A Multimodal Dataset for Autism Intervention Analysis in Therapeutic Environments", ACM TOCHI / ICMI.
* **Domain:** Play and music-based occupational therapy intervention sessions for children with ASD.

### 3.2 Modality Breakdown in Raw Storage

1. **2D Skeletons (`2D_openpose.zip`):**
   * **Format:** JSON files per frame (`output/<class_dir>/<clip_id>/<clip_id>_<frame_id>_keypoints.json`).
   * **Keypoint Topology:** OpenPose `BODY_25` topology (25 physical joints).
   * **Data Stored per Joint:** $[x, y, c]$, where $x, y$ are 2D pixel coordinates and $c \in [0.0, 1.0]$ is the raw detector confidence score.
   * **Additional Arrays in JSON:** `face_keypoints_2d` (70 keypoints), `hand_left_keypoints_2d` (21 keypoints), `hand_right_keypoints_2d` (21 keypoints). In the standard local pipeline, only the 25 body keypoints are modeled.

2. **3D Skeletons (`ROMP_3D_Coordinates_71-joints.zip`):**
   * **Format:** `.npz` files per frame (`<class_dir>/<clip_id>/<frame_id>.npz`).
   * **Keypoint Topology:** ROMP / SMPL 71-joint topology.
   * **Data Stored:** Array named `coordinates` of shape `(3, 71, 3)` representing 3 tracked individuals per frame, 71 joints, and 3D camera coordinates $(X, Y, Z)$ in millimeters/meters.
   * **Critical Forensic Finding:** The 3D ROMP files **do not contain explicit detection confidence scores** ($c$) per joint. Furthermore, person identity across the 3 detected slots is unassigned and subject to tracking swaps (e.g., child vs therapist vs parent).

3. **Clinical Ratings (`ados_clinical_ratings.csv`):**
   * Contains diagnostic Autism Diagnostic Observation Schedule (ADOS-2) scores for 36 child identifiers.
   * Fields include: `ID#`, `Gender`, `Chronological Age`, `Module` (1, 2, or 3), `Social Affect Total`, `RRB Total`, `Overall Total`, `ADOS Comparison Score (1-10)`, and `Severity of Autism`.
   * **Critical Forensic Finding:** These ratings are diagnostic clinical assessments of the **child**, not activity-level or frame-level ratings of movement quality or task performance.

### 3.3 Activity Class Inventory & Discrepancies vs Published Literature

The local 2D OpenPose archive contains **1,374 clip folders** covering **10 activity classes**. 

A strict forensic comparison against the published 10-class setup from Ay et al. (*Sensors* 2026, PMC13431331) reveals a major discrepancy:

| Class Code | Activity Name | Local Clip Count | Published 10-Class Count (Ay et al. 2026) | Discrepancy ($\Delta$) | Notes |
|---|---|---:|---:|---:|---|
| `as` | Arm swing | 105 | 105 | 0 | Exact match |
| `bs` | Body swing | 117 | 119 | -2 | 2 clips omitted or unreadable in local zip |
| `ce` | Chest expansion | 114 | 114 | 0 | Exact match |
| `dr` | Drumming | **343** | **168** | **+175** | **Major discrepancy: local archive contains 175 extra drumming takes/recordings** |
| `mfs` | Maracas forward shaking | 101 | 103 | -2 | 2 clips omitted |
| `ms` | Maracas shaking | 130 | 130 | 0 | Exact match |
| `sac` | Sing and clap | 113 | 113 | 0 | Exact match |
| `sq` | Squat | 102 | 101 | +1 | 1 extra take in local archive |
| `tr` | Tree pose | 129 | 129 | 0 | Exact match |
| `tw` | Twist pose | 120 | 120 | 0 | Exact match |
| **Total (10 classes)** | | **1,374** | **1,202** | **+172** | **Local raw archive has 172 more clips than published benchmark** |
| `fg` | Frog pose | 0 in 2D zip (present in feature bank: ~130) | Excluded in published 10-class setup | N/A | Excluded from 10-class benchmark due to low visibility/prone posture |

#### Significance of the Discrepancy
In Ay et al. (2026), the 10-class dataset was filtered to exactly 1,202 clips. In the uncurated raw `2D_openpose.zip`, drumming contains 343 clips because multiple therapy trial repetitions and camera sub-angles were retained. Any model trained on the uncurated 1,374 clips cannot claim to reproduce the exact numerical benchmark of Ay et al. (89.29% accuracy) without explicitly documenting this difference in clip distribution.

### 3.4 Stored Group Identifiers vs Verified Participant Identity

* In `data/processed/mmasd_openpose/mmasd_openpose_64frames.npz`, there are **32 unique stored IDs**:
  `['20564', '20583', '20594', '20613', '20626', '40023', '40074', '40143', '40163', '40323', '40453', '40493', '40504', '40533', '40543', '40555', '40683', '40714', '40723', '40743', '40753', '40794', '41004', '41023', '41063', '41083', '41093', '41103', '41113', '41123', '41133', '41143']`
* **Forensic Audit:** Cross-referencing these 32 IDs with `ados_clinical_ratings.csv` shows that:
  1. 22 of the 32 IDs match exact numerical child clinical IDs in the ADOS-2 spreadsheet (e.g., `20564`, `20583`, `40023`, `40794`).
  2. 10 of the stored IDs (the `41xxx` series: `41004`, `41023`, `41063`, `41083`, `41093`, `41103`, `41113`, `41123`, `41133`, `41143`) do not appear in the ADOS-2 diagnostic rating sheet.
  3. These identifiers are extracted strictly from filename tokens (e.g., `as_20583_D16_000_i` $\rightarrow$ `20583`).
  4. While grouping by these 32 IDs enforces cross-validation separation across recording sessions, **it cannot be guaranteed that every ID represents a completely distinct clinical participant without upstream institutional metadata.** This must be reported transparently as a methodological threat to validity.

---

## 4. AV-ASD Dataset Forensic Analysis

### 4.1 Dataset Overview & Provenance

* **Dataset Name:** Audio-Visual Autism Spectrum Disorder (AV-ASD)
* **Official Repository:** `https://github.com/ShijianDeng/AV-ASD`
* **Published Paper:** Deng et al., "Hear Me, See Me, Understand Me: Audio-Visual Autism Behavior Recognition", *IEEE Transactions on Multimedia*, 2025 (arXiv:2406.02554).
* **Domain:** Naturalistic, in-the-wild audio-visual clips collected from public video platforms depicting children with ASD engaging in various behaviors.

### 4.2 Annotation Structure & Split Reconciliation

Inspecting `csvs-20261112T164029Z-1-001.zip`:

| CSV File | Total Rows (incl. header) | Number of Clips | Purpose in Benchmark |
|---|---:|---:|---|
| `dataset.csv` | 929 | 928 | Full published benchmark set |
| `train.csv` | 554 | 553 | Official training split (59.6%) |
| `val.csv` | 194 | 193 | Official validation split (20.8%) |
| `test.csv` | 183 | 182 | Official test split (19.6%) |
| `clip_extraction_log.csv` | 929 | 928 | Maps `Video_ID` to source `Original_ID`, start time, end time, and duration |
| `recovery_attempts.csv` | 546 | 545 attempts | Audit log of video recovery attempts from original web hosts |

The published task defines **10 multi-label behavior categories**:
1. Absence or Avoidance of Eye Contact
2. Aggressive Behavior
3. Hyper- or Hyporeactivity to Sensory Input
4. Non-Responsiveness to Verbal Interaction
5. Non-Typical Language
6. Object Lining-Up
7. Self-Hitting or Self-Injurious Behavior
8. Self-Spinning or Spinning Objects
9. Upper Limb Stereotypies
10. Background

### 4.3 Raw Video Archive Forensic Audit & Missing Media

Inspecting `clips_video-20261113T032829Z-1-001.zip` (Part 1) and `clips_video-20261113T032829Z-1-002.zip` (Part 2):

* Total clips in Part 1: **333 `.mp4` video files**
* Total clips in Part 2: **333 `.mp4` video files**
* Total video clips present locally: **666 clips**
* Total clips required by `dataset.csv`: **928 clips**
* **Forensic Finding:** **262 video clips are missing from the local archives.** 
  The download consists of two ~2GB chunks from Google Drive; the remaining chunk (Part 3, ~1.5 GB containing the remaining 262 clips) was never completed or downloaded.
* Furthermore, inspecting `recovery_attempts.csv`:
  * `recovered`: 359
  * `download_failed`: 184
  * `bad_timestamp`: 2
  This confirms that in-the-wild video collection suffered from permanent YouTube video deletions and copyright takedowns upstream.

### 4.4 Local Audio Bundle Limitations

In previous local experiments (`AV_ASD_audio_stage1_audit_bundle.zip`):
* Contains 5,410 segment-level 32-dimensional acoustic descriptors extracted via OpenSMILE / VAD.
* When aligned to the official 928 annotated clips, **only 227 clips possessed usable aligned audio features.**
* There are **zero pre-extracted visual features, pose keypoints, or speech transcripts** in the local AV-ASD directory.
* An earlier experiment in `experiments/avasd_multimodal_gated_fusion.py` attempted "gated multimodal fusion" by blending audio predictions with class priors; **it did not ingest actual video frames or visual features.** Calling that experiment "multimodal" was a misnomer in the historical notes.

---

## 5. Critical Forensic Findings & Constraints Summary

```
+---------------------------------------------------------------------------------------------------+
|                                  DATASET FORENSIC AUDIT SUMMARY                                   |
+------------------------------------+--------------------------------------------------------------+
| Dimension                          | Forensic Finding / Hard Constraint                           |
+------------------------------------+--------------------------------------------------------------+
| MMASD 2D OpenPose Keypoints        | 1,374 clips (254,674 frames); 25 joints with (x, y, c).      |
|                                    | Detection confidence c in [0, 1] exists and is real.        |
|                                    | 6.48% of keypoints have c = 0 (missing/occluded).            |
+------------------------------------+--------------------------------------------------------------+
| MMASD 3D ROMP Coordinates          | 71 joints for 3 detected persons; no confidence values.      |
|                                    | Severe multi-person identity swapping.                       |
+------------------------------------+--------------------------------------------------------------+
| MMASD Activity Classes             | 10 classes in 2D OpenPose; 11 classes in feature bank.       |
|                                    | 343 drumming clips vs 168 in published paper (+175 takes).   |
+------------------------------------+--------------------------------------------------------------+
| MMASD Clinical ADOS-2 Data         | Child-level diagnostic scores (n=36).                        |
|                                    | NO activity-level or frame-level movement quality ratings.   |
|                                    | CANNOT be used as sample-level supervision.                  |
+------------------------------------+--------------------------------------------------------------+
| AV-ASD Video Media                 | Only 666 of 928 clips present locally (262 missing).         |
+------------------------------------+--------------------------------------------------------------+
| AV-ASD Aligned Audio               | Only 227 clips have usable aligned acoustic descriptors.     |
|                                    | No local visual/pose features extracted.                    |
+------------------------------------+--------------------------------------------------------------+
| Recommendation for Core Project    | Primary novel contribution MUST be developed on MMASD 2D    |
|                                    | OpenPose where measured confidence and clips are intact.     |
+------------------------------------+--------------------------------------------------------------+
```
