"""
Automated Verification & Test Suite for the PR-STGT Research Repository.
Verifies all data loaders, model variants, checkpoints, forward passes, and inference logic.
"""

from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np

ROOT = Path("d:/Documents/Autism_Project")
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import torch
import unittest

from novel_contribution.data_processing.dataset_loader import (
    load_raw_mmasd,
    create_subject_independent_folds,
    partition_train_val_test,
    CLASSES
)
from research_stage3.models.pr_stgt_stage3 import PRSTGTStage3, count_parameters
from inference_pipeline import load_model_from_checkpoint, run_clinical_inference


class TestPRSTGTPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.data = load_raw_mmasd()
        cls.device = torch.device("cpu")

    def test_01_dataset_integrity(self):
        """Verify MMASD raw data dimensions and class counts."""
        self.assertEqual(len(self.data["y"]), 1374, "Sample count must be exactly 1,374")
        self.assertEqual(self.data["X"].shape[1:], (64, 25, 4), "X shape must be (64, 25, 4)")
        self.assertEqual(self.data["Q"].shape[1:], (64, 25), "Q shape must be (64, 25)")
        self.assertEqual(len(np.unique(self.data["groups"])), 32, "Must have 32 cohorts")
        self.assertEqual(len(CLASSES), 10, "Must have 10 therapy classes")

    def test_02_group_independent_splits(self):
        """Verify zero cohort leakage across 5 GroupKFold splits."""
        folds = create_subject_independent_folds(self.data["groups"], n_folds=5, seed=42)
        self.assertEqual(len(folds), 5, "Must have exactly 5 folds")
        all_test_groups = []
        for fold, (train_idx, test_idx) in enumerate(folds):
            train_grps = set(self.data["groups"][train_idx])
            test_grps = set(self.data["groups"][test_idx])
            overlap = train_grps.intersection(test_grps)
            self.assertEqual(len(overlap), 0, f"DATA LEAKAGE: Fold {fold} has overlapping cohorts {overlap}!")
            all_test_groups.extend(list(test_grps))
        self.assertEqual(len(all_test_groups), 32, "All 32 cohorts must be tested across the 5 folds")

    def test_03_pr_stgt_model_variants_forward(self):
        """Test forward passes of all 4 architectural configurations."""
        batch_size = 4
        x = torch.randn(batch_size, 64, 25, 4)
        q = torch.rand(batch_size, 64, 25)
        
        configs = [
            ("E1_Baseline", False, False, 78250),
            ("E2_Topology", True, False, 78250),
            ("E3_Phase", False, True, 78506),
            ("E5_Full", True, True, 78506)
        ]
        
        for name, use_topo, use_phase, expected_params in configs:
            model = PRSTGTStage3(
                in_channels=4,
                num_classes=10,
                d_model=64,
                num_spatial_heads=2,
                num_temporal_heads=2,
                use_relational_edges=use_topo,
                use_relational_reliability=use_topo,
                use_phase_stream=use_phase
            )
            model.eval()
            param_cnt = count_parameters(model)
            self.assertEqual(param_cnt, expected_params, f"{name} parameter mismatch: {param_cnt} vs {expected_params}")
            
            with torch.no_grad():
                out = model(x, q)
            self.assertEqual(out.shape, (batch_size, 10), f"{name} output shape mismatch: {out.shape}")

    def test_04_checkpoint_loading(self):
        """Verify that saved Stage 2 and Stage 3 checkpoints load correctly."""
        ckpts_to_test = [
            (ROOT / "research_stage2/experiments/checkpoints/e2_topology_only/fold_1.pt", "e2"),
            (ROOT / "research_stage2/experiments/checkpoints/e5_full_pr_stgt/fold_1.pt", "e5"),
            (ROOT / "research_stage3/experiments/checkpoints/e7_topo_margin/fold_1.pt", "e7"),
            (ROOT / "research_stage3/experiments/checkpoints/e9_full_reproduction/fold_1.pt", "e9"),
        ]
        for p, m_type in ckpts_to_test:
            self.assertTrue(p.exists(), f"Checkpoint missing: {p}")
            model, temp = load_model_from_checkpoint(p, model_type=m_type)
            self.assertGreater(temp, 0.5, f"Temperature must be realistic, got {temp}")
            self.assertIsInstance(model, PRSTGTStage3)

    def test_05_clinical_inference_logic(self):
        """Verify the complete end-to-end inference and selective prediction decision logic."""
        p = ROOT / "research_stage2/experiments/checkpoints/e2_topology_only/fold_1.pt"
        model, _ = load_model_from_checkpoint(p, model_type="e2")
        
        # Simulated clean clip (64 frames, 25 joints, 4 feature channels: x, y, vx, vy)
        x_clean = np.random.randn(64, 25, 4)
        q_clean = np.ones((64, 25)) * 0.85
        
        res = run_clinical_inference(model, x_clean, q_clean, risk_threshold=0.50)
        self.assertIn("top1_class", res)
        self.assertIn("confidence", res)
        self.assertIn("decision", res)
        self.assertIn(res["decision"], ["ACCEPT_AUTOMATED_LOG", "ABSTAIN_ROUTE_TO_CLINICIAN"])
        self.assertEqual(len(res["top3_predictions"]), 3)
        self.assertAlmostEqual(sum(res["all_probabilities"].values()), 1.0, places=4)

        # Simulated occluded wrist clip (should trigger occlusion alert and ABSTAIN)
        q_occluded = np.ones((64, 25)) * 0.85
        q_occluded[:, [4, 7]] = 0.05  # Wrists occluded
        res_occ = run_clinical_inference(model, x_clean, q_occluded, risk_threshold=0.50)
        self.assertTrue(res_occ["wrist_occluded"], "Must detect wrist occlusion")
        self.assertEqual(res_occ["decision"], "ABSTAIN_ROUTE_TO_CLINICIAN", "Must abstain when wrists are occluded")


if __name__ == "__main__":
    unittest.main()
