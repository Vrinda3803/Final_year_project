"""
End-to-End Clinical Inference Pipeline for Autism Therapy Physical Activity Recognition.
Supports PR-STGT Stage 3 & Stage 2 Checkpoints (e.g., E2 Topology Only, E5 Full PR-STGT).

Features:
- Loads trained PyTorch fold checkpoints with temperature calibration.
- Computes joint kinematic reliability gating from 2D coordinates.
- Outputs calibrated probabilities, Top-1 and Top-3 predictions.
- Applies risk-coverage selective prediction (ACCEPT vs ROUTE TO CLINICIAN).
- Detects joint occlusion / tracking degradation (e.g. wrist occlusion alert).
"""

from __future__ import annotations
import sys, os, argparse, json
from pathlib import Path

ROOT = Path("d:/Documents/Autism_Project")
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from novel_contribution.data_processing.dataset_loader import load_raw_mmasd, CLASSES
from research_stage3.models.pr_stgt_stage3 import PRSTGTStage3

# Descriptive class mapping for clinicians
CLASS_DESCRIPTIONS = {
    "as": "Arm Swing (Coordination)",
    "bs": "Ball Squeeze (Fine Motor Grip)",
    "ce": "Chest Expansion (Bilateral Upper Body)",
    "dr": "Drumming (Alternating Rhythm)",
    "mfs": "Maracas Fast Shake (Rapid Cadence)",
    "ms": "Maracas Regular Shake (Cadence)",
    "sac": "Arm Swing / Arm Circle (Range of Motion)",
    "sq": "Squats (Lower Limb Strength)",
    "tr": "Tree Pose (Static Balance)",
    "tw": "Twist (Trunk Rotation)"
}


def load_model_from_checkpoint(ckpt_path: Path, model_type: str = "e2", device: torch.device = torch.device("cpu")):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Configure architecture based on experiment type
    use_topo = True
    use_phase = True if model_type in ("e3", "e5", "e6", "e8", "e9") else False
    
    if model_type in ("e0", "e1", "e3", "e4", "e8"):
        use_topo = False
        
    model = PRSTGTStage3(
        in_channels=4,
        num_classes=10,
        d_model=64,
        num_spatial_heads=2,
        num_temporal_heads=2,
        dropout=0.25,
        use_relational_edges=use_topo,
        use_relational_reliability=use_topo,
        use_phase_stream=use_phase
    ).to(device)
    
    # Load state dict
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict)
    
    # Set calibrated temperature
    temperature = ckpt.get("temperature", 1.0)
    model.temperature.data.fill_(float(temperature))
    model.eval()
    return model, temperature


def run_clinical_inference(
    model: PRSTGTStage3,
    x_input: np.ndarray,
    q_input: np.ndarray,
    device: torch.device = torch.device("cpu"),
    risk_threshold: float = 0.60
) -> dict:
    """
    x_input: (T, V, 2) or (B, T, V, 2)
    q_input: (T, V) or (B, T, V)
    """
    if x_input.ndim == 3:
        x_input = x_input[np.newaxis, ...]
        q_input = q_input[np.newaxis, ...]
        
    x_tensor = torch.tensor(x_input, dtype=torch.float32, device=device)
    q_tensor = torch.tensor(q_input, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # Raw logits and temperature-calibrated probabilities
        logits = model(x_tensor, q_tensor)
        cal_logits = logits / model.temperature
        probs = torch.softmax(cal_logits, dim=-1).cpu().numpy()[0]
        
    top1_idx = int(np.argmax(probs))
    top1_class = CLASSES[top1_idx]
    confidence = float(probs[top1_idx])
    
    # Sort top-3
    top3_indices = np.argsort(probs)[::-1][:3]
    top3_preds = [
        {
            "rank": rank + 1,
            "class_code": CLASSES[idx],
            "class_name": CLASS_DESCRIPTIONS.get(CLASSES[idx], CLASSES[idx]),
            "probability": float(probs[idx])
        }
        for rank, idx in enumerate(top3_indices)
    ]
    
    # Check joint occlusion status (wrists: joints 4 and 7)
    mean_wrist_conf = float(np.mean(q_input[0, :, [4, 7]]))
    mean_overall_conf = float(np.mean(q_input[0, :, :]))
    wrist_occluded = mean_wrist_conf < 0.20
    
    # Selective prediction routing decision
    if confidence >= risk_threshold and not wrist_occluded:
        decision = "ACCEPT_AUTOMATED_LOG"
        explanation = f"Confidence ({confidence:.1%}) meets risk threshold ({risk_threshold:.1%}); keypoints well-tracked."
    else:
        decision = "ABSTAIN_ROUTE_TO_CLINICIAN"
        reasons = []
        if confidence < risk_threshold:
            reasons.append(f"Confidence ({confidence:.1%}) below clinical threshold ({risk_threshold:.1%})")
        if wrist_occluded:
            reasons.append(f"Severe wrist occlusion detected (mean wrist confidence: {mean_wrist_conf:.2f})")
        explanation = "; ".join(reasons)
        
    return {
        "top1_class": top1_class,
        "top1_description": CLASS_DESCRIPTIONS.get(top1_class, top1_class),
        "confidence": confidence,
        "temperature": float(model.temperature.item()),
        "decision": decision,
        "explanation": explanation,
        "wrist_occluded": wrist_occluded,
        "mean_wrist_confidence": mean_wrist_conf,
        "mean_overall_confidence": mean_overall_conf,
        "top3_predictions": top3_preds,
        "all_probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    }


def main():
    parser = argparse.ArgumentParser(description="Clinical Inference for Autism Physical Therapy Recognition")
    parser.add_argument("--model-type", type=str, default="e2", choices=["e2", "e5", "e7", "e9"], help="Model type to load (default: e2 topology only - best performer)")
    parser.add_argument("--fold", type=int, default=1, help="Checkpoint fold to use (0-4, default: 1)")
    parser.add_argument("--sample-idx", type=int, default=42, help="Sample index from MMASD dataset to test")
    parser.add_argument("--threshold", type=float, default=0.60, help="Clinical selective prediction confidence threshold")
    args = parser.parse_args()
    
    device = torch.device("cpu")
    
    # Select checkpoint path
    if args.model_type == "e2":
        ckpt_path = ROOT / "research_stage2" / "experiments" / "checkpoints" / "e2_topology_only" / f"fold_{args.fold}.pt"
    elif args.model_type == "e5":
        ckpt_path = ROOT / "research_stage2" / "experiments" / "checkpoints" / "e5_full_pr_stgt" / f"fold_{args.fold}.pt"
    elif args.model_type == "e7":
        ckpt_path = ROOT / "research_stage3" / "experiments" / "checkpoints" / "e7_topo_margin" / f"fold_{args.fold}.pt"
    else:
        ckpt_path = ROOT / "research_stage3" / "experiments" / "checkpoints" / "e9_full_reproduction" / f"fold_{args.fold}.pt"
        
    print(f"\n{'='*75}\nAUTISM THERAPY ACTIVITY RECOGNITION — CLINICAL INFERENCE PIPELINE\n{'='*75}")
    print(f"Loading Model Checkpoint: {ckpt_path.name} (Type: {args.model_type.upper()}, Fold: {args.fold})")
    model, temp = load_model_from_checkpoint(ckpt_path, model_type=args.model_type, device=device)
    print(f"Model successfully loaded! Calibrated temperature: T = {temp:.4f}")
    
    print("\nLoading MMASD dataset and partition for Fold...")
    raw_data = load_raw_mmasd()
    from novel_contribution.data_processing.dataset_loader import create_subject_independent_folds, partition_train_val_test
    
    folds = create_subject_independent_folds(raw_data["groups"], n_folds=5, seed=42)
    train_idx, test_idx = folds[args.fold]
    fit_d, val_d, test_d, norm_stats = partition_train_val_test(raw_data, train_idx, test_idx, seed=42 + args.fold)
    
    # Fit velocity priors exactly on fit partition
    model.reliability_estimator.fit_velocity_priors(fit_d["dataset"].X.numpy(), fit_d["dataset"].Q.numpy())
    
    # Select sample from test partition
    sample_sub_idx = min(args.sample_idx, len(test_d["dataset"]) - 1)
    x_clip = test_d["dataset"].X[sample_sub_idx].numpy()
    q_clip = test_d["dataset"].Q[sample_sub_idx].numpy()
    true_label_idx = int(test_d["dataset"].y[sample_sub_idx].item())
    true_label = CLASSES[true_label_idx]
    clip_name = str(test_d["dataset"].clips[sample_sub_idx])
    group_id = str(test_d["dataset"].groups[sample_sub_idx])
    
    print(f"Test Partition Clip: {clip_name} (Cohort: {group_id}, Frames: {x_clip.shape[0]}, True Activity: {true_label} - {CLASS_DESCRIPTIONS[true_label]})")
    
    # Run clinical inference
    res = run_clinical_inference(model, x_clip, q_clip, device=device, risk_threshold=args.threshold)
    
    print(f"\n{'-'*75}\nCLINICAL PREDICTION RESULTS\n{'-'*75}")
    print(f"Top-1 Predicted Activity: {res['top1_class']} ({res['top1_description']})")
    print(f"Calibrated Confidence:   {res['confidence']:.2%}")
    print(f"Ground Truth Match:      {'CORRECT ✓' if res['top1_class'] == true_label else 'INCORRECT ✗'}")
    print(f"Clinical Decision:       {res['decision']}")
    print(f"System Explanation:      {res['explanation']}")
    print(f"Mean Wrist Confidence:   {res['mean_wrist_confidence']:.2f} ({'Occlusion Alert!' if res['wrist_occluded'] else 'Clean Tracking'})")
    
    print("\nTop-3 Probability Distribution:")
    for pred in res["top3_predictions"]:
        bar = "█" * int(pred["probability"] * 30)
        print(f"  [{pred['rank']}] {pred['class_code']:<5} {pred['class_name']:<35} {pred['probability']:>6.1%}  {bar}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
