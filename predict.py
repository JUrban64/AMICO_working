import os
import argparse
import json
from pathlib import Path
import torch
import numpy as np

from model_ligand_cross_att import LigandCrossAttentionMIL, TARGET_NAMES
from model_self_attention import SelfAttentionMIL
from p2rank_utils import run_p2rank, parse_p2rank_output, find_p2rank_executable


class AMICOPredictor:
    """
    Inference predictor class for AMICO models (LigandCrossAttentionMIL / SelfAttentionMIL).
    Supports cofactor specificity prediction, binding pocket localization, epistemic
    uncertainty estimation via Monte Carlo Dropout, and end-to-end inference directly
    from raw PDB structures (P2Rank + ESM-2).
    """
    def __init__(self, checkpoint_path=None, config_json=None, model_type='auto', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        else:
            self.device = torch.device(device)

        hidden_dim = 256
        num_heads = 4
        dropout = 0.15

        if config_json and os.path.exists(config_json):
            with open(config_json, 'r') as f:
                cfg = json.load(f)
                hidden_dim = cfg.get('hidden_dim', hidden_dim)
                num_heads = cfg.get('num_heads', num_heads)
                dropout = cfg.get('dropout', dropout)
                if model_type == 'auto' and 'model' in cfg:
                    model_type = cfg['model']

        # Check if checkpoint exists in direct path or in weights/ directory
        if checkpoint_path and not os.path.exists(checkpoint_path):
            candidate_weights = os.path.join("weights", checkpoint_path)
            if os.path.exists(candidate_weights):
                checkpoint_path = candidate_weights

        # Load weights and auto-detect architecture if requested
        loaded_state = None
        ecfp_dim = 2048
        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(state, dict):
                if 'model_state_dict' in state:
                    state = state['model_state_dict']
                elif 'state_dict' in state:
                    state = state['state_dict']
                # Strip potential 'module.' prefix from DistributedDataParallel
                state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
            loaded_state = state

            # Auto-detect architecture from checkpoint weight keys
            if model_type == 'auto':
                if 'self_attn.in_proj_weight' in loaded_state:
                    model_type = 'self_attention_mil'
                elif 'cross_attn.in_proj_weight' in loaded_state:
                    model_type = 'ligand_cross_mil'

            # Auto-detect ECFP dimension from checkpoint if available
            if 'ligand_proj.0.weight' in loaded_state:
                ecfp_dim = loaded_state['ligand_proj.0.weight'].shape[1]

        # Initialize chosen architecture
        if model_type in ['self_attention_mil', 'self_att']:
            self.model_type = 'self_attention_mil'
            self.model = SelfAttentionMIL(
                feature_dim=1280,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_classes=5,
                dropout=dropout
            ).to(self.device)
        else:
            self.model_type = 'ligand_cross_mil'
            self.model = LigandCrossAttentionMIL(
                feature_dim=1280,
                ecfp_dim=ecfp_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_classes=5,
                dropout=dropout
            ).to(self.device)

        if loaded_state is not None:
            self.model.load_state_dict(loaded_state)
            print(f"Checkpoint successfully loaded from {checkpoint_path} (Architecture: {self.model.__class__.__name__})")
        else:
            print(f"Warning: Checkpoint '{checkpoint_path}' not found. Initialized {self.model.__class__.__name__} with random weights.")

        self.model.eval()
        self._esm_extractor = None

    def _get_esm_extractor(self, model_name="facebook/esm2_t33_650M_UR50D"):
        """Lazy initialization of ESM-2 feature extractor."""
        if self._esm_extractor is None:
            from esm_extractor import ESMFeatureExtractor
            self._esm_extractor = ESMFeatureExtractor(model_name=model_name, device=self.device)
        return self._esm_extractor

    def _enable_mc_dropout(self):
        """Keep model in eval mode but activate Dropout layers for Monte Carlo sampling."""
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def predict(self, pocket_features, full_protein_feature, mc_samples=30, 
                confidence_threshold=0.50, uncertainty_threshold=0.15):
        """
        Runs model inference with optional Monte Carlo Dropout uncertainty estimation.
        
        Args:
            pocket_features: Tensor [N_pockets, 1280] or np.ndarray
            full_protein_feature: Tensor [1280] or np.ndarray
            mc_samples: Number of stochastic forward passes (1 = deterministic, >=20 for MC Dropout)
            confidence_threshold: Minimum mean probability required to confirm cofactor binding
            uncertainty_threshold: Maximum allowed standard deviation (variance) for confident prediction

        Returns:
            dict containing prediction results, epistemic uncertainty, and non-binder detection
        """
        if isinstance(pocket_features, np.ndarray):
            pocket_features = torch.FloatTensor(pocket_features)
        if isinstance(full_protein_feature, np.ndarray):
            full_protein_feature = torch.FloatTensor(full_protein_feature)

        if pocket_features.dim() == 1 and pocket_features.numel() == 0:
            pocket_features = pocket_features.view(0, 1280)
        elif pocket_features.dim() == 2:
            pocket_features = pocket_features.unsqueeze(0)  # [1, N, 1280]
        
        if full_protein_feature.dim() == 1:
            full_protein_feature = full_protein_feature.unsqueeze(0)  # [1, 1280]

        pocket_features = pocket_features.to(self.device)
        full_protein_feature = full_protein_feature.to(self.device)
        
        num_pockets = pocket_features.size(1)
        padding_mask = torch.zeros(1, num_pockets, dtype=torch.bool, device=self.device)

        if mc_samples > 1:
            # === MONTE CARLO DROPOUT MODE ===
            self._enable_mc_dropout()
            all_probs = []
            all_attns = []

            with torch.no_grad():
                for _ in range(mc_samples):
                    logits, attn_weights = self.model(pocket_features, padding_mask, full_protein_feature)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    attn = attn_weights.squeeze(0).cpu().numpy()
                    all_probs.append(probs)
                    all_attns.append(attn)

            all_probs = np.stack(all_probs, axis=0)  # [T, 5]
            all_attns = np.stack(all_attns, axis=0)  # [T, 5, N+1] or [T, N+1, N+1]

            mean_probs = np.mean(all_probs, axis=0)  # [5]
            std_probs = np.std(all_probs, axis=0)    # [5] (Epistemic uncertainty)
            mean_attn = np.mean(all_attns, axis=0)   # [5, N+1] or [N+1, N+1]
            
            # Predictive entropy: H = - sum(p * log(p))
            predictive_entropy = -float(np.sum(mean_probs * np.log(mean_probs + 1e-12)))

            pred_idx = int(np.argmax(mean_probs))
            pred_label = TARGET_NAMES[pred_idx]
            confidence = float(mean_probs[pred_idx])
            uncertainty = float(std_probs[pred_idx])
            
            # Non-binder / Out-of-Distribution protein detection
            is_non_binder = (confidence < confidence_threshold) or (uncertainty > uncertainty_threshold)
            binding_status = "NON_BINDER / UNKNOWN_COFACTOR" if is_non_binder else "BINDER"

            probs_dict = {
                TARGET_NAMES[i]: {
                    "mean_probability": round(float(mean_probs[i]), 4),
                    "uncertainty_std": round(float(std_probs[i]), 4)
                }
                for i in range(len(TARGET_NAMES))
            }
            used_attn = mean_attn

        else:
            # === DETERMINISTIC MODE ===
            self.model.eval()
            with torch.no_grad():
                logits, attn_weights = self.model(pocket_features, padding_mask, full_protein_feature)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                used_attn = attn_weights.squeeze(0).cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_label = TARGET_NAMES[pred_idx]
            confidence = float(probs[pred_idx])
            uncertainty = 0.0
            predictive_entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
            is_non_binder = confidence < confidence_threshold
            binding_status = "NON_BINDER / UNKNOWN_COFACTOR" if is_non_binder else "BINDER"

            probs_dict = {
                TARGET_NAMES[i]: {
                    "probability": round(float(probs[i]), 4)
                }
                for i in range(len(TARGET_NAMES))
            }

        # Interpret binding pocket attention
        if used_attn.ndim == 2 and used_attn.shape[0] == len(TARGET_NAMES):
            # LigandCrossAttentionMIL: [5, N+1] -> attention distribution for predicted cofactor
            cofactor_attn = used_attn[pred_idx]
            global_context_weight = float(cofactor_attn[0])
            pocket_attn_weights = cofactor_attn[1:]
        elif used_attn.ndim == 2:
            # SelfAttentionMIL: [N+1, N+1] -> attention from CLS global protein token (index 0)
            cls_attn = used_attn[0]
            global_context_weight = float(cls_attn[0])
            pocket_attn_weights = cls_attn[1:]
        elif used_attn.ndim == 1:
            global_context_weight = float(used_attn[0])
            pocket_attn_weights = used_attn[1:]
        else:
            global_context_weight = 0.0
            pocket_attn_weights = np.array([])

        best_pocket_idx = int(np.argmax(pocket_attn_weights)) + 1 if len(pocket_attn_weights) > 0 else None
        best_pocket_weight = float(np.max(pocket_attn_weights)) if len(pocket_attn_weights) > 0 else 0.0

        pocket_rankings = [
            {"pocket_id": i + 1, "attention_score": round(float(w), 4)}
            for i, w in sorted(enumerate(pocket_attn_weights), key=lambda x: x[1], reverse=True)
        ]

        return {
            "predicted_cofactor": "NONE (Non-binder)" if is_non_binder else pred_label,
            "raw_top_class": pred_label,
            "binding_status": binding_status,
            "confidence": round(confidence, 4),
            "uncertainty_std": round(uncertainty, 4),
            "predictive_entropy": round(predictive_entropy, 4),
            "is_confident_prediction": not is_non_binder,
            "mc_samples_used": mc_samples,
            "probabilities": probs_dict,
            "best_binding_pocket": best_pocket_idx if not is_non_binder else None,
            "raw_best_pocket": best_pocket_idx,
            "best_pocket_attention": round(best_pocket_weight, 4),
            "global_context_weight": round(global_context_weight, 4),
            "pocket_rankings": pocket_rankings
        }

    def predict_from_pdb(self, pdb_path, prank_exec=None, prank_out_dir=None, min_prob=0.0,
                         esm_model="facebook/esm2_t33_650M_UR50D", mc_samples=30,
                         confidence_threshold=0.50, uncertainty_threshold=0.15):
        """
        End-to-End prediction from a PDB structure:
          1. Execute P2Rank pocket prediction (or reuse cached output)
          2. Parse pocket amino acid sequences and 3D coordinates
          3. Compute ESM-2 embeddings (pockets + whole-protein global context)
          4. Run AMICO inference with Monte Carlo Dropout
          5. Map attention weights to physical 3D pocket coordinates for docking
        """
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        # 1. P2Rank
        if prank_out_dir and Path(prank_out_dir).exists():
            out_dir = Path(prank_out_dir)
            print(f"-> Reusing existing P2Rank outputs from {out_dir}")
        else:
            out_dir = run_p2rank(pdb_path, prank_exec=prank_exec)

        # 2. Parse pockets
        print(f"-> Parsing predicted pockets from {out_dir}...")
        parsed_data = parse_p2rank_output(out_dir, pdb_path, min_prob=min_prob)
        pockets = parsed_data['pockets']
        print(f"-> Found {len(pockets)} candidate pockets (min_prob >= {min_prob}).")

        # 3. ESM-2 extraction
        extractor = self._get_esm_extractor(model_name=esm_model)
        print("-> Generating ESM-2 embeddings...")
        pocket_features, full_protein_feature = extractor.extract_all_from_parsed(parsed_data)

        # 4. AMICO model inference
        print(f"-> Running {self.model_type} inference...")
        res = self.predict(
            pocket_features=pocket_features,
            full_protein_feature=full_protein_feature,
            mc_samples=mc_samples,
            confidence_threshold=confidence_threshold,
            uncertainty_threshold=uncertainty_threshold
        )

        # 5. Enrich results with P2Rank metadata
        pocket_map = {p['pocket_id']: p for p in pockets}
        enriched_rankings = []
        for r in res['pocket_rankings']:
            pid = r['pocket_id']
            p_info = pocket_map.get(pid, {})
            enriched_rankings.append({
                'pocket_id': pid,
                'name': p_info.get('name', f"pocket{pid}"),
                'attention_score': r['attention_score'],
                'p2rank_prob': p_info.get('probability', 0.0),
                'p2rank_score': p_info.get('score', 0.0),
                'center': p_info.get('center', [0.0, 0.0, 0.0]),
                'residue_count': p_info.get('residue_count', 0),
                'sequence': p_info.get('sequence', '')
            })

        res['pocket_rankings'] = enriched_rankings
        res['p2rank_output_dir'] = str(out_dir)

        # Identify top P2Rank pocket (rank 1 by score / probability) for docking
        if pockets:
            best_p2rank_pocket = max(pockets, key=lambda p: (p.get('score', 0.0), p.get('probability', 0.0)))
            res['best_p2rank_pocket_id'] = best_p2rank_pocket['pocket_id']
            res['best_p2rank_pocket_name'] = best_p2rank_pocket['name']
            res['best_p2rank_pocket_center'] = best_p2rank_pocket['center']
            res['best_p2rank_score'] = best_p2rank_pocket.get('score', 0.0)
            res['best_p2rank_prob'] = best_p2rank_pocket.get('probability', 0.0)
        else:
            res['best_p2rank_pocket_id'] = None
            res['best_p2rank_pocket_name'] = None
            res['best_p2rank_pocket_center'] = [0.0, 0.0, 0.0]
            res['best_p2rank_score'] = 0.0
            res['best_p2rank_prob'] = 0.0

        best_pid = res['best_binding_pocket']
        raw_pid = res.get('raw_best_pocket', best_pid)

        if best_pid and best_pid in pocket_map:
            res['best_pocket_center'] = pocket_map[best_pid]['center']
            res['best_pocket_name'] = pocket_map[best_pid]['name']
        else:
            res['best_pocket_center'] = [0.0, 0.0, 0.0]
            res['best_pocket_name'] = None

        if raw_pid and raw_pid in pocket_map:
            res['raw_best_pocket_center'] = pocket_map[raw_pid]['center']
            res['raw_best_pocket_name'] = pocket_map[raw_pid]['name']
        else:
            res['raw_best_pocket_center'] = [0.0, 0.0, 0.0]
            res['raw_best_pocket_name'] = None

        return res


def main():
    parser = argparse.ArgumentParser(description="AMICO: End-to-End P2Rank + ESM-2 + Attention MIL Inference & Docking")
    parser.add_argument('--pdb', type=str, default=None, help='Path to target protein PDB structure file')
    parser.add_argument('--prank', type=str, default=None, help='Path to P2Rank prank binary (default: auto-detected)')
    parser.add_argument('--p2rank-dir', type=str, default=None, help='Path to existing P2Rank output directory')
    parser.add_argument('--min-prob', type=float, default=0.0, help='Minimum P2Rank pocket probability threshold (0.0 = all)')
    parser.add_argument('--esm-model', type=str, default='facebook/esm2_t33_650M_UR50D', help='HuggingFace ESM-2 model identifier')

    parser.add_argument(
        '--model-type',
        type=str,
        default='auto',
        choices=['auto', 'ligand_cross_mil', 'self_attention_mil', 'self_att'],
        help='Model architecture type (auto, ligand_cross_mil, or self_attention_mil)'
    )
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model weights checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to Optuna configuration JSON')
    parser.add_argument('--mc-samples', type=int, default=30, help='Number of MC Dropout stochastic samples (1 = deterministic, 30 = MC Dropout)')
    parser.add_argument('--confidence-thresh', type=float, default=0.50, help='Minimum confidence threshold for binder classification')
    parser.add_argument('--uncertainty-thresh', type=float, default=0.15, help='Maximum allowed uncertainty std dev for binder classification')

    parser.add_argument('--dock', action='store_true', help='Automatically dock predicted cofactor into identified binding pocket')
    parser.add_argument('--force-dock', action='store_true', help='Force molecular docking even if classified as non-binder')
    parser.add_argument('--pocket-center', nargs=3, type=float, default=None, help='Manual pocket center coordinates x y z (optional)')
    parser.add_argument('--dock-out', type=str, default='docking_results', help='Directory for docking output files')
    args = parser.parse_args()

    # Default checkpoint path selection
    if args.checkpoint is None:
        if args.model_type in ['self_attention_mil', 'self_att']:
            args.checkpoint = 'self_attention_mil_best.pt'
        elif os.path.exists('ligand_cross_mil_best.pt'):
            args.checkpoint = 'ligand_cross_mil_best.pt'
        elif os.path.exists('self_attention_mil_best.pt'):
            args.checkpoint = 'self_attention_mil_best.pt'
        else:
            args.checkpoint = 'ligand_cross_mil_best.pt'

    predictor = AMICOPredictor(checkpoint_path=args.checkpoint, config_json=args.config, model_type=args.model_type)

    if args.pdb or args.p2rank_dir:
        # === END-TO-END MODE FROM PDB OR P2RANK OUTPUTS ===
        pdb_file = args.pdb
        if not pdb_file and args.p2rank_dir:
            pdb_candidates = list(Path(args.p2rank_dir).parent.glob("*.pdb"))
            if pdb_candidates:
                pdb_file = str(pdb_candidates[0])
            else:
                raise ValueError("Specified --p2rank-dir without a corresponding --pdb file.")

        print("\n" + "=" * 65)
        print(f"  Starting End-to-End Prediction Pipeline: {Path(pdb_file).name}")
        print("=" * 65)

        res = predictor.predict_from_pdb(
            pdb_path=pdb_file,
            prank_exec=args.prank,
            prank_out_dir=args.p2rank_dir,
            min_prob=args.min_prob,
            esm_model=args.esm_model,
            mc_samples=args.mc_samples,
            confidence_threshold=args.confidence_thresh,
            uncertainty_threshold=args.uncertainty_thresh
        )
        prot_id = Path(pdb_file).stem
    else:
        # === DEMO MODE (Synthetic input) ===
        print("\n[INFO] No --pdb provided. Running demo inference with synthetic tensors...")
        dummy_pockets = torch.randn(3, 1280)
        dummy_full_prot = torch.randn(1280)

        res = predictor.predict(
            dummy_pockets, 
            dummy_full_prot, 
            mc_samples=args.mc_samples,
            confidence_threshold=args.confidence_thresh,
            uncertainty_threshold=args.uncertainty_thresh
        )
        prot_id = "sample_protein_demo"

    # Display results
    print("\n" + "=" * 65)
    print(f"            AMICO INFERENCE RESULTS: {prot_id}")
    print("=" * 65)
    print(f"Binding Status:          {res['binding_status']}")
    print(f"Predicted Cofactor:      {res['predicted_cofactor']}")
    print(f"Mean Confidence:         {res['confidence'] * 100:.2f} %")
    if args.mc_samples > 1:
        print(f"Uncertainty (MC std):    ±{res['uncertainty_std'] * 100:.2f} % (Samples: {res['mc_samples_used']})")
        print(f"Predictive Entropy:      {res['predictive_entropy']:.4f}")
    
    if res['is_confident_prediction']:
        print(f"\nBinding Pocket Localization:")
        if res.get('best_p2rank_pocket_id') is not None:
            cx, cy, cz = res['best_p2rank_pocket_center']
            print(f" -> Top P2Rank Pocket (Docking Site): Pocket #{res['best_p2rank_pocket_id']} ({res['best_p2rank_pocket_name']}) | Score: {res['best_p2rank_score']:.2f}, Prob: {res['best_p2rank_prob']:.2f}")
            print(f"    3D Coordinates:                  [x={cx:.2f}, y={cy:.2f}, z={cz:.2f}]")
        print(f" -> Model Attention Focus:           Pocket #{res['best_binding_pocket']} (Attention: {res['best_pocket_attention']:.4f})")
        print(f" -> Global Enzyme Influence:         {res['global_context_weight']:.4f}")
    else:
        print("\n⚠️ Model classified protein as NON-BINDER (True Negative) or prediction is below confidence threshold.")

    print("\nCofactor Probabilities:")
    for name, p_data in res['probabilities'].items():
        if args.mc_samples > 1:
            mean_p = p_data['mean_probability'] * 100
            std_p = p_data['uncertainty_std'] * 100
            print(f" - {name:<12}: {mean_p:6.2f} %  (±{std_p:4.2f} %)")
        else:
            p = p_data['probability'] * 100
            print(f" - {name:<12}: {p:6.2f} %")

    print("\nPocket Compatibility Rankings:")
    for p_info in res['pocket_rankings']:
        p_str = f" - Pocket #{p_info['pocket_id']:<2}: Attention = {p_info['attention_score']:.4f}"
        if 'p2rank_prob' in p_info:
            p_str += f" | P2Rank Prob: {p_info['p2rank_prob']:.2f} | Residues: {p_info['residue_count']}"
        print(p_str)

    # Molecular docking if requested (into best P2Rank pocket)
    if args.dock or args.force_dock:
        if not res['is_confident_prediction'] and not args.force_dock:
            print("\n[DOCKING SKIPPED] Docking skipped: protein evaluated as non-binder (use --force-dock to override).")
        else:
            from docking_utils import dock_predicted_cofactor

            pred_cofactor = res['raw_top_class']
            pdb_path = args.pdb if args.pdb else "sample_protein.pdb"
            
            # Prioritize top P2Rank pocket for molecular docking
            center = None
            docking_site_label = ""
            if args.pocket_center:
                center = np.array(args.pocket_center)
                docking_site_label = f"manual coordinates [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]"
            elif 'best_p2rank_pocket_center' in res and res['best_p2rank_pocket_center'] and sum(abs(x) for x in res['best_p2rank_pocket_center']) > 1e-4:
                center = np.array(res['best_p2rank_pocket_center'])
                docking_site_label = f"top P2Rank Pocket #{res.get('best_p2rank_pocket_id', 1)} ({res.get('best_p2rank_pocket_name', 'pocket1')}, score={res.get('best_p2rank_score', 0.0):.2f})"
            elif 'best_pocket_center' in res and res['best_pocket_center'] and sum(abs(x) for x in res['best_pocket_center']) > 1e-4:
                center = np.array(res['best_pocket_center'])
                docking_site_label = f"attention focus Pocket #{res.get('best_binding_pocket', 1)}"
            else:
                from docking_utils import get_pocket_center_from_pdb
                center = get_pocket_center_from_pdb(pdb_path) if os.path.exists(pdb_path) else np.array([0.0, 0.0, 0.0])
                docking_site_label = "protein center of mass fallback"

            print(f"\n-> Launching molecular docking for {pred_cofactor} into {docking_site_label} at center [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}] Å...")
            dock_res = dock_predicted_cofactor(
                protein_pdb=pdb_path,
                cofactor_name=pred_cofactor,
                pocket_center=center,
                p2rank_dir=res.get('p2rank_output_dir'),
                out_dir=args.dock_out
            )
            print(f"✨ Docking completed. Output files stored in: {dock_res['output_dir']}/")


if __name__ == '__main__':
    main()
