import os
import argparse
import json
from pathlib import Path
import torch
import numpy as np

from model import LigandCrossAttentionMIL, TARGET_NAMES
from p2rank_utils import run_p2rank, parse_p2rank_output, find_p2rank_executable

class AMICOPredictor:
    """
    Inferenční třída pro LigandCrossAttentionMIL.
    Umožňuje predikovat kofaktor, lokalizovat vazebnou kapsu, odhadovat epistemickou nejistotu 
    pomocí MC Dropoutu a provádět end-to-end inferenci přímo z PDB souboru (P2Rank + ESM-2).
    """
    def __init__(self, checkpoint_path=None, config_json=None, device=None):
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

        self.model = LigandCrossAttentionMIL(
            feature_dim=1280,
            ecfp_dim=1024,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_classes=5,
            dropout=dropout
        ).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state)
            print(f"Checkpoint načten z {checkpoint_path}")
        else:
            print("Upozornění: Model inicializován bez načtení checkpointu.")

        self.model.eval()
        self._esm_extractor = None

    def _get_esm_extractor(self, model_name="facebook/esm2_t33_650M_UR50D"):
        """Líná inicializace ESM-2 extraktoru příznaků."""
        if self._esm_extractor is None:
            from esm_extractor import ESMFeatureExtractor
            self._esm_extractor = ESMFeatureExtractor(model_name=model_name, device=self.device)
        return self._esm_extractor

    def _enable_mc_dropout(self):
        """Ponechá model v eval módu, ale aktivuje Dropout vrstvy pro MC vzorkování."""
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def predict(self, pocket_features, full_protein_feature, mc_samples=30, 
                confidence_threshold=0.50, uncertainty_threshold=0.15):
        """
        Provede inferenci s volitelným Monte Carlo Dropout vzorkováním.
        
        Args:
            pocket_features: Tensor [N_pockets, 1280] nebo np.ndarray
            full_protein_feature: Tensor [1280] nebo np.ndarray
            mc_samples: Počet stochastických průchodů (1 = standardní deterministický režim, >=20 pro MC Dropout)
            confidence_threshold: Minimální průměrná pravděpodobnost pro potvrzení vazby kofaktoru
            uncertainty_threshold: Maximální povolená směrodatná odchylka (rozptyl) pro jistou predikci

        Returns:
            dict s kompletními výsledky predikce, nejistoty a detekce True Negatives
        """
        if isinstance(pocket_features, np.ndarray):
            pocket_features = torch.FloatTensor(pocket_features)
        if isinstance(full_protein_feature, np.ndarray):
            full_protein_feature = torch.FloatTensor(full_protein_feature)

        if pocket_features.dim() == 1 and pocket_features.numel() == 0:
            pocket_features = pocket_features.view(0, 1280)
        elif pocket_features.dim() == 2:
            pocket_features = pocket_features.unsqueeze(0) # [1, N, 1280]
        
        if full_protein_feature.dim() == 1:
            full_protein_feature = full_protein_feature.unsqueeze(0) # [1, 1280]

        pocket_features = pocket_features.to(self.device)
        full_protein_feature = full_protein_feature.to(self.device)
        
        num_pockets = pocket_features.size(1)
        padding_mask = torch.zeros(1, num_pockets, dtype=torch.bool, device=self.device)

        if mc_samples > 1:
            # === MONTE CARLO DROPOUT REŽIM ===
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

            all_probs = np.stack(all_probs, axis=0) # [T, 5]
            all_attns = np.stack(all_attns, axis=0) # [T, 5, N+1]

            mean_probs = np.mean(all_probs, axis=0) # [5]
            std_probs = np.std(all_probs, axis=0)   # [5] (Epistemická nejistota)
            mean_attn = np.mean(all_attns, axis=0)  # [5, N+1]
            
            # Prediktivní entropie: H = - sum(p * log(p))
            predictive_entropy = -float(np.sum(mean_probs * np.log(mean_probs + 1e-12)))

            pred_idx = int(np.argmax(mean_probs))
            pred_label = TARGET_NAMES[pred_idx]
            confidence = float(mean_probs[pred_idx])
            uncertainty = float(std_probs[pred_idx])
            
            # Detekce True Negatives (Nevazačů / Out-of-Distribution proteinů)
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
            # === DETERMINISTICKÝ REŽIM ===
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

        # Interpretace vazebných kapes
        cofactor_attn = used_attn[pred_idx]
        global_context_weight = float(cofactor_attn[0])
        pocket_attn_weights = cofactor_attn[1:]

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
            "best_pocket_attention": round(best_pocket_weight, 4),
            "global_context_weight": round(global_context_weight, 4),
            "pocket_rankings": pocket_rankings
        }

    def predict_from_pdb(self, pdb_path, prank_exec=None, prank_out_dir=None, min_prob=0.0,
                         esm_model="facebook/esm2_t33_650M_UR50D", mc_samples=30,
                         confidence_threshold=0.50, uncertainty_threshold=0.15):
        """
        End-to-End predikce z PDB struktury:
        1. Spuštění P2Ranku (nebo načtení existující složky)
        2. Extrakce sekvencí a 3D souřadnic kapes
        3. Výpočet ESM-2 embeddingů (kapsy + globální kontext proteinu)
        4. LigandCrossAttentionMIL inference s MC Dropoutem
        5. Mapování pozornosti na fyzické souřadnice kapes pro dokování
        """
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB soubor nebyl nalezen: {pdb_path}")

        # 1. P2Rank
        if prank_out_dir and Path(prank_out_dir).exists():
            out_dir = Path(prank_out_dir)
            print(f"-> Používám existující výstupy P2Ranku z {out_dir}")
        else:
            out_dir = run_p2rank(pdb_path, prank_exec=prank_exec)

        # 2. Parsování kapes
        print(f"-> Analyzuji nalezené kapsy z {out_dir}...")
        parsed_data = parse_p2rank_output(out_dir, pdb_path, min_prob=min_prob)
        pockets = parsed_data['pockets']
        print(f"-> Nalezeno {len(pockets)} vhodných kapes (min_prob >= {min_prob}).")

        # 3. ESM-2 extrakce
        extractor = self._get_esm_extractor(model_name=esm_model)
        print("-> Generuji ESM-2 embeddingy...")
        pocket_features, full_protein_feature = extractor.extract_all_from_parsed(parsed_data)

        # 4. AMICO Model Inference
        print("-> Spouštím model LigandCrossAttentionMIL...")
        res = self.predict(
            pocket_features=pocket_features,
            full_protein_feature=full_protein_feature,
            mc_samples=mc_samples,
            confidence_threshold=confidence_threshold,
            uncertainty_threshold=uncertainty_threshold
        )

        # 5. Obohacení výsledků o metadata z P2Ranku
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

        best_pid = res['best_binding_pocket']
        if best_pid and best_pid in pocket_map:
            res['best_pocket_center'] = pocket_map[best_pid]['center']
            res['best_pocket_name'] = pocket_map[best_pid]['name']
        else:
            res['best_pocket_center'] = [0.0, 0.0, 0.0]
            res['best_pocket_name'] = None

        return res


def main():
    parser = argparse.ArgumentParser(description="AMICO: End-to-End P2Rank + ESM-2 + Ligand Cross-Attention Inference & Dokování")
    parser.add_argument('--pdb', type=str, default=None, help='Cesta k PDB souboru proteinu pro end-to-end predikci')
    parser.add_argument('--prank', type=str, default=None, help='Cesta ke spustitelnému souboru P2Rank (výchozí: hledá v projektu/PATH)')
    parser.add_argument('--p2rank-dir', type=str, default=None, help='Cesta k již hotové složce s výstupy P2Ranku (přeskočí běh P2Ranku)')
    parser.add_argument('--min-prob', type=float, default=0.0, help='Minimální pravděpodobnost kapsy z P2Ranku (0.0 = všechny)')
    parser.add_argument('--esm-model', type=str, default='facebook/esm2_t33_650M_UR50D', help='Model ESM-2 z HuggingFace')

    parser.add_argument('--checkpoint', type=str, default='ligand_cross_mil_best.pt', help='Cesta k vahám modelu')
    parser.add_argument('--config', type=str, default=None, help='Cesta ke konfiguraci JSON (z Optuna tuningu)')
    parser.add_argument('--mc-samples', type=int, default=30, help='Počet MC Dropout vzorků (1 = deterministický, 30 = MC Dropout)')
    parser.add_argument('--confidence-thresh', type=float, default=0.50, help='Minimální jistota pro klasifikaci jako vazač')
    parser.add_argument('--uncertainty-thresh', type=float, default=0.15, help='Maximální rozptyl pro klasifikaci jako vazač')

    parser.add_argument('--dock', action='store_true', help='Automaticky nadokovat předpovězený kofaktor do identifikované kapsy')
    parser.add_argument('--pocket-center', nargs=3, type=float, default=None, help='Manuální souřadnice středu kapsy x y z (volitelné)')
    parser.add_argument('--dock-out', type=str, default='docking_results', help='Složka pro uložení výsledků dokování')
    args = parser.parse_args()

    predictor = AMICOPredictor(checkpoint_path=args.checkpoint, config_json=args.config)

    if args.pdb or args.p2rank_dir:
        # === END-TO-END REŽIM Z PDB NEBO P2RANK VÝSTUPŮ ===
        pdb_file = args.pdb
        if not pdb_file and args.p2rank_dir:
            # Zkusíme najít PDB soubor v okolí p2rank_dir
            pdb_candidates = list(Path(args.p2rank_dir).parent.glob("*.pdb"))
            if pdb_candidates:
                pdb_file = str(pdb_candidates[0])
            else:
                raise ValueError("Byl zadán --p2rank-dir, ale nebyl zadán odpovídající --pdb soubor.")

        print(f"\n=======================================================")
        print(f"  Spouštím End-to-End Pipeline: {Path(pdb_file).name}")
        print(f"=======================================================")

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
        # === DEMO / MOCK REŽIM (Bez PDB souboru) ===
        print("\n[INFO] Nebyl zadán parametr --pdb. Spouštím ukázkovou predikci na mock datech...")
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

    # Výpis výsledků
    print("\n" + "="*65)
    print(f"            VÝSLEDEK INFERENCE: {prot_id}")
    print("="*65)
    print(f"Status vazby:            {res['binding_status']}")
    print(f"Předpovězený kofaktor:   {res['predicted_cofactor']}")
    print(f"Průměrná jistota (mean): {res['confidence']*100:.2f} %")
    if args.mc_samples > 1:
        print(f"Nejistota (MC std dev):  ±{res['uncertainty_std']*100:.2f} % (vzorků: {res['mc_samples_used']})")
        print(f"Prediktivní entropie:    {res['predictive_entropy']:.4f}")
    
    if res['is_confident_prediction']:
        print(f"\nLokalizace vazebného místa:")
        print(f" -> Nejlepší kapsa:      Pocket #{res['best_binding_pocket']} (Attention: {res['best_pocket_attention']:.4f})")
        if 'best_pocket_center' in res:
            cx, cy, cz = res['best_pocket_center']
            print(f" -> 3D Střed kapsy:      [x={cx:.2f}, y={cy:.2f}, z={cz:.2f}]")
        print(f" -> Vliv celého enzymu:  {res['global_context_weight']:.4f}")
    else:
        print("\n⚠️ Model vyhodnotil protein jako NE-VAZAČ (True Negative) nebo je predikce příliš nejistá.")

    print("\nPravděpodobnosti jednotlivých kofaktorů:")
    for name, p_data in res['probabilities'].items():
        if args.mc_samples > 1:
            mean_p = p_data['mean_probability'] * 100
            std_p = p_data['uncertainty_std'] * 100
            print(f" - {name:<12}: {mean_p:6.2f} %  (±{std_p:4.2f} %)")
        else:
            p = p_data['probability'] * 100
            print(f" - {name:<12}: {p:6.2f} %")

    print("\nPořadí kapes podle chemické kompatibility:")
    for p_info in res['pocket_rankings']:
        p_str = f" - Kapsa #{p_info['pocket_id']:<2}: Attention = {p_info['attention_score']:.4f}"
        if 'p2rank_prob' in p_info:
            p_str += f" | P2Rank Prob: {p_info['p2rank_prob']:.2f} | Reziduí: {p_info['residue_count']}"
        print(p_str)

    # Dokování předpovězeného kofaktoru, pokud je vyžádáno
    if args.dock:
        if not res['is_confident_prediction']:
            print("\n[DOCKING SKIP] Dokování přeskočeno: protein byl vyhodnocen jako nevazač.")
        else:
            from docking_utils import dock_predicted_cofactor

            pred_cofactor = res['raw_top_class']
            pdb_path = args.pdb if args.pdb else "sample_protein.pdb"
            
            if args.pocket_center:
                center = np.array(args.pocket_center)
            elif 'best_pocket_center' in res and sum(abs(x) for x in res['best_pocket_center']) > 1e-4:
                center = np.array(res['best_pocket_center'])
            else:
                from docking_utils import get_pocket_center_from_pdb
                center = get_pocket_center_from_pdb(pdb_path) if os.path.exists(pdb_path) else np.array([0.0, 0.0, 0.0])

            print(f"\n-> Spouštím molekulární dokování ({pred_cofactor}) do středu [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]...")
            dock_res = dock_predicted_cofactor(
                protein_pdb=pdb_path,
                cofactor_name=pred_cofactor,
                pocket_center=center,
                out_dir=args.dock_out
            )
            print(f"✨ Dokování dokončeno. Výstupní soubory uloženy do: {dock_res['output_dir']}/")


if __name__ == '__main__':
    main()
