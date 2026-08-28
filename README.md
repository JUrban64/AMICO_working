# AMICO (Final Model & Deployment)
**Ligand-Protein Cross-Attention Multi-Instance Learning for Cofactor Specificity Prediction**

---

## 🌟 Overview

AMICO predicts specific cofactor binding (`ATP`, `NAD`, `FAD`, `B12`, `acetyl-CoA`) for protein structures under strict **zero-shot cross-fold structural generalization**.

This repository contains the standalone, production-ready implementation of the winning **`LigandCrossAttentionMIL`** model.

### 🧬 Core Architectural Features:
1. **End-to-End P2Rank Pocket Detection & Extraction:** Directly accepts `.pdb` structures, detects 3D pockets, and extracts pocket residue sequences + 3D coordinates.
2. **ESM-2 Multi-Instance & Global Context:** Computes per-pocket embeddings ($[N_{\text{pockets}}, 1280]$) alongside a global whole-protein context anchor on token index 0 ($[1280]$).
3. **Morgan ECFP4 Chemical Fingerprints:** 1024-bit cofactor fingerprints act as queries cross-attending over both protein context and candidate pockets.
4. **Bayesian Monte Carlo Dropout:** $T=30$ stochastic passes quantify epistemic uncertainty (std dev) and detect True Negatives (non-binders or out-of-distribution enzymes).
5. **Direct Pocket Localization & Docking:** Cross-attention weights highlight the exact functional pocket and automatically guide **AutoDock Vina** docking into the pocket center.

---

## 📁 Repository Structure

```text
AMICOfin/
├── PROJECT_KNOWLEDGE_SUMMARY.md     # Full scientific documentation & findings
├── README.md                        # Usage guide & documentation
├── requirements.txt                 # Dependencies (torch, transformers, rdkit, etc.)
│
├── model.py                         # ⭐ LigandCrossAttentionMIL neural architecture
├── dataset.py                       # Pockets + Full Protein data loader & collator
├── train.py                         # Full training script with Optuna JSON config support
├── predict.py                       # Production End-to-End inference CLI & API
├── p2rank_utils.py                  # P2Rank runner & pocket CSV parser
├── esm_extractor.py                 # ESM-2 feature extractor (GPU/MPS/CPU)
├── docking_utils.py                 # AutoDock Vina molecular docking pipeline
├── tune_optuna.py                   # Automated Bayesian hyperparameter search (SQLite)
│
└── data_prep/
    ├── two_level_structure_clustering.py # StratifiedGroupKFold cross-fold splitting pipeline
    ├── structure_clustering.py           # Structure clustering
    └── generate_full_protein_embeddings.py # ESM-2 sequence embedding generator
```

---

## 🚀 Quickstart Guide

### 1. End-to-End Prediction from a PDB File (P2Rank + ESM-2 + AMICO)
```bash
# Spuštění kompletní pipeline na novém PDB souboru (s MC Dropoutem):
python predict.py \
    --pdb /path/to/my_protein.pdb \
    --checkpoint ligand_cross_mil_best.pt \
    --mc-samples 30

# End-to-End predikce + automatické dokování do lokalizované kapsy:
python predict.py \
    --pdb /path/to/my_protein.pdb \
    --checkpoint ligand_cross_mil_best.pt \
    --mc-samples 30 \
    --dock \
    --dock-out ./my_docking_results
```

### 2. Trénování Modelu
```bash
# Standardní trénování na zadaném splitu:
python train.py --split-suffix mil_0.5 --epochs 50

# Trénování s nejlepšími parametry z Optuna tuningu:
python train.py --config-json best_params_ligand_cross_mil_0.5.json --epochs 50
```

### 3. Hyperparameter Optimization (Optuna)
```bash
python tune_optuna.py --split-suffix mil_0.5 --n-trials 50
```

---

## 💻 Python API Example

### A. End-to-End z PDB souboru
```python
from predict import AMICOPredictor

predictor = AMICOPredictor(checkpoint_path="ligand_cross_mil_best.pt")

# Automaticky spustí P2Rank, ESM-2 a LigandCrossAttentionMIL
result = predictor.predict_from_pdb(
    pdb_path="alphafold_structure.pdb",
    mc_samples=30
)

print(f"Predicted Cofactor:  {result['predicted_cofactor']}")
print(f"Confidence:          {result['confidence'] * 100:.2f} %")
print(f"Uncertainty (std):   ±{result['uncertainty_std'] * 100:.2f} %")
print(f"Best Binding Pocket: Pocket #{result['best_binding_pocket']}")
print(f"3D Pocket Center:    {result['best_pocket_center']}")
```

### B. Přímá inference z embeddingů
```python
from predict import AMICOPredictor
import torch

predictor = AMICOPredictor(checkpoint_path="ligand_cross_mil_best.pt")

# pocket_features [N_pockets, 1280], full_protein [1280]
dummy_pockets = torch.randn(3, 1280)
dummy_full_prot = torch.randn(1280)

result = predictor.predict(dummy_pockets, dummy_full_prot, mc_samples=30)
print(f"Predicted: {result['predicted_cofactor']}, Status: {result['binding_status']}")
```
